#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build train-only raw ARG subtype prototypes for the full-length / long model.

Input
-----
/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding/
    Long_fusion_subtype2_newsplit/long_fusion_subtype2_newsplit.pt

The input PT is produced by fusion_long2.py. It already contains one fused
embedding per full-length sequence, so NO fragment-to-original aggregation is
performed here.

Method
------
1. Load the full-length 512D gated-fusion embeddings.
2. L2-normalize every sequence embedding.
3. Use only model_splits == "train" ARG sequences.
4. Average train embeddings within each ARG Subtype.
5. L2-normalize each subtype mean to form one raw prototype per ARG subtype.
6. Build a separate train-only nonARG centroid.
7. Evaluate ARG-only nearest-prototype cosine classification on validation/test.

The external test split is never used for prototype construction.

Output
------
Exactly one PT:
    Raw_subtype_prototypes_subtype2_newsplit/raw_ARG_subtype_prototypes_and_baseline.pt

CUDA is mandatory. No CPU fallback.
"""

from __future__ import annotations

import newsplit_contract as contract

import argparse
import math
import os
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(
    "/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding"
)

DEFAULT_INPUT_PT = (
    ROOT
    / "Long_fusion_subtype2_newsplit"
    / "long_fusion_subtype2_newsplit.pt"
)

DEFAULT_OUTPUT_PT = (
    ROOT
    / "Raw_subtype_prototypes_subtype2_newsplit"
    / "raw_ARG_subtype_prototypes_and_baseline.pt"
)

SCRIPT_VERSION = (
    "2026-08-22-long-raw-ARG-subtype-prototype-baseline-v1"
)

NONARG = "nonARG"


# =============================================================================
# Arguments
# =============================================================================

SCRIPT_VERSION = SCRIPT_VERSION + "-unified-newsplit-v1"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build train-only raw ARG subtype prototypes "
            "from full-length gated-fusion embeddings."
        )
    )

    parser.add_argument(
        "--input-pt",
        type=Path,
        default=DEFAULT_INPUT_PT,
    )

    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# Heartbeat
# =============================================================================

class Heartbeat:

    def __init__(self, interval: int):

        self.interval = max(
            5,
            int(interval),
        )

        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None

        self.phase = "startup"
        self.note = ""

    def update(self, phase=None, note=None):

        with self.lock:

            if phase is not None:
                self.phase = str(phase)

            if note is not None:
                self.note = str(note)

    def emit(self):

        with self.lock:
            phase = self.phase
            note = self.note

        try:

            allocated = (
                torch.cuda.memory_allocated(0)
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved(0)
                / (1024 ** 3)
            )

            gpu_text = (
                f"GPU_alloc={allocated:.2f}GB | "
                f"GPU_reserved={reserved:.2f}GB"
            )

        except Exception:

            gpu_text = "GPU_mem=NA"

        print(
            f"[HEARTBEAT] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"phase={phase} | "
            f"{gpu_text}"
            f"{' | ' + note if note else ''}",
            flush=True,
        )

    def _loop(self):

        while not self.stop_event.wait(
            self.interval
        ):
            self.emit()

    def start(self):

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )

        self.thread.start()

        self.emit()

    def stop(self):

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(
                timeout=5,
            )

        self.emit()


# =============================================================================
# Utilities
# =============================================================================

def canonical_id(value):

    text = str(value).strip()

    return (
        text.split()[0]
        if text
        else ""
    )


def is_nonarg(value):

    compact = "".join(
        character
        for character
        in str(value).lower()
        if character.isalnum()
    )

    return compact in {
        "nonarg",
        "nonresistance",
        "negative",
        "nonresistant",
    }


def torch_load_cpu(path: Path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu",
        )


def safe_save(obj, path: Path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    torch.save(
        obj,
        temporary,
    )

    if (
        not temporary.is_file()
        or temporary.stat().st_size <= 0
    ):

        raise RuntimeError(
            f"Temporary output is invalid: {temporary}"
        )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# Input validation
# =============================================================================

def load_and_validate_input(path: Path):

    obj = torch_load_cpu(
        path
    )
    contract.validate_fusion(obj)


    if not isinstance(
        obj,
        dict,
    ):

        raise TypeError(
            f"Input PT must contain a dictionary: {path}"
        )

    required = {
        "ids",
        "embeddings",
        "splits",
        "model_splits",
        "subtypes",
    }

    missing = sorted(
        required
        - set(obj)
    )

    if missing:

        raise KeyError(
            f"Input PT is missing keys: {missing}"
        )

    ids = [
        canonical_id(value)
        for value
        in obj["ids"]
    ]

    subtypes = [
        str(value).strip()
        for value
        in obj["subtypes"]
    ]

    external_splits = [
        str(value).strip().lower()
        for value
        in obj["splits"]
    ]

    model_splits = [
        str(value).strip().lower()
        for value
        in obj["model_splits"]
    ]

    embeddings = obj["embeddings"]

    number = len(ids)

    for name, values in {
        "subtypes":
            subtypes,

        "splits":
            external_splits,

        "model_splits":
            model_splits,
    }.items():

        if len(values) != number:

            raise ValueError(
                f"Length mismatch: "
                f"ids={number}, "
                f"{name}={len(values)}"
            )

    if any(
        not value
        for value
        in ids
    ):

        raise ValueError(
            "Empty ID detected."
        )

    if any(
        not value
        for value
        in subtypes
    ):

        raise ValueError(
            "Empty Subtype detected."
        )

    if len(ids) != len(set(ids)):

        raise ValueError(
            "Duplicate sequence IDs detected in input PT."
        )

    invalid_external = (
        set(external_splits)
        - {
            "train",
            "test",
        }
    )

    if invalid_external:

        raise ValueError(
            f"Invalid external splits: "
            f"{sorted(invalid_external)}"
        )

    invalid_model = (
        set(model_splits)
        - {
            "train",
            "validation",
            "test",
        }
    )

    if invalid_model:

        raise ValueError(
            f"Invalid model splits: "
            f"{sorted(invalid_model)}"
        )

    if (
        not isinstance(
            embeddings,
            torch.Tensor,
        )
        or embeddings.ndim != 2
    ):

        raise ValueError(
            "Input embeddings must be a 2D torch.Tensor."
        )

    if embeddings.shape[0] != number:

        raise ValueError(
            f"Embedding row mismatch: "
            f"rows={embeddings.shape[0]}, "
            f"IDs={number}"
        )

    embeddings = (
        embeddings
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    if not bool(
        torch.isfinite(
            embeddings
        ).all()
    ):

        raise ValueError(
            "Non-finite fused embeddings detected."
        )

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=1,
    )

    if bool(
        (
            norms
            <= 1e-12
        ).any()
    ):

        bad = torch.where(
            norms
            <= 1e-12
        )[0][:10].tolist()

        raise ValueError(
            f"Zero fused embeddings detected: rows={bad}"
        )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    ).contiguous()

    if not any(
        is_nonarg(value)
        for value
        in subtypes
    ):

        raise ValueError(
            "Cannot identify the nonARG class."
        )

    print(
        f"Input sequences        : "
        f"{number:,}",
        flush=True,
    )

    print(
        f"Input embedding shape  : "
        f"{tuple(embeddings.shape)}",
        flush=True,
    )

    print(
        f"Input subtype classes  : "
        f"{len(set(subtypes)):,}",
        flush=True,
    )

    for split in (
        "train",
        "validation",
        "test",
    ):

        count = sum(
            value == split
            for value
            in model_splits
        )

        unique_subtypes = len(
            {
                subtype
                for subtype, value
                in zip(
                    subtypes,
                    model_splits,
                )
                if value == split
            }
        )

        print(
            f"{split:10s} records      : "
            f"{count:,} | "
            f"subtypes={unique_subtypes:,}",
            flush=True,
        )

    return (
        obj,
        ids,
        subtypes,
        external_splits,
        model_splits,
        embeddings,
    )


# =============================================================================
# Ordered subtype classes
# =============================================================================

def ordered_arg_subtypes(
    input_obj,
    observed_subtypes,
):

    label_map = input_obj.get(
        "subtype_to_index"
    )

    if isinstance(
        label_map,
        dict,
    ):

        ordered = [
            str(label)
            for label, _
            in sorted(
                label_map.items(),
                key=lambda item:
                    int(item[1]),
            )
            if not is_nonarg(
                label
            )
        ]

    else:

        ordered = sorted(
            {
                label
                for label
                in observed_subtypes
                if not is_nonarg(
                    label
                )
            }
        )

    observed_arg = {
        label
        for label
        in observed_subtypes
        if not is_nonarg(
            label
        )
    }

    if set(ordered) != observed_arg:

        missing = sorted(
            observed_arg
            - set(ordered)
        )

        extra = sorted(
            set(ordered)
            - observed_arg
        )

        raise ValueError(
            "ARG subtype label-map mismatch: "
            f"missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )

    return ordered


# =============================================================================
# Prototype construction
# =============================================================================

def build_train_only_prototypes(
    input_obj,
    subtypes,
    model_splits,
    embeddings,
):

    arg_subtypes = ordered_arg_subtypes(
        input_obj,
        subtypes,
    )

    subtype_to_index = {
        label:
            index
        for index, label
        in enumerate(
            arg_subtypes
        )
    }

    number_subtypes = len(
        arg_subtypes
    )

    embedding_dim = int(
        embeddings.shape[
            1
        ]
    )

    train_arg_indices = [
        index
        for index, (
            subtype,
            split,
        )
        in enumerate(
            zip(
                subtypes,
                model_splits,
            )
        )
        if (
            split == "train"
            and not is_nonarg(
                subtype
            )
        )
    ]

    if not train_arg_indices:

        raise ValueError(
            "No ARG train sequences available for prototypes."
        )

    index_tensor = torch.tensor(
        train_arg_indices,
        dtype=torch.long,
    )

    train_features = (
        embeddings
        .index_select(
            0,
            index_tensor,
        )
    )

    train_targets = torch.tensor(
        [
            subtype_to_index[
                subtypes[index]
            ]
            for index
            in train_arg_indices
        ],
        dtype=torch.long,
    )

    sums = torch.zeros(
        (
            number_subtypes,
            embedding_dim,
        ),
        dtype=torch.float32,
    )

    sums.index_add_(
        0,
        train_targets,
        train_features,
    )

    counts = torch.bincount(
        train_targets,
        minlength=number_subtypes,
    )

    if bool(
        (
            counts
            == 0
        ).any()
    ):

        missing_indices = torch.where(
            counts
            == 0
        )[0].tolist()

        missing_labels = [
            arg_subtypes[
                index
            ]
            for index
            in missing_indices
        ]

        raise ValueError(
            "ARG subtypes missing from model-train: "
            f"{missing_labels[:20]}"
        )

    prototypes = (
        sums
        / counts
        .to(
            torch.float32
        )
        .unsqueeze(
            1
        )
    )

    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=1,
    ).contiguous()

    nonarg_train_indices = [
        index
        for index, (
            subtype,
            split,
        )
        in enumerate(
            zip(
                subtypes,
                model_splits,
            )
        )
        if (
            split == "train"
            and is_nonarg(
                subtype
            )
        )
    ]

    if not nonarg_train_indices:

        raise ValueError(
            "No nonARG train sequences available."
        )

    nonarg_features = (
        embeddings
        .index_select(
            0,
            torch.tensor(
                nonarg_train_indices,
                dtype=torch.long,
            ),
        )
    )

    nonarg_centroid = F.normalize(
        nonarg_features.mean(
            dim=0,
            keepdim=True,
        ),
        p=2,
        dim=1,
    )[0].contiguous()

    count_values = counts.tolist()

    count_distribution = Counter(
        count_values
    )

    print(
        "\n===== TRAIN-ONLY PROTOTYPES =====",
        flush=True,
    )

    print(
        f"ARG subtype prototypes : "
        f"{tuple(prototypes.shape)}",
        flush=True,
    )

    print(
        f"ARG train sequences    : "
        f"{len(train_arg_indices):,}",
        flush=True,
    )

    print(
        f"ARG train/subtype      : "
        f"min={counts.min().item():,} | "
        f"median={statistics.median(count_values)} | "
        f"max={counts.max().item():,}",
        flush=True,
    )

    print(
        f"Singleton train prototypes: "
        f"{count_distribution.get(1, 0):,}",
        flush=True,
    )

    print(
        f"nonARG train sequences : "
        f"{len(nonarg_train_indices):,}",
        flush=True,
    )

    return {
        "arg_subtype_names":
            arg_subtypes,

        "arg_subtype_to_index":
            subtype_to_index,

        "raw_arg_subtype_prototypes":
            prototypes,

        "prototype_train_sequence_counts":
            counts,

        "nonarg_reference_centroid":
            nonarg_centroid,

        "nonarg_train_sequence_count":
            len(
                nonarg_train_indices
            ),
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    true_indices,
    predicted_indices,
    number_classes,
    top5_correct,
    maximum_similarities,
    subtype_names,
):

    true_np = (
        true_indices
        .numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    predicted_np = (
        predicted_indices
        .numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    confusion = np.zeros(
        (
            number_classes,
            number_classes,
        ),
        dtype=np.int64,
    )

    np.add.at(
        confusion,
        (
            true_np,
            predicted_np,
        ),
        1,
    )

    tp = np.diag(
        confusion
    ).astype(
        np.float64
    )

    support = (
        confusion
        .sum(
            axis=1
        )
        .astype(
            np.float64
        )
    )

    predicted_count = (
        confusion
        .sum(
            axis=0
        )
        .astype(
            np.float64
        )
    )

    precision = np.divide(
        tp,
        predicted_count,
        out=np.zeros_like(
            tp
        ),
        where=(
            predicted_count
            != 0
        ),
    )

    recall = np.divide(
        tp,
        support,
        out=np.zeros_like(
            tp
        ),
        where=(
            support
            != 0
        ),
    )

    f1 = np.divide(
        2.0
        * precision
        * recall,
        precision
        + recall,
        out=np.zeros_like(
            tp
        ),
        where=(
            precision
            + recall
        )
        != 0,
    )

    total = float(
        confusion.sum()
    )

    active = (
        support
        > 0
    )

    sample_weights = (
        support
        / total
        if total
        else np.zeros_like(
            support
        )
    )

    correct = float(
        np.trace(
            confusion
        )
    )

    predicted_marginal = (
        confusion
        .sum(
            axis=0
        )
        .astype(
            np.float64
        )
    )

    true_marginal = (
        confusion
        .sum(
            axis=1
        )
        .astype(
            np.float64
        )
    )

    numerator = (
        correct
        * total
        - float(
            np.dot(
                predicted_marginal,
                true_marginal,
            )
        )
    )

    denominator = math.sqrt(
        max(
            total
            * total
            - float(
                np.dot(
                    predicted_marginal,
                    predicted_marginal,
                )
            ),
            0.0,
        )
        * max(
            total
            * total
            - float(
                np.dot(
                    true_marginal,
                    true_marginal,
                )
            ),
            0.0,
        )
    )

    mcc = (
        numerator
        / denominator
        if denominator > 0
        else 0.0
    )

    per_subtype = {
        subtype_names[index]:
            {
                "support":
                    int(
                        support[
                            index
                        ]
                    ),

                "predicted_count":
                    int(
                        predicted_count[
                            index
                        ]
                    ),

                "precision":
                    float(
                        precision[
                            index
                        ]
                    ),

                "recall":
                    float(
                        recall[
                            index
                        ]
                    ),

                "f1":
                    float(
                        f1[
                            index
                        ]
                    ),
            }
        for index
        in range(
            number_classes
        )
    }

    metrics = {
        "support":
            int(
                total
            ),

        "evaluated_subtypes":
            int(
                active.sum()
            ),

        "top1_accuracy":
            (
                float(
                    correct
                    / total
                )
                if total
                else 0.0
            ),

        "top5_accuracy":
            float(
                top5_correct
                .float()
                .mean()
                .item()
            ),

        "macro_precision":
            (
                float(
                    precision[
                        active
                    ].mean()
                )
                if bool(
                    active.any()
                )
                else 0.0
            ),

        "macro_recall":
            (
                float(
                    recall[
                        active
                    ].mean()
                )
                if bool(
                    active.any()
                )
                else 0.0
            ),

        "macro_f1":
            (
                float(
                    f1[
                        active
                    ].mean()
                )
                if bool(
                    active.any()
                )
                else 0.0
            ),

        "weighted_precision":
            float(
                np.sum(
                    precision
                    * sample_weights
                )
            ),

        "weighted_recall":
            float(
                np.sum(
                    recall
                    * sample_weights
                )
            ),

        "weighted_f1":
            float(
                np.sum(
                    f1
                    * sample_weights
                )
            ),

        "mcc":
            float(
                mcc
            ),

        "mean_max_cosine_similarity":
            float(
                maximum_similarities
                .mean()
                .item()
            ),

        "median_max_cosine_similarity":
            float(
                maximum_similarities
                .median()
                .item()
            ),

        "per_subtype":
            per_subtype,
    }

    return (
        metrics,
        torch.from_numpy(
            confusion.astype(
                np.int32,
                copy=False,
            )
        ),
    )


# =============================================================================
# Nearest prototype evaluation
# =============================================================================

@torch.inference_mode()
def nearest_prototype_evaluation(
    embeddings,
    labels,
    record_ids,
    selected_indices,
    prototypes,
    subtype_names,
    subtype_to_index,
    device,
    batch_size,
    hb,
    split_name,
):

    arg_indices = [
        index
        for index
        in selected_indices
        if not is_nonarg(
            labels[
                index
            ]
        )
    ]

    excluded_nonarg = (
        len(
            selected_indices
        )
        - len(
            arg_indices
        )
    )

    if not arg_indices:

        raise ValueError(
            f"{split_name} has no ARG records."
        )

    index_tensor = torch.tensor(
        arg_indices,
        dtype=torch.long,
    )

    selected_features = (
        embeddings
        .index_select(
            0,
            index_tensor,
        )
    )

    true_indices = torch.tensor(
        [
            subtype_to_index[
                labels[
                    index
                ]
            ]
            for index
            in arg_indices
        ],
        dtype=torch.long,
    )

    selected_ids = [
        record_ids[
            index
        ]
        for index
        in arg_indices
    ]

    prototype_device = (
        prototypes
        .to(
            device,
            non_blocking=True,
        )
    )

    predicted_parts = []
    max_similarity_parts = []
    top5_parts = []

    top_k = min(
        5,
        prototypes.shape[
            0
        ],
    )

    total_batches = math.ceil(
        len(
            arg_indices
        )
        / batch_size
    )

    for batch_no, start in enumerate(
        range(
            0,
            len(
                arg_indices
            ),
            batch_size,
        ),
        start=1,
    ):

        end = min(
            start
            + batch_size,
            len(
                arg_indices
            ),
        )

        hb.update(
            phase=(
                f"evaluate_{split_name}"
            ),
            note=(
                f"batch={batch_no}/{total_batches} | "
                f"ARG_records={start:,}-{end:,}"
            ),
        )

        batch = (
            selected_features[
                start:
                end
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        similarity = (
            batch
            @ prototype_device.T
        )

        values, indices = (
            similarity.topk(
                k=top_k,
                dim=1,
            )
        )

        true_batch = (
            true_indices[
                start:
                end
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        predicted_parts.append(
            indices[
                :,
                0
            ]
            .cpu()
        )

        max_similarity_parts.append(
            values[
                :,
                0
            ]
            .float()
            .cpu()
        )

        top5_parts.append(
            (
                indices
                == true_batch.unsqueeze(
                    1
                )
            )
            .any(
                dim=1
            )
            .cpu()
        )

    predicted = torch.cat(
        predicted_parts
    ).long()

    max_similarity = torch.cat(
        max_similarity_parts
    ).float()

    top5_correct = torch.cat(
        top5_parts
    ).bool()

    metrics, confusion = compute_metrics(
        true_indices,
        predicted,
        prototypes.shape[
            0
        ],
        top5_correct,
        max_similarity,
        subtype_names,
    )

    return {
        "metrics":
            metrics,

        "evaluated_ids":
            selected_ids,

        "true_subtype_indices":
            true_indices,

        "predicted_subtype_indices":
            predicted,

        "maximum_cosine_similarities":
            max_similarity,

        "top5_correct":
            top5_correct,

        "confusion_matrix":
            confusion,

        "excluded_nonarg_records":
            excluded_nonarg,
    }


def print_result(
    name,
    result,
):

    metrics = result[
        "metrics"
    ]

    print(
        f"{name:12s} | "
        f"n={metrics['support']:,} | "
        f"subtypes={metrics['evaluated_subtypes']:,} | "
        f"ACC={metrics['top1_accuracy']:.6f} | "
        f"Top5={metrics['top5_accuracy']:.6f} | "
        f"macro_P={metrics['macro_precision']:.6f} | "
        f"macro_R={metrics['macro_recall']:.6f} | "
        f"macro_F1={metrics['macro_f1']:.6f} | "
        f"weighted_F1={metrics['weighted_f1']:.6f} | "
        f"MCC={metrics['mcc']:.6f}",
        flush=True,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()
    print("Device:", "cuda" if torch.cuda.is_available() else "CPU (CUDA required)", flush=True)
    print("Resolved arguments:", vars(args), flush=True)

    args.input_pt = (
        args.input_pt
        .expanduser()
        .resolve()
    )

    args.output_pt = (
        args.output_pt
        .expanduser()
        .resolve()
    )

    if not args.input_pt.is_file():

        raise FileNotFoundError(
            f"Input gated-fusion PT not found: "
            f"{args.input_pt}"
        )

    if (
        args.output_pt.exists()
        and not args.overwrite
    ):

        raise FileExistsError(
            f"Output already exists: "
            f"{args.output_pt}\n"
            "Use --overwrite to replace it."
        )

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be positive."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable. "
            "This script does not fall back to CPU."
        )

    torch.cuda.set_device(
        0
    )

    device = torch.device(
        "cuda:0"
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):

        torch.set_float32_matmul_precision(
            "high"
        )

    args.output_pt.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hb = Heartbeat(
        args.log_interval
    )

    hb.start()

    started = time.time()

    try:

        print(
            "=" * 120,
            flush=True,
        )

        print(
            "LONG / FULL-LENGTH TRAIN-ONLY RAW ARG SUBTYPE PROTOTYPES",
            flush=True,
        )

        print(
            "=" * 120,
            flush=True,
        )

        print(
            f"Script version : "
            f"{SCRIPT_VERSION}",
            flush=True,
        )

        print(
            f"Input PT       : "
            f"{args.input_pt}",
            flush=True,
        )

        print(
            f"Output PT      : "
            f"{args.output_pt}",
            flush=True,
        )

        print(
            f"Device         : "
            f"{device}",
            flush=True,
        )

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

        print(
            f"Batch size     : "
            f"{args.batch_size:,}",
            flush=True,
        )

        print(
            "Prototype data : "
            "model_splits == train, ARG sequences only",
            flush=True,
        )

        print(
            "nonARG policy  : "
            "excluded from ARG prototypes; "
            "train-only centroid stored",
            flush=True,
        )

        print(
            "Full-length    : "
            "one fused embedding per sequence; "
            "no fragment aggregation",
            flush=True,
        )

        hb.update(
            phase="load_input",
            note="loading long_fusion_subtype2_newsplit.pt",
        )

        (
            input_obj,
            ids,
            subtypes,
            external_splits,
            model_splits,
            embeddings,
        ) = load_and_validate_input(
            args.input_pt
        )

        hb.update(
            phase="build_prototypes",
            note="using model-train ARG sequences only",
        )

        prototype_data = (
            build_train_only_prototypes(
                input_obj,
                subtypes,
                model_splits,
                embeddings,
            )
        )

        prototypes = (
            prototype_data[
                "raw_arg_subtype_prototypes"
            ]
        )

        subtype_names = (
            prototype_data[
                "arg_subtype_names"
            ]
        )

        subtype_to_index = (
            prototype_data[
                "arg_subtype_to_index"
            ]
        )

        split_indices = {
            split:
                [
                    index
                    for index, value
                    in enumerate(
                        model_splits
                    )
                    if value == split
                ]
            for split
            in (
                "validation",
                "test",
            )
        }

        print(
            "\n===== COSINE NEAREST-PROTOTYPE BASELINES (ARG ONLY) =====",
            flush=True,
        )

        evaluations = {}

        for split_name in (
            "validation",
            "test",
        ):

            evaluations[
                split_name
            ] = nearest_prototype_evaluation(
                embeddings,
                subtypes,
                ids,
                split_indices[
                    split_name
                ],
                prototypes,
                subtype_names,
                subtype_to_index,
                device,
                args.batch_size,
                hb,
                split_name,
            )

            print_result(
                split_name,
                evaluations[
                    split_name
                ],
            )

        output = {
            "arg_subtype_names":
                subtype_names,

            "arg_subtype_to_index":
                subtype_to_index,

            "raw_arg_subtype_prototypes":
                prototypes,

            "prototype_train_sequence_counts":
                prototype_data[
                    "prototype_train_sequence_counts"
                ],

            "nonarg_reference_centroid":
                prototype_data[
                    "nonarg_reference_centroid"
                ],

            "nonarg_train_sequence_count":
                prototype_data[
                    "nonarg_train_sequence_count"
                ],

            "baseline_evaluations":
                evaluations,

            "embedding_dimension":
                int(
                    prototypes.shape[
                        1
                    ]
                ),

            "num_ARG_subtypes":
                len(
                    subtype_names
                ),

            "prototype_construction":
                (
                    "L2 normalize each full-length fused embedding -> "
                    "mean model-train sequences per ARG Subtype -> "
                    "L2 normalize subtype prototype"
                ),

            "classification_rule":
                (
                    "maximum cosine similarity to all "
                    "raw ARG subtype prototypes"
                ),

            "nonARG_policy":
                (
                    "excluded from ARG prototype graph; "
                    "train-only reference centroid stored separately"
                ),

            "sequence_level":
                "full-length / long",

            "source_gated_fusion_pt":
                str(
                    args.input_pt
                ),

            "source_fusion_script_version":
                input_obj.get(
                    "script_version"
                ),

            "script_version":
                SCRIPT_VERSION,
        }

        hb.update(
            phase="saving",
            note=str(
                args.output_pt
            ),
        )

        contract.stamp_derived(output, input_obj, args.input_pt, 'raw')

        safe_save(
            output,
            args.output_pt,
        )

        print(
            "\n"
            + "=" * 120,
            flush=True,
        )

        print(
            "LONG RAW PROTOTYPE BASELINE COMPLETED",
            flush=True,
        )

        print(
            "=" * 120,
            flush=True,
        )

        print(
            f"Raw ARG prototypes : "
            f"{tuple(prototypes.shape)}",
            flush=True,
        )

        print(
            f"ARG subtype count  : "
            f"{len(subtype_names):,}",
            flush=True,
        )

        print(
            f"Embedding dimension: "
            f"{prototypes.shape[1]:,}",
            flush=True,
        )

        print(
            f"Output PT          : "
            f"{args.output_pt}",
            flush=True,
        )

        print(
            f"Elapsed seconds    : "
            f"{time.time()-started:.3f}",
            flush=True,
        )

    finally:

        hb.stop()


if __name__ == "__main__":
    main()
