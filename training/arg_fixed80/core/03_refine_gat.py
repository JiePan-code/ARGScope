#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Long/full-length Mutual-KNN + residual GAT refinement of ARG subtype prototypes.

Inputs
------
ROOT/Raw_subtype_prototypes_subtype2_newsplit/raw_ARG_subtype_prototypes_and_baseline.pt
ROOT/Long_fusion_subtype2_newsplit/long_fusion_subtype2_newsplit.pt
structure table with columns:
    SARG.Seq.ID
    Subtype
    Type

Where:
    ROOT=/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding

Key differences from the short-read implementation
---------------------------------------------------
* One full-length sequence = one fused embedding.
* No fragment/original aggregation.
* GAT optimization uses model-train ARG full-length sequence embeddings.
* Mutual-KNN graph nodes/edges are built only from train-only raw prototypes.
* K and early stopping are selected only by validation full-length macro-F1.
* External test is evaluated only after K and best epoch are fixed.
* nonARG is excluded from the ARG subtype graph.
* Type auxiliary targets are read from the real structure table rather than
  inferred from string formatting of Subtype names.

Output
------
ROOT/Refined_subtype_prototypes_subtype2_newsplit/
    refined_ARG_subtype_prototypes_residual_GAT.pt

CUDA/A100 only. No CPU fallback.
"""

from __future__ import annotations

import newsplit_contract as contract

import argparse
import csv
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(
    "/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding"
)

DEFAULT_RAW_PT = (
    ROOT
    / "Raw_subtype_prototypes_subtype2_newsplit"
    / "raw_ARG_subtype_prototypes_and_baseline.pt"
)

DEFAULT_FUSED_PT = (
    ROOT
    / "Long_fusion_subtype2_newsplit"
    / "long_fusion_subtype2_newsplit.pt"
)

DEFAULT_OUTPUT_PT = (
    ROOT
    / "Refined_subtype_prototypes_subtype2_newsplit"
    / "refined_ARG_subtype_prototypes_residual_GAT.pt"
)

SCRIPT_VERSION = (
    "2026-08-22-long-mutual-knn-residual-gat-refined-prototypes-v1"
)

NONARG = "nonARG"


@dataclass(frozen=True)
class Configuration:
    k_candidates: tuple[int, ...]
    heads: int
    dropout: float
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    temperature: float
    anchor_weight: float
    type_weight: float
    label_smoothing: float
    maximum_residual_scale: float
    evaluation_batch_size: int
    seed: int


# =============================================================================
# Arguments
# =============================================================================

SCRIPT_VERSION = SCRIPT_VERSION + "-unified-newsplit-v1"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train long/full-length residual GAT refined "
            "ARG subtype prototypes."
        )
    )

    parser.add_argument(
        "--raw-pt",
        type=Path,
        default=DEFAULT_RAW_PT,
    )

    parser.add_argument(
        "--fused-pt",
        type=Path,
        default=DEFAULT_FUSED_PT,
    )

    parser.add_argument(
        "--structure",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
    )

    parser.add_argument(
        "--k-candidates",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--type-weight",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--maximum-residual-scale",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260822,
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
        self.k = None
        self.epoch = None
        self.note = ""

    def update(
        self,
        phase=None,
        k=None,
        epoch=None,
        note=None,
    ):

        with self.lock:

            if phase is not None:
                self.phase = str(phase)

            if k is not None:
                self.k = int(k)

            if epoch is not None:
                self.epoch = int(epoch)

            if note is not None:
                self.note = str(note)

    def emit(self):

        with self.lock:
            phase = self.phase
            k = self.k
            epoch = self.epoch
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

            max_allocated = (
                torch.cuda.max_memory_allocated(0)
                / (1024 ** 3)
            )

            gpu_text = (
                f"GPU_alloc={allocated:.2f}GB | "
                f"GPU_reserved={reserved:.2f}GB | "
                f"GPU_peak={max_allocated:.2f}GB"
            )

        except Exception:

            gpu_text = "GPU_mem=NA"

        progress = ""

        if k is not None:
            progress += f" | K={k}"

        if epoch is not None:
            progress += f" | epoch={epoch}"

        print(
            f"[HEARTBEAT] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"phase={phase}"
            f"{progress} | "
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
# General utilities
# =============================================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Faster A100 mode. Scientific model/loss logic is unchanged.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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


def canonical_id(value):

    text = str(value).strip()

    return (
        text.split()[0]
        if text
        else ""
    )


def normalize_label(value):

    text = str(value).strip()

    compact = "".join(
        ch
        for ch in text.lower()
        if ch.isalnum()
    )

    if compact in {
        "nonarg",
        "nonresistance",
        "negative",
        "nonresistant",
    }:

        return NONARG

    return text


def is_nonarg(value):

    return normalize_label(
        value
    ) == NONARG


def validate_tensor(
    tensor,
    shape,
    description,
):

    if (
        not isinstance(
            tensor,
            torch.Tensor,
        )
        or tensor.ndim != len(shape)
    ):

        raise ValueError(
            f"{description} must be a {len(shape)}D tensor."
        )

    for observed, expected in zip(
        tensor.shape,
        shape,
    ):

        if (
            expected is not None
            and observed != expected
        ):

            raise ValueError(
                f"Unexpected {description} shape: "
                f"{tuple(tensor.shape)}, "
                f"expected={shape}"
            )

    tensor = (
        tensor
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    if not bool(
        torch.isfinite(
            tensor
        ).all()
    ):

        raise ValueError(
            f"Non-finite values in {description}."
        )

    return tensor


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

    with temporary.open(
        "wb"
    ) as handle:

        torch.save(
            obj,
            handle,
        )

        handle.flush()
        os.fsync(
            handle.fileno()
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
# Type mapping from actual structure table
# =============================================================================

def detect_delimiter(path: Path):

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:

        first_line = handle.readline()

    return (
        "\t"
        if "\t" in first_line
        else ","
    )


def load_subtype_to_type(
    structure_path: Path,
    required_arg_subtypes: list[str],
):

    delimiter = detect_delimiter(
        structure_path
    )

    mapping = {}

    with structure_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
        )

        if not reader.fieldnames:

            raise ValueError(
                f"Structure file has no header: {structure_path}"
            )

        required_columns = {
            "Subtype",
            "Type",
        }

        missing = sorted(
            required_columns
            - set(
                reader.fieldnames
            )
        )

        if missing:

            raise ValueError(
                f"Structure file missing columns "
                f"{missing}: {structure_path}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):

            subtype = normalize_label(
                row.get(
                    "Subtype",
                    "",
                )
            )

            type_name = str(
                row.get(
                    "Type",
                    "",
                )
            ).strip()

            if not subtype:

                raise ValueError(
                    f"Empty Subtype at line {line_number}"
                )

            if is_nonarg(
                subtype
            ):
                continue

            if not type_name:

                raise ValueError(
                    f"Empty Type at line {line_number} "
                    f"for Subtype={subtype!r}"
                )

            previous = mapping.get(
                subtype
            )

            if (
                previous is not None
                and previous != type_name
            ):

                raise ValueError(
                    f"Subtype maps to multiple Types: "
                    f"{subtype!r}: "
                    f"{previous!r} vs {type_name!r}"
                )

            mapping[
                subtype
            ] = type_name

    missing_subtypes = sorted(
        set(
            required_arg_subtypes
        )
        - set(
            mapping
        )
    )

    if missing_subtypes:

        raise ValueError(
            f"{len(missing_subtypes):,} ARG subtype(s) "
            f"lack Type mapping in structure; "
            f"examples={missing_subtypes[:20]}"
        )

    mapping = {
        subtype:
            mapping[
                subtype
            ]
        for subtype
        in required_arg_subtypes
    }

    type_names = sorted(
        set(
            mapping.values()
        )
    )

    if not type_names:

        raise ValueError(
            "No ARG Type labels were found."
        )

    print(
        f"ARG subtype->Type mappings : "
        f"{len(mapping):,}",
        flush=True,
    )

    print(
        f"ARG Types                  : "
        f"{len(type_names):,}",
        flush=True,
    )

    return (
        mapping,
        type_names,
    )


# =============================================================================
# Input loading
# =============================================================================

def load_inputs(
    raw_path: Path,
    fused_path: Path,
):

    raw = torch_load_cpu(
        raw_path
    )

    fused = torch_load_cpu(
        fused_path
    )
    contract.validate_derived(raw, fused, fused_path, 'raw')


    if (
        not isinstance(
            raw,
            dict,
        )
        or not isinstance(
            fused,
            dict,
        )
    ):

        raise TypeError(
            "Both PT files must contain dictionaries."
        )

    raw_required = {
        "arg_subtype_names",
        "arg_subtype_to_index",
        "raw_arg_subtype_prototypes",
        "prototype_train_sequence_counts",
        "baseline_evaluations",
    }

    fused_required = {
        "ids",
        "embeddings",
        "subtypes",
        "model_splits",
        "splits",
    }

    missing_raw = sorted(
        raw_required
        - set(
            raw
        )
    )

    missing_fused = sorted(
        fused_required
        - set(
            fused
        )
    )

    if missing_raw:

        raise KeyError(
            f"Raw PT missing keys: {missing_raw}"
        )

    if missing_fused:

        raise KeyError(
            f"Fused PT missing keys: {missing_fused}"
        )

    subtype_names = [
        str(value)
        for value
        in raw[
            "arg_subtype_names"
        ]
    ]

    if len(
        subtype_names
    ) != len(
        set(
            subtype_names
        )
    ):

        raise ValueError(
            "Duplicate ARG subtype names."
        )

    if any(
        is_nonarg(
            value
        )
        for value
        in subtype_names
    ):

        raise ValueError(
            "nonARG must not be present in ARG subtype prototype names."
        )

    subtype_to_index = {
        str(key):
            int(
                value
            )
        for key, value
        in raw[
            "arg_subtype_to_index"
        ].items()
    }

    expected_map = {
        name:
            index
        for index, name
        in enumerate(
            subtype_names
        )
    }

    if subtype_to_index != expected_map:

        raise ValueError(
            "arg_subtype_to_index is inconsistent "
            "with arg_subtype_names."
        )

    number_classes = len(
        subtype_names
    )

    prototypes = validate_tensor(
        raw[
            "raw_arg_subtype_prototypes"
        ],
        (
            number_classes,
            None,
        ),
        "raw prototypes",
    )

    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=1,
    )

    embedding_dim = int(
        prototypes.shape[
            1
        ]
    )

    sequence_ids = [
        canonical_id(
            value
        )
        for value
        in fused[
            "ids"
        ]
    ]

    sequence_subtypes = [
        normalize_label(
            value
        )
        for value
        in fused[
            "subtypes"
        ]
    ]

    sequence_model_splits = [
        str(value).strip().lower()
        for value
        in fused[
            "model_splits"
        ]
    ]

    sequence_external_splits = [
        str(value).strip().lower()
        for value
        in fused[
            "splits"
        ]
    ]

    sequence_embeddings = validate_tensor(
        fused[
            "embeddings"
        ],
        (
            len(
                sequence_ids
            ),
            embedding_dim,
        ),
        "full-length fused embeddings",
    )

    sequence_embeddings = F.normalize(
        sequence_embeddings,
        p=2,
        dim=1,
    )

    number = len(
        sequence_ids
    )

    if not (
        number
        == len(
            sequence_subtypes
        )
        == len(
            sequence_model_splits
        )
        == len(
            sequence_external_splits
        )
    ):

        raise ValueError(
            "Full-length metadata length mismatch."
        )

    if any(
        not value
        for value
        in sequence_ids
    ):

        raise ValueError(
            "Empty sequence ID detected."
        )

    if len(
        sequence_ids
    ) != len(
        set(
            sequence_ids
        )
    ):

        raise ValueError(
            "Duplicate full-length sequence IDs."
        )

    invalid_model = (
        set(
            sequence_model_splits
        )
        - {
            "train",
            "validation",
            "test",
        }
    )

    if invalid_model:

        raise ValueError(
            f"Invalid model_splits: "
            f"{sorted(invalid_model)}"
        )

    invalid_external = (
        set(
            sequence_external_splits
        )
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

    observed_arg = {
        label
        for label
        in sequence_subtypes
        if not is_nonarg(
            label
        )
    }

    if observed_arg != set(
        subtype_names
    ):

        raise ValueError(
            f"ARG subtype set mismatch: "
            f"prototypes={len(subtype_names):,}, "
            f"observed={len(observed_arg):,}, "
            f"missing_in_fused="
            f"{len(set(subtype_names)-observed_arg):,}, "
            f"extra_in_fused="
            f"{len(observed_arg-set(subtype_names)):,}"
        )

    print(
        f"ARG raw prototypes        : "
        f"{tuple(prototypes.shape)}",
        flush=True,
    )

    print(
        f"Full-length embeddings    : "
        f"{tuple(sequence_embeddings.shape)}",
        flush=True,
    )

    print(
        f"ARG subtype classes       : "
        f"{number_classes:,}",
        flush=True,
    )

    print(
        f"Full-length sequences     : "
        f"{number:,}",
        flush=True,
    )

    for split in (
        "train",
        "validation",
        "test",
    ):

        arg_count = sum(
            (
                model_split == split
                and not is_nonarg(
                    subtype
                )
            )
            for subtype, model_split
            in zip(
                sequence_subtypes,
                sequence_model_splits,
            )
        )

        print(
            f"{split:10s} ARG sequences : "
            f"{arg_count:,}",
            flush=True,
        )

    return {
        'fusion_type_mapping': fused['subtype_to_type'],
        "raw_object":
            raw,

        "fused_object":
            fused,

        "subtype_names":
            subtype_names,

        "subtype_to_index":
            subtype_to_index,

        "raw_prototypes":
            prototypes,

        "sequence_ids":
            sequence_ids,

        "sequence_subtypes":
            sequence_subtypes,

        "sequence_model_splits":
            sequence_model_splits,

        "sequence_external_splits":
            sequence_external_splits,

        "sequence_embeddings":
            sequence_embeddings,
    }


# =============================================================================
# Select ARG split
# =============================================================================

def select_arg_split(
    features,
    ids,
    labels,
    splits,
    target_split,
    label_map,
):

    indices = [
        index
        for index, (
            label,
            split,
        )
        in enumerate(
            zip(
                labels,
                splits,
            )
        )
        if (
            split == target_split
            and not is_nonarg(
                label
            )
        )
    ]

    if not indices:

        raise ValueError(
            f"No ARG samples in split={target_split}"
        )

    tensor_indices = torch.tensor(
        indices,
        dtype=torch.long,
    )

    selected_features = (
        features
        .index_select(
            0,
            tensor_indices,
        )
        .contiguous()
    )

    selected_targets = torch.tensor(
        [
            label_map[
                labels[
                    index
                ]
            ]
            for index
            in indices
        ],
        dtype=torch.long,
    )

    selected_ids = [
        ids[
            index
        ]
        for index
        in indices
    ]

    return (
        selected_features,
        selected_targets,
        selected_ids,
    )


# =============================================================================
# Mutual-KNN graph
# =============================================================================

def build_mutual_knn_graph(
    prototypes: torch.Tensor,
    k: int,
):

    number_nodes = int(
        prototypes.shape[
            0
        ]
    )

    if not (
        1
        <= k
        < number_nodes
    ):

        raise ValueError(
            f"K must satisfy "
            f"1 <= K < {number_nodes}, "
            f"observed {k}"
        )

    similarity = (
        prototypes
        @ prototypes.T
    )

    similarity.fill_diagonal_(
        -torch.inf
    )

    neighbors = similarity.topk(
        k=k,
        dim=1,
    ).indices

    directed = torch.zeros(
        (
            number_nodes,
            number_nodes,
        ),
        dtype=torch.bool,
        device=prototypes.device,
    )

    rows = torch.arange(
        number_nodes,
        device=prototypes.device,
    ).unsqueeze(
        1
    ).expand_as(
        neighbors
    )

    directed[
        rows,
        neighbors
    ] = True

    mutual = (
        directed
        & directed.T
    )

    adjacency = mutual.clone()

    adjacency.fill_diagonal_(
        True
    )

    undirected_edge_count = int(
        mutual
        .triu(
            diagonal=1
        )
        .sum()
        .item()
    )

    neighbor_degrees = (
        mutual
        .sum(
            dim=1
        )
        .to(
            torch.long
        )
    )

    isolated_without_self = int(
        (
            neighbor_degrees
            == 0
        )
        .sum()
        .item()
    )

    edge_index = (
        adjacency
        .nonzero(
            as_tuple=False
        )
        .T
        .contiguous()
    )

    return {
        "adjacency":
            adjacency,

        "edge_index":
            edge_index,

        "undirected_edge_count":
            undirected_edge_count,

        "neighbor_degrees":
            neighbor_degrees,

        "isolated_nodes_without_self":
            isolated_without_self,
    }


# =============================================================================
# Residual GAT
# =============================================================================

class DenseGraphAttentionLayer(
    nn.Module
):

    def __init__(
        self,
        dimension,
        heads,
        dropout,
    ):

        super().__init__()

        if (
            dimension
            % heads
            != 0
        ):

            raise ValueError(
                f"dimension={dimension} "
                f"must be divisible by heads={heads}"
            )

        self.dimension = dimension
        self.heads = heads
        self.head_dimension = (
            dimension
            // heads
        )

        self.projection = nn.Linear(
            dimension,
            dimension,
            bias=False,
        )

        self.attention_source = nn.Parameter(
            torch.empty(
                heads,
                self.head_dimension,
            )
        )

        self.attention_target = nn.Parameter(
            torch.empty(
                heads,
                self.head_dimension,
            )
        )

        self.output_projection = nn.Linear(
            dimension,
            dimension,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.leaky_relu = nn.LeakyReLU(
            0.2
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.projection.weight
        )

        nn.init.xavier_uniform_(
            self.output_projection.weight
        )

        nn.init.xavier_uniform_(
            self.attention_source
        )

        nn.init.xavier_uniform_(
            self.attention_target
        )

    def forward(
        self,
        nodes,
        adjacency,
    ):

        number_nodes = int(
            nodes.shape[
                0
            ]
        )

        transformed = (
            self.projection(
                nodes
            )
            .view(
                number_nodes,
                self.heads,
                self.head_dimension,
            )
            .permute(
                1,
                0,
                2,
            )
        )

        source_score = (
            transformed
            * self.attention_source[
                :,
                None,
                :,
            ]
        ).sum(
            dim=2
        )

        target_score = (
            transformed
            * self.attention_target[
                :,
                None,
                :,
            ]
        ).sum(
            dim=2
        )

        scores = self.leaky_relu(
            source_score[
                :,
                :,
                None,
            ]
            + target_score[
                :,
                None,
                :,
            ]
        )

        scores = scores.masked_fill(
            ~adjacency.unsqueeze(
                0
            ),
            -torch.inf,
        )

        attention = torch.softmax(
            scores,
            dim=2,
        )

        attention = self.dropout(
            attention
        )

        aggregated = torch.matmul(
            attention,
            transformed,
        )

        aggregated = (
            aggregated
            .permute(
                1,
                0,
                2,
            )
            .reshape(
                number_nodes,
                self.dimension,
            )
        )

        return self.output_projection(
            aggregated
        )


class ResidualPrototypeGAT(
    nn.Module
):

    def __init__(
        self,
        dimension,
        heads,
        dropout,
        number_types,
        maximum_residual_scale,
    ):

        super().__init__()

        self.gat1 = DenseGraphAttentionLayer(
            dimension,
            heads,
            dropout,
        )

        self.gat2 = DenseGraphAttentionLayer(
            dimension,
            heads,
            dropout,
        )

        self.norm1 = nn.LayerNorm(
            dimension
        )

        self.norm2 = nn.LayerNorm(
            dimension
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.type_classifier = nn.Linear(
            dimension,
            number_types,
        )

        self.residual_scale_logit = nn.Parameter(
            torch.tensor(
                -2.0
            )
        )

        self.maximum_residual_scale = float(
            maximum_residual_scale
        )

    def forward(
        self,
        raw_prototypes,
        adjacency,
    ):

        hidden = self.gat1(
            raw_prototypes,
            adjacency,
        )

        hidden = self.norm1(
            F.elu(
                hidden
            )
        )

        hidden = self.dropout(
            hidden
        )

        update = self.gat2(
            hidden,
            adjacency,
        )

        update = self.norm2(
            update
        )

        residual_scale = (
            self.maximum_residual_scale
            * torch.sigmoid(
                self.residual_scale_logit
            )
        )

        refined = F.normalize(
            raw_prototypes
            + residual_scale
            * update,
            p=2,
            dim=1,
        )

        type_logits = self.type_classifier(
            refined
        )

        return (
            refined,
            type_logits,
            residual_scale,
        )


# =============================================================================
# Weights and metrics
# =============================================================================

def class_weights(
    targets,
    number_classes,
):

    counts = torch.bincount(
        targets,
        minlength=number_classes,
    ).float()

    if bool(
        (
            counts
            == 0
        ).any()
    ):

        missing = torch.where(
            counts
            == 0
        )[0].tolist()

        raise ValueError(
            f"At least one class is absent "
            f"from supplied targets; "
            f"examples={missing[:20]}"
        )

    weights = counts.rsqrt()

    return (
        weights
        / weights.mean()
    )


def confusion_metrics(
    true_targets,
    predictions,
    top5_correct,
    maximum_similarity,
    number_classes,
):

    true_np = (
        true_targets
        .cpu()
        .numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    pred_np = (
        predictions
        .cpu()
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
            pred_np,
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
        tp.sum()
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

    return {
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
                maximum_similarity
                .mean()
                .item()
            ),

        "median_max_cosine_similarity":
            float(
                maximum_similarity
                .median()
                .item()
            ),

        "confusion_matrix":
            torch.from_numpy(
                confusion.astype(
                    np.int32,
                    copy=False,
                )
            ),
    }


@torch.inference_mode()
def evaluate_prototypes(
    features,
    targets,
    prototypes,
    batch_size,
):

    predictions = []
    maximum_similarities = []
    top5_results = []

    top_k = min(
        5,
        prototypes.shape[
            0
        ],
    )

    for start in range(
        0,
        features.shape[
            0
        ],
        batch_size,
    ):

        end = min(
            start
            + batch_size,
            features.shape[
                0
            ],
        )

        similarities = (
            features[
                start:
                end
            ]
            @ prototypes.T
        )

        values, indices = similarities.topk(
            k=top_k,
            dim=1,
        )

        true_batch = targets[
            start:
            end
        ]

        predictions.append(
            indices[
                :,
                0
            ]
            .cpu()
        )

        maximum_similarities.append(
            values[
                :,
                0
            ]
            .float()
            .cpu()
        )

        top5_results.append(
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

    predictions = torch.cat(
        predictions
    )

    maximum_similarities = torch.cat(
        maximum_similarities
    )

    top5_correct = torch.cat(
        top5_results
    )

    metrics = confusion_metrics(
        targets.cpu(),
        predictions,
        top5_correct,
        maximum_similarities,
        prototypes.shape[
            0
        ],
    )

    return {
        "metrics":
            metrics,

        "true_subtype_indices":
            targets.cpu(),

        "predicted_subtype_indices":
            predictions,

        "maximum_cosine_similarities":
            maximum_similarities,

        "top5_correct":
            top5_correct,
    }


# =============================================================================
# Train one K
# =============================================================================

def train_one_k(
    k,
    raw_prototypes,
    graph,
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    type_targets,
    number_types,
    config,
    device,
    hb,
):

    set_seed(
        config.seed
        + k
        * 1009
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    model = ResidualPrototypeGAT(
        dimension=raw_prototypes.shape[
            1
        ],
        heads=config.heads,
        dropout=config.dropout,
        number_types=number_types,
        maximum_residual_scale=(
            config.maximum_residual_scale
        ),
    ).to(
        device
    )

    adjacency = graph[
        "adjacency"
    ]

    subtype_weight = class_weights(
        train_targets,
        raw_prototypes.shape[
            0
        ],
    )

    type_weight = class_weights(
        type_targets,
        number_types,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=(
            config.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
    )

    best_score = -math.inf
    best_accuracy = -math.inf
    best_epoch = -1
    best_state = None
    best_validation_metrics = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(
        1,
        config.epochs
        + 1,
    ):

        hb.update(
            phase="train_k",
            k=k,
            epoch=epoch,
            note="full-length validation-only model selection",
        )

        epoch_started = time.time()

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        refined, type_logits, residual_scale = model(
            raw_prototypes,
            adjacency,
        )

        subtype_logits = (
            train_features
            @ refined.T
        ) / config.temperature

        subtype_loss = F.cross_entropy(
            subtype_logits,
            train_targets,
            weight=subtype_weight,
            label_smoothing=(
                config.label_smoothing
            ),
        )

        anchor_loss = (
            1.0
            - (
                refined
                * raw_prototypes
            ).sum(
                dim=1
            )
        ).mean()

        type_loss = F.cross_entropy(
            type_logits,
            type_targets,
            weight=type_weight,
        )

        loss = (
            subtype_loss
            + config.anchor_weight
            * anchor_loss
            + config.type_weight
            * type_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        model.eval()

        with torch.inference_mode():

            refined_validation, _, scale_validation = model(
                raw_prototypes,
                adjacency,
            )

            validation = evaluate_prototypes(
                validation_features,
                validation_targets,
                refined_validation,
                config.evaluation_batch_size,
            )

        validation_metrics = validation[
            "metrics"
        ]

        score = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        accuracy = float(
            validation_metrics[
                "top1_accuracy"
            ]
        )

        scheduler.step(
            score
        )

        current_lr = (
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )

        epoch_seconds = (
            time.time()
            - epoch_started
        )

        history.append(
            {
                "epoch":
                    epoch,

                "total_loss":
                    float(
                        loss.item()
                    ),

                "subtype_loss":
                    float(
                        subtype_loss.item()
                    ),

                "anchor_loss":
                    float(
                        anchor_loss.item()
                    ),

                "type_loss":
                    float(
                        type_loss.item()
                    ),

                "validation_accuracy":
                    accuracy,

                "validation_macro_precision":
                    float(
                        validation_metrics[
                            "macro_precision"
                        ]
                    ),

                "validation_macro_recall":
                    float(
                        validation_metrics[
                            "macro_recall"
                        ]
                    ),

                "validation_macro_f1":
                    score,

                "validation_weighted_f1":
                    float(
                        validation_metrics[
                            "weighted_f1"
                        ]
                    ),

                "validation_mcc":
                    float(
                        validation_metrics[
                            "mcc"
                        ]
                    ),

                "residual_scale":
                    float(
                        scale_validation.item()
                    ),

                "learning_rate":
                    float(
                        current_lr
                    ),

                "epoch_seconds":
                    float(
                        epoch_seconds
                    ),
            }
        )

        print(
            f"K={k:02d} "
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"loss={loss.item():.6f} | "
            f"subtype={subtype_loss.item():.6f} | "
            f"anchor={anchor_loss.item():.6f} | "
            f"type={type_loss.item():.6f} | "
            f"val_ACC={accuracy:.4f} | "
            f"val_P={validation_metrics['macro_precision']:.4f} | "
            f"val_R={validation_metrics['macro_recall']:.4f} | "
            f"val_macro_F1={score:.4f} | "
            f"val_weighted_F1="
            f"{validation_metrics['weighted_f1']:.4f} | "
            f"val_MCC="
            f"{validation_metrics['mcc']:.4f} | "
            f"scale={scale_validation.item():.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_seconds:.2f}s",
            flush=True,
        )

        improved = (
            score
            > best_score
            + 1e-7
            or (
                abs(
                    score
                    - best_score
                )
                <= 1e-7
                and accuracy
                > best_accuracy
                + 1e-7
            )
        )

        if improved:

            best_score = score
            best_accuracy = accuracy
            best_epoch = epoch
            best_validation_metrics = (
                validation_metrics
            )

            best_state = {
                name:
                    tensor
                    .detach()
                    .cpu()
                    .clone()
                for name, tensor
                in model
                .state_dict()
                .items()
            }

            epochs_without_improvement = 0

            print(
                f"  -> NEW BEST K={k}: "
                f"epoch={epoch}, "
                f"val_macro_F1={score:.6f}, "
                f"val_ACC={accuracy:.6f}",
                flush=True,
            )

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= config.patience
            ):

                print(
                    f"K={k:02d} early stopping "
                    f"at epoch {epoch}; "
                    f"best_epoch={best_epoch}, "
                    f"best_macro_F1="
                    f"{best_score:.6f}",
                    flush=True,
                )

                break

    if (
        best_state is None
        or best_validation_metrics is None
    ):

        raise RuntimeError(
            f"K={k} did not produce a valid model."
        )

    peak_gpu_gb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024 ** 3)
    )

    del model

    torch.cuda.empty_cache()

    return {
        "k":
            k,

        "graph_undirected_edges":
            graph[
                "undirected_edge_count"
            ],

        "graph_isolated_nodes_without_self":
            graph[
                "isolated_nodes_without_self"
            ],

        "graph_neighbor_degrees":
            graph[
                "neighbor_degrees"
            ]
            .detach()
            .cpu(),

        "best_epoch":
            best_epoch,

        "best_validation_metrics":
            best_validation_metrics,

        "best_state_dict":
            best_state,

        "peak_gpu_memory_gb":
            float(
                peak_gpu_gb
            ),

        "history":
            history,
    }


# =============================================================================
# Output helpers
# =============================================================================

def metrics_without_confusion(
    metrics,
):

    return {
        key:
            value
        for key, value
        in metrics.items()
        if key
        != "confusion_matrix"
    }


def metric_deltas(
    refined_metrics,
    raw_metrics,
):

    keys = [
        "top1_accuracy",
        "top5_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "mcc",
    ]

    return {
        key:
            float(
                refined_metrics[
                    key
                ]
            )
            - float(
                raw_metrics[
                    key
                ]
            )
        for key
        in keys
        if (
            key
            in refined_metrics
            and key
            in raw_metrics
        )
    }


def print_metrics(
    name,
    evaluation,
):

    metrics = evaluation[
        "metrics"
    ]

    print(
        f"{name:12s} | "
        f"n={metrics['support']:,} | "
        f"subtypes="
        f"{metrics['evaluated_subtypes']:,} | "
        f"ACC="
        f"{metrics['top1_accuracy']:.6f} | "
        f"Top5="
        f"{metrics['top5_accuracy']:.6f} | "
        f"macro_P="
        f"{metrics['macro_precision']:.6f} | "
        f"macro_R="
        f"{metrics['macro_recall']:.6f} | "
        f"macro_F1="
        f"{metrics['macro_f1']:.6f} | "
        f"weighted_F1="
        f"{metrics['weighted_f1']:.6f} | "
        f"MCC="
        f"{metrics['mcc']:.6f}",
        flush=True,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()
    print("Device:", "cuda" if torch.cuda.is_available() else "CPU (CUDA required)", flush=True)
    print("Resolved arguments:", vars(args), flush=True)

    args.raw_pt = (
        args.raw_pt
        .expanduser()
        .resolve()
    )

    args.fused_pt = (
        args.fused_pt
        .expanduser()
        .resolve()
    )

    args.structure = (
        args.structure
        .expanduser()
        .resolve()
    )

    args.output_pt = (
        args.output_pt
        .expanduser()
        .resolve()
    )

    for path, description in (
        (
            args.raw_pt,
            "Raw prototype PT",
        ),
        (
            args.fused_pt,
            "Gated fusion PT",
        ),
        (
            args.structure,
            "Structure table",
        ),
    ):

        if not path.is_file():

            raise FileNotFoundError(
                f"{description} not found: {path}"
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

    if (
        not args.k_candidates
        or any(
            k < 1
            for k
            in args.k_candidates
        )
    ):

        raise ValueError(
            "All K candidates must be positive."
        )

    if len(
        set(
            args.k_candidates
        )
    ) != len(
        args.k_candidates
    ):

        raise ValueError(
            "K candidates must be unique."
        )

    if (
        args.heads < 1
        or args.epochs < 1
        or args.patience < 1
    ):

        raise ValueError(
            "Heads, epochs and patience "
            "must be positive."
        )

    if args.temperature <= 0:

        raise ValueError(
            "Temperature must be positive."
        )

    if not (
        0.0
        < args.maximum_residual_scale
        <= 1.0
    ):

        raise ValueError(
            "Maximum residual scale "
            "must be in (0, 1]."
        )

    if args.evaluation_batch_size < 1:

        raise ValueError(
            "Evaluation batch size "
            "must be positive."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is unavailable. "
            "CPU fallback is disabled."
        )

    torch.cuda.set_device(
        0
    )

    device = torch.device(
        "cuda:0"
    )

    # Keep all scientific tensors in float32, but allow A100 TF32 matmul.
    torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):

        torch.set_float32_matmul_precision(
            "high"
        )

    config = Configuration(
        k_candidates=tuple(
            args.k_candidates
        ),
        heads=args.heads,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        anchor_weight=args.anchor_weight,
        type_weight=args.type_weight,
        label_smoothing=args.label_smoothing,
        maximum_residual_scale=(
            args.maximum_residual_scale
        ),
        evaluation_batch_size=(
            args.evaluation_batch_size
        ),
        seed=args.seed,
    )

    set_seed(
        config.seed
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
            "=" * 125,
            flush=True,
        )

        print(
            "LONG / FULL-LENGTH MUTUAL-KNN + RESIDUAL GAT "
            "REFINED ARG SUBTYPE PROTOTYPES",
            flush=True,
        )

        print(
            "=" * 125,
            flush=True,
        )

        print(
            f"Script version : "
            f"{SCRIPT_VERSION}",
            flush=True,
        )

        print(
            f"Raw PT         : "
            f"{args.raw_pt}",
            flush=True,
        )

        print(
            f"Fused PT       : "
            f"{args.fused_pt}",
            flush=True,
        )

        print(
            f"Structure      : "
            f"{args.structure}",
            flush=True,
        )

        print(
            f"Output PT      : "
            f"{args.output_pt}",
            flush=True,
        )

        print(
            f"K candidates   : "
            f"{list(config.k_candidates)}",
            flush=True,
        )

        print(
            f"Heads          : "
            f"{config.heads}",
            flush=True,
        )

        print(
            f"Epochs         : "
            f"{config.epochs}",
            flush=True,
        )

        print(
            f"Patience       : "
            f"{config.patience}",
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
            f"CUDA version   : "
            f"{torch.version.cuda}",
            flush=True,
        )

        print(
            "CUDA fallback  : disabled",
            flush=True,
        )

        print(
            "AMP            : False "
            "(GAT/softmax kept FP32)",
            flush=True,
        )

        print(
            "TF32 matmul    : True",
            flush=True,
        )

        hb.update(
            phase="load_inputs",
            note="loading raw prototypes + long fused embeddings",
        )

        data = load_inputs(
            args.raw_pt,
            args.fused_pt,
        )

        subtype_names = data[
            "subtype_names"
        ]

        subtype_to_index = data[
            "subtype_to_index"
        ]

        raw_prototypes = data[
            "raw_prototypes"
        ]

        if (
            raw_prototypes.shape[
                1
            ]
            % config.heads
            != 0
        ):

            raise ValueError(
                f"Embedding dimension "
                f"{raw_prototypes.shape[1]} "
                f"is not divisible by "
                f"heads={config.heads}."
            )

        hb.update(
            phase="load_type_mapping",
            note="reading actual Type labels from structure",
        )

        (
            subtype_type_map,
            type_names,
        ) = load_subtype_to_type(
            args.structure,
            subtype_names,
        )

        if contract.type_mapping(args.structure, subtype_names) != data['fusion_type_mapping']:
            raise ValueError('GAT structure Type mapping differs from fusion')

        type_to_index = {
            name:
                index
            for index, name
            in enumerate(
                type_names
            )
        }

        type_targets = torch.tensor(
            [
                type_to_index[
                    subtype_type_map[
                        subtype
                    ]
                ]
                for subtype
                in subtype_names
            ],
            dtype=torch.long,
        )

        (
            train_features,
            train_targets,
            train_ids,
        ) = select_arg_split(
            data[
                "sequence_embeddings"
            ],
            data[
                "sequence_ids"
            ],
            data[
                "sequence_subtypes"
            ],
            data[
                "sequence_model_splits"
            ],
            "train",
            subtype_to_index,
        )

        (
            validation_features,
            validation_targets,
            validation_ids,
        ) = select_arg_split(
            data[
                "sequence_embeddings"
            ],
            data[
                "sequence_ids"
            ],
            data[
                "sequence_subtypes"
            ],
            data[
                "sequence_model_splits"
            ],
            "validation",
            subtype_to_index,
        )

        (
            test_features,
            test_targets,
            test_ids,
        ) = select_arg_split(
            data[
                "sequence_embeddings"
            ],
            data[
                "sequence_ids"
            ],
            data[
                "sequence_subtypes"
            ],
            data[
                "sequence_model_splits"
            ],
            "test",
            subtype_to_index,
        )

        print(
            "\n===== FULL-LENGTH ARG SPLITS =====",
            flush=True,
        )

        print(
            f"Train ARG sequences      : "
            f"{train_features.shape[0]:,}",
            flush=True,
        )

        print(
            f"Validation ARG sequences : "
            f"{validation_features.shape[0]:,}",
            flush=True,
        )

        print(
            f"Test ARG sequences       : "
            f"{test_features.shape[0]:,}",
            flush=True,
        )

        print(
            f"ARG subtype prototypes   : "
            f"{raw_prototypes.shape[0]:,}",
            flush=True,
        )

        print(
            f"ARG Types                : "
            f"{len(type_names):,}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Move all repeatedly used tensors to A100 exactly once.
        # ---------------------------------------------------------------------

        hb.update(
            phase="move_to_gpu",
            note="moving full training/validation tensors to A100 once",
        )

        raw_prototypes = raw_prototypes.to(
            device,
            non_blocking=True,
        )

        train_features = train_features.to(
            device,
            non_blocking=True,
        )

        train_targets = train_targets.to(
            device,
            non_blocking=True,
        )

        validation_features = validation_features.to(
            device,
            non_blocking=True,
        )

        validation_targets = validation_targets.to(
            device,
            non_blocking=True,
        )

        # Test embeddings are moved now for efficiency but never evaluated
        # until all K/epoch selection is complete.
        test_features = test_features.to(
            device,
            non_blocking=True,
        )

        test_targets = test_targets.to(
            device,
            non_blocking=True,
        )

        type_targets = type_targets.to(
            device,
            non_blocking=True,
        )

        torch.cuda.synchronize()

        print(
            f"GPU memory allocated     : "
            f"{torch.cuda.memory_allocated(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        print(
            f"GPU memory reserved      : "
            f"{torch.cuda.memory_reserved(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Validation-only K selection.
        # ---------------------------------------------------------------------

        candidate_results = []
        candidate_states = {}
        graphs = {}

        print(
            "\n===== VALIDATION-ONLY K SELECTION =====",
            flush=True,
        )

        for k in config.k_candidates:

            hb.update(
                phase="build_graph",
                k=k,
                epoch=0,
                note="Mutual-KNN from train-only raw prototypes",
            )

            graph = build_mutual_knn_graph(
                raw_prototypes,
                k,
            )

            graphs[
                k
            ] = graph

            print(
                f"\nK={k}: "
                f"undirected_edges="
                f"{graph['undirected_edge_count']:,}, "
                f"isolated_without_self="
                f"{graph['isolated_nodes_without_self']:,}, "
                f"degree_min="
                f"{graph['neighbor_degrees'].min().item()}, "
                f"degree_mean="
                f"{graph['neighbor_degrees'].float().mean().item():.3f}, "
                f"degree_max="
                f"{graph['neighbor_degrees'].max().item()}",
                flush=True,
            )

            result = train_one_k(
                k,
                raw_prototypes,
                graph,
                train_features,
                train_targets,
                validation_features,
                validation_targets,
                type_targets,
                len(
                    type_names
                ),
                config,
                device,
                hb,
            )

            candidate_states[
                k
            ] = result.pop(
                "best_state_dict"
            )

            candidate_results.append(
                result
            )

        selected_result = max(
            candidate_results,
            key=lambda item:
                (
                    item[
                        "best_validation_metrics"
                    ][
                        "macro_f1"
                    ],
                    item[
                        "best_validation_metrics"
                    ][
                        "top1_accuracy"
                    ],
                    -item[
                        "k"
                    ],
                ),
        )

        selected_k = int(
            selected_result[
                "k"
            ]
        )

        selected_graph = graphs[
            selected_k
        ]

        selected_state = candidate_states[
            selected_k
        ]

        print(
            "\n===== SELECTED BY VALIDATION ONLY =====",
            flush=True,
        )

        print(
            f"Selected K                   : "
            f"{selected_k}",
            flush=True,
        )

        print(
            f"Selected best epoch          : "
            f"{selected_result['best_epoch']}",
            flush=True,
        )

        print(
            f"Selected validation ACC      : "
            f"{selected_result['best_validation_metrics']['top1_accuracy']:.6f}",
            flush=True,
        )

        print(
            f"Selected validation macro F1 : "
            f"{selected_result['best_validation_metrics']['macro_f1']:.6f}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Load selected model and only now evaluate test.
        # ---------------------------------------------------------------------

        hb.update(
            phase="selected_model",
            k=selected_k,
            epoch=selected_result[
                "best_epoch"
            ],
            note="K/epoch fixed; external test can now be evaluated",
        )

        selected_model = ResidualPrototypeGAT(
            dimension=raw_prototypes.shape[
                1
            ],
            heads=config.heads,
            dropout=config.dropout,
            number_types=len(
                type_names
            ),
            maximum_residual_scale=(
                config.maximum_residual_scale
            ),
        ).to(
            device
        )

        selected_model.load_state_dict(
            selected_state
        )

        selected_model.eval()

        with torch.inference_mode():

            (
                refined_prototypes,
                type_logits,
                residual_scale,
            ) = selected_model(
                raw_prototypes,
                selected_graph[
                    "adjacency"
                ],
            )

        refined_prototypes = (
            refined_prototypes
            .detach()
        )

        print(
            "\n===== FINAL EVALUATION AFTER MODEL SELECTION =====",
            flush=True,
        )

        hb.update(
            phase="final_validation",
            k=selected_k,
            note="selected model",
        )

        validation_evaluation = evaluate_prototypes(
            validation_features,
            validation_targets,
            refined_prototypes,
            config.evaluation_batch_size,
        )

        print_metrics(
            "validation",
            validation_evaluation,
        )

        hb.update(
            phase="final_test",
            k=selected_k,
            note="first test evaluation after all model selection",
        )

        test_evaluation = evaluate_prototypes(
            test_features,
            test_targets,
            refined_prototypes,
            config.evaluation_batch_size,
        )

        print_metrics(
            "test",
            test_evaluation,
        )

        final_evaluations = {
            "validation":
                validation_evaluation,

            "test":
                test_evaluation,
        }

        raw_baselines = {
            name:
                data[
                    "raw_object"
                ][
                    "baseline_evaluations"
                ][
                    name
                ][
                    "metrics"
                ]
            for name
            in (
                "validation",
                "test",
            )
        }

        deltas = {
            name:
                metric_deltas(
                    final_evaluations[
                        name
                    ][
                        "metrics"
                    ],
                    raw_baselines[
                        name
                    ],
                )
            for name
            in (
                "validation",
                "test",
            )
        }

        print(
            "\n===== REFINED MINUS RAW BASELINE =====",
            flush=True,
        )

        for name, values in deltas.items():

            print(
                f"{name:12s} | "
                f"delta_ACC="
                f"{values['top1_accuracy']:+.6f} | "
                f"delta_macro_F1="
                f"{values['macro_f1']:+.6f} | "
                f"delta_MCC="
                f"{values['mcc']:+.6f}",
                flush=True,
            )

        # ---------------------------------------------------------------------
        # Save.
        # ---------------------------------------------------------------------

        output = {
            "arg_subtype_names":
                subtype_names,

            "arg_subtype_to_index":
                subtype_to_index,

            "raw_arg_subtype_prototypes":
                raw_prototypes
                .detach()
                .float()
                .cpu()
                .contiguous(),

            "refined_arg_subtype_prototypes":
                refined_prototypes
                .float()
                .cpu()
                .contiguous(),

            "selected_k":
                selected_k,

            "selected_best_epoch":
                int(
                    selected_result[
                        "best_epoch"
                    ]
                ),

            "selected_residual_scale":
                float(
                    residual_scale.item()
                ),

            "selected_graph_edge_index_with_self_loops":
                selected_graph[
                    "edge_index"
                ]
                .detach()
                .cpu(),

            "selected_graph_neighbor_degrees_without_self":
                selected_graph[
                    "neighbor_degrees"
                ]
                .detach()
                .cpu(),

            "selected_graph_undirected_edge_count":
                selected_graph[
                    "undirected_edge_count"
                ],

            "selected_model_state_dict":
                selected_state,

            "type_names":
                type_names,

            "type_to_index":
                type_to_index,

            "subtype_to_type":
                subtype_type_map,

            "type_targets_per_ARG_subtype":
                type_targets
                .detach()
                .cpu(),

            "candidate_validation_results":
                candidate_results,

            "final_evaluations":
                final_evaluations,

            "raw_baseline_metrics":
                {
                    name:
                        metrics_without_confusion(
                            metrics
                        )
                    for name, metrics
                    in raw_baselines.items()
                },

            "refined_minus_raw_metric_deltas":
                deltas,

            "evaluation_ids":
                {
                    "validation":
                        validation_ids,

                    "test":
                        test_ids,
                },

            "train_ARG_ids":
                train_ids,

            "configuration":
                asdict(
                    config
                ),

            "training_rule":
                (
                    "model-train ARG full-length sequences optimize GAT; "
                    "validation full-length ARG sequences select K/epoch; "
                    "external test evaluated only after selection"
                ),

            "graph_rule":
                (
                    "Mutual-KNN graph built only from "
                    "train-only raw ARG subtype prototypes"
                ),

            "fusion_formula":
                (
                    "L2Norm(raw_prototype + "
                    "residual_scale * two_layer_GAT_update)"
                ),

            "loss_formula":
                (
                    "weighted_subtype_CE + "
                    "anchor_weight*(1-cos(refined,raw)) + "
                    "type_weight*type_CE"
                ),

            "type_target_source":
                str(
                    args.structure
                ),

            "nonARG_policy":
                "excluded from ARG subtype graph",

            "sequence_level":
                "full-length / long",

            "source_raw_prototype_pt":
                str(
                    args.raw_pt
                ),

            "source_gated_fusion_pt":
                str(
                    args.fused_pt
                ),

            "script_version":
                SCRIPT_VERSION,
        }

        hb.update(
            phase="saving",
            k=selected_k,
            note=str(
                args.output_pt
            ),
        )

        contract.stamp_derived(output, torch_load_cpu(args.fused_pt), args.fused_pt, 'refined')
        output['source_raw_sha256'] = contract.sha256(args.raw_pt)

        safe_save(
            output,
            args.output_pt,
        )

        print(
            "\n"
            + "=" * 125,
            flush=True,
        )

        print(
            "LONG RESIDUAL GAT REFINED PROTOTYPES COMPLETED",
            flush=True,
        )

        print(
            "=" * 125,
            flush=True,
        )

        print(
            f"Selected K              : "
            f"{selected_k}",
            flush=True,
        )

        print(
            f"Selected best epoch     : "
            f"{selected_result['best_epoch']}",
            flush=True,
        )

        print(
            f"Selected residual scale : "
            f"{residual_scale.item():.6f}",
            flush=True,
        )

        print(
            f"Refined prototype shape : "
            f"{tuple(refined_prototypes.shape)}",
            flush=True,
        )

        print(
            f"Final validation ACC    : "
            f"{validation_evaluation['metrics']['top1_accuracy']:.6f}",
            flush=True,
        )

        print(
            f"Final validation F1     : "
            f"{validation_evaluation['metrics']['macro_f1']:.6f}",
            flush=True,
        )

        print(
            f"Final test ACC          : "
            f"{test_evaluation['metrics']['top1_accuracy']:.6f}",
            flush=True,
        )

        print(
            f"Final test macro F1     : "
            f"{test_evaluation['metrics']['macro_f1']:.6f}",
            flush=True,
        )

        print(
            f"Final test MCC          : "
            f"{test_evaluation['metrics']['mcc']:.6f}",
            flush=True,
        )

        print(
            f"Output PT               : "
            f"{args.output_pt}",
            flush=True,
        )

        print(
            f"Elapsed minutes         : "
            f"{(time.time()-started)/60.0:.3f}",
            flush=True,
        )

    finally:

        hb.stop()


if __name__ == "__main__":
    main()
