#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-length / long-aa subtype-supervised gated fusion.

Inputs
------
ROOT/Dataset_embedding/
    ESM2_full_1to1/full_1to1_esm2_mean.pt          [N, 1280]
    DNABERT2_full_1to1/full_1to1_dnabert2_mean.pt [N,  768]
    train_ids.txt
    test_ids.txt

Structure/label file:
    searched automatically in Dataset_embedding/ and ROOT, or pass --structure.

Fusion model
------------
ESM-2 1280D -> standardize -> Linear(1280,512) -> LN -> GELU -> Dropout
DNABERT2 768D -> standardize -> Linear(768,512) -> LN -> GELU -> Dropout
gate = sigmoid(MLP([protein_projected, dna_projected]))
fused = LN(gate * protein + (1-gate) * dna)
Subtype classifier = Linear(512, n_subtypes)

The external test IDs are NEVER used for:
    normalization
    optimization
    internal validation
    early stopping
    model selection

Internal validation is sampled ONLY from external train IDs.

Rare subtype handling: retain at least one model-train ARG per subtype;
validation fraction 0.20 with matched nonARG validation.

Performance
-----------
The complete embedding tensors are small enough for an A100-40GB, so this
script moves the aligned protein/DNA/target tensors to GPU once and performs
GPU-resident minibatching. This avoids DataLoader CPU-transfer overhead.

Logs
----
Frequent per-epoch logs plus a background heartbeat (default every 30 s).
"""

from __future__ import annotations

import newsplit_contract as contract

import argparse
import csv
import hashlib
import json
import math
import os
import random
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(
    "/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding"
)

DATA = ROOT

DEFAULT_PROTEIN_PT = (
    DATA
    / "ESM2_full_1to1"
    / "full_1to1_esm2_mean.pt"
)

DEFAULT_DNA_PT = (
    DATA
    / "DNABERT2_full_1to1"
    / "full_1to1_dnabert2_mean.pt"
)

DEFAULT_TRAIN_IDS = (
    DATA
    / "train_ids_subtype2.txt"
)

DEFAULT_TEST_IDS = (
    DATA
    / "test_ids_subtype2.txt"
)

DEFAULT_OUTPUT_PT = (
    ROOT
    / "Long_fusion_subtype2_newsplit"
    / "long_fusion_subtype2_newsplit.pt"
)

SCRIPT_VERSION = (
    "2026-08-22-long-full-supervised-gated-fusion-dataset-root-v4-nonarg-fix"
)

NONARG = "nonARG"


# =============================================================================
# Arguments
# =============================================================================

SCRIPT_VERSION = SCRIPT_VERSION + "-unified-newsplit-v1"


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--protein-pt",
        type=Path,
        default=DEFAULT_PROTEIN_PT,
    )

    p.add_argument(
        "--dna-pt",
        type=Path,
        default=DEFAULT_DNA_PT,
    )

    p.add_argument(
        "--train-ids",
        type=Path,
        default=DEFAULT_TRAIN_IDS,
    )

    p.add_argument(
        "--test-ids",
        type=Path,
        default=DEFAULT_TEST_IDS,
    )

    p.add_argument(
        "--structure",
        type=Path,
        default=None,
        help=(
            "Structure/label table containing SARG.Seq.ID and Subtype. "
            "If omitted, common filenames are searched automatically."
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
    )

    p.add_argument(
        "--fusion-dim",
        type=int,
        default=512,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8192,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=12,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260822,
    )

    p.add_argument(
        "--log-interval",
        type=int,
        default=30,
    )

    p.add_argument(
        "--no-amp",
        action="store_true",
    )

    p.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use deterministic CUDA behavior. Default is faster A100 mode."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


# =============================================================================
# Heartbeat
# =============================================================================

class Heartbeat:

    def __init__(
        self,
        interval: int,
    ):

        self.interval = max(
            5,
            int(
                interval
            ),
        )

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.phase = "startup"
        self.epoch = 0
        self.batch = 0
        self.total_batches = 0
        self.note = ""

        self.started = time.time()

    def update(
        self,
        *,
        phase=None,
        epoch=None,
        batch=None,
        total_batches=None,
        note=None,
    ):

        with self.lock:

            if phase is not None:
                self.phase = str(
                    phase
                )

            if epoch is not None:
                self.epoch = int(
                    epoch
                )

            if batch is not None:
                self.batch = int(
                    batch
                )

            if total_batches is not None:
                self.total_batches = int(
                    total_batches
                )

            if note is not None:
                self.note = str(
                    note
                )

    def emit(self):

        with self.lock:
            phase = self.phase
            epoch = self.epoch
            batch = self.batch
            total_batches = self.total_batches
            note = self.note

        try:

            alloc = (
                torch.cuda.memory_allocated(
                    0
                )
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved(
                    0
                )
                / (1024 ** 3)
            )

            gpu_text = (
                f"GPU_alloc={alloc:.2f}GB | "
                f"GPU_reserved={reserved:.2f}GB"
            )

        except Exception:

            gpu_text = "GPU_mem=NA"

        progress = ""

        if total_batches > 0:

            progress = (
                f" | epoch={epoch} | "
                f"batch={batch}/{total_batches}"
            )

        elif epoch > 0:

            progress = (
                f" | epoch={epoch}"
            )

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

        self.started = time.time()

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
# Reproducibility / speed
# =============================================================================

def set_seed(
    seed: int,
    deterministic: bool,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    if deterministic:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# =============================================================================
# General utilities
# =============================================================================

def canonical_id(
    value,
):

    text = str(
        value
    ).strip()

    return (
        text.split()[0]
        if text
        else ""
    )


def normalized_label(
    value,
):

    text = str(
        value
    ).strip()

    if not text:

        return NONARG

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


def stable_seed(
    seed,
    label,
):

    payload = (
        f"{seed}\0{label}"
        .encode(
            "utf-8"
        )
    )

    return int.from_bytes(
        hashlib.sha256(
            payload
        ).digest()[:8],
        "big",
    )


def torch_load_cpu(
    path,
):

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


def safe_save(
    obj,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    torch.save(
        obj,
        tmp,
    )

    if (
        not tmp.is_file()
        or tmp.stat().st_size <= 0
    ):

        raise RuntimeError(
            f"Temporary output is invalid: {tmp}"
        )

    os.replace(
        tmp,
        path,
    )


# =============================================================================
# Structure / split files
# =============================================================================

def find_structure(
    requested,
):

    if requested is not None:

        path = (
            requested
            .expanduser()
            .resolve()
        )

        if not path.is_file():

            raise FileNotFoundError(
                f"Structure file not found: {path}"
            )

        return path

    candidates = [
        ROOT / "SARG_full_final_structure.txt",
        ROOT / "all_ARG_nonARG_1to1_structure.txt",
        ROOT.parent / "SARG_full_final_structure.txt",
        ROOT.parent / "all_ARG_nonARG_1to1_structure.txt",
    ]

    for path in candidates:

        if path.is_file():
            return path

    raise FileNotFoundError(
        "Could not find a structure/label file automatically.\n"
        "Put SARG_full_final_structure.txt (or "
        "all_ARG_nonARG_1to1_structure.txt) in Dataset_embedding/ "
        "or pass --structure /path/to/file."
    )


def detect_delimiter(
    path,
):

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:

        first = handle.readline()

    return (
        "\t"
        if "\t" in first
        else ","
    )


def read_labels(
    path,
):

    """
    Reads labels for any IDs present in structure.

    If the structure is ARG-only, IDs absent from structure are later assigned
    nonARG. This matches the balanced dataset construction where the nonARG
    sequences were added after the SARG ARG structure was created.
    """

    delimiter = detect_delimiter(
        path
    )

    labels = {}

    with path.open(
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
                f"Structure file has no header: {path}"
            )

        required = {
            "SARG.Seq.ID",
            "Subtype",
        }

        missing = sorted(
            required
            - set(
                reader.fieldnames
            )
        )

        if missing:

            raise ValueError(
                f"Structure missing columns {missing}: {path}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):

            seq_id = canonical_id(
                row.get(
                    "SARG.Seq.ID",
                    "",
                )
            )

            subtype = normalized_label(
                row.get(
                    "Subtype",
                    "",
                )
            )

            if not seq_id:

                raise ValueError(
                    f"Empty ID at structure line {line_number}"
                )

            if seq_id in labels:

                raise ValueError(
                    f"Duplicate structure ID: {seq_id}"
                )

            labels[
                seq_id
            ] = subtype

    if not labels:

        raise ValueError(
            f"No rows found in structure: {path}"
        )

    return labels


def read_id_file(
    path,
):

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    ids = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:

        for line in handle:

            seq_id = canonical_id(
                line
            )

            if seq_id:

                ids.append(
                    seq_id
                )

    if not ids:

        raise ValueError(
            f"No IDs found: {path}"
        )

    if len(ids) != len(set(ids)):

        raise ValueError(
            f"Duplicate IDs found: {path}"
        )

    return ids


# =============================================================================
# Embedding PT
# =============================================================================

def load_embedding_pt(
    path,
    expected_dim,
):

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    obj = torch_load_cpu(
        path
    )

    if (
        not isinstance(
            obj,
            dict,
        )
        or "ids" not in obj
        or "embeddings" not in obj
    ):

        raise KeyError(
            f"PT requires ids + embeddings: {path}"
        )

    ids = [
        canonical_id(
            value
        )
        for value in obj[
            "ids"
        ]
    ]

    embeddings = (
        obj[
            "embeddings"
        ]
    )

    if any(
        not seq_id
        for seq_id in ids
    ):

        raise ValueError(
            f"Empty ID in PT: {path}"
        )

    if len(ids) != len(set(ids)):

        raise ValueError(
            f"Duplicate IDs in PT: {path}"
        )

    if (
        not isinstance(
            embeddings,
            torch.Tensor,
        )
        or embeddings.ndim != 2
    ):

        raise ValueError(
            f"Embeddings must be 2D: {path}"
        )

    if tuple(
        embeddings.shape
    ) != (
        len(
            ids
        ),
        expected_dim,
    ):

        raise ValueError(
            f"Unexpected shape {tuple(embeddings.shape)} "
            f"in {path}; expected "
            f"({len(ids)}, {expected_dim})"
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
            f"NaN/Inf in embeddings: {path}"
        )

    return (
        ids,
        embeddings,
    )


# =============================================================================
# Align all full-length data
# =============================================================================

def align_inputs(args, structure_path):
    protein_ids, protein_all = load_embedding_pt(args.protein_pt, expected_dim=1280)
    dna_ids, dna_all = load_embedding_pt(args.dna_pt, expected_dim=768)
    train = contract.read_ids(args.train_ids)
    test = contract.read_ids(args.test_ids)
    requested = train | test
    if train & test:
        raise ValueError('Authoritative train/test overlap')
    for name, source_ids in [('Protein', protein_ids), ('DNA', dna_ids)]:
        if len(source_ids) != len(set(source_ids)) or requested - set(source_ids):
            raise ValueError(f'{name}: duplicate IDs or missing authoritative IDs')
    ids = [x for x in protein_ids if x in requested]
    pindex = {x:i for i,x in enumerate(protein_ids)}
    dindex = {x:i for i,x in enumerate(dna_ids)}
    protein = protein_all[[pindex[x] for x in ids]].contiguous()
    dna = dna_all[[dindex[x] for x in ids]].contiguous()
    labels = read_labels(structure_path)
    # Preserve supplied ARG-only structure convention: unlisted IDs are nonARG.
    subtypes = [labels.get(x, NONARG) for x in ids]
    mapping = contract.type_mapping(structure_path, subtypes)
    print(f'Selected IDs: {len(ids)}; external train={len(train)}, test={len(test)}; ARG subtype count={len(mapping)}; Type count={len(set(mapping.values()))}', flush=True)
    print(f'IDs absent from structure, treated as nonARG: {sum(x not in labels for x in ids)}', flush=True)
    return ids, subtypes, ['train' if x in train else 'test' for x in ids], protein, dna


# =============================================================================
# Internal validation, train only
# =============================================================================

def derive_validation(ids, subtypes, external_splits, validation_fraction, seed):
    train = {x for x,s in zip(ids, external_splits) if s == 'train'}
    test = {x for x,s in zip(ids, external_splits) if s == 'test'}
    model, _ = contract.build_model_splits_from_new_ids(ids, subtypes, train, test, validation_fraction, seed)
    print('Exact model split counts:', dict(Counter(model)), flush=True)
    return model


# =============================================================================
# Model
# =============================================================================

class SupervisedGatedFusion(
    nn.Module
):

    def __init__(
        self,
        protein_dim,
        dna_dim,
        fusion_dim,
        number_subtypes,
        dropout,
        protein_mean,
        protein_std,
        dna_mean,
        dna_std,
    ):

        super().__init__()

        self.register_buffer(
            "protein_mean",
            protein_mean,
        )

        self.register_buffer(
            "protein_std",
            protein_std,
        )

        self.register_buffer(
            "dna_mean",
            dna_mean,
        )

        self.register_buffer(
            "dna_std",
            dna_std,
        )

        self.protein_projection = (
            nn.Sequential(
                nn.Linear(
                    protein_dim,
                    fusion_dim,
                ),
                nn.LayerNorm(
                    fusion_dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
            )
        )

        self.dna_projection = (
            nn.Sequential(
                nn.Linear(
                    dna_dim,
                    fusion_dim,
                ),
                nn.LayerNorm(
                    fusion_dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
            )
        )

        self.gate = (
            nn.Sequential(
                nn.Linear(
                    fusion_dim * 2,
                    fusion_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
                nn.Linear(
                    fusion_dim,
                    fusion_dim,
                ),
                nn.Sigmoid(),
            )
        )

        self.fusion_norm = (
            nn.LayerNorm(
                fusion_dim
            )
        )

        self.subtype_classifier = (
            nn.Linear(
                fusion_dim,
                number_subtypes,
            )
        )

    def standardize(
        self,
        protein,
        dna,
    ):

        return (
            (
                protein
                - self.protein_mean
            )
            / self.protein_std,

            (
                dna
                - self.dna_mean
            )
            / self.dna_std,
        )

    def encode(
        self,
        protein,
        dna,
    ):

        protein, dna = (
            self.standardize(
                protein,
                dna,
            )
        )

        protein_projected = (
            self.protein_projection(
                protein
            )
        )

        dna_projected = (
            self.dna_projection(
                dna
            )
        )

        gate = self.gate(
            torch.cat(
                [
                    protein_projected,
                    dna_projected,
                ],
                dim=1,
            )
        )

        fused = self.fusion_norm(
            gate
            * protein_projected
            + (
                1.0
                - gate
            )
            * dna_projected
        )

        return (
            fused,
            gate,
        )

    def forward(
        self,
        protein,
        dna,
    ):

        fused, gate = (
            self.encode(
                protein,
                dna,
            )
        )

        logits = (
            self.subtype_classifier(
                fused
            )
        )

        return (
            logits,
            fused,
            gate,
        )


# =============================================================================
# Metrics
# =============================================================================

def calculate_class_weights(
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
            f"Classes absent from model train: "
            f"{missing[:20]}"
        )

    weights = counts.rsqrt()

    return (
        weights
        / weights.mean()
    )


def classification_metrics(
    true_values,
    predicted_values,
    number_classes,
):

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
            true_values,
            predicted_values,
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

    accuracy = (
        float(
            tp.sum()
            / total
        )
        if total
        else 0.0
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

    active = (
        support
        > 0
    )

    return {
        "accuracy":
            accuracy,

        "macro_precision":
            float(
                precision[
                    active
                ].mean()
            )
            if bool(
                active.any()
            )
            else 0.0,

        "macro_recall":
            float(
                recall[
                    active
                ].mean()
            )
            if bool(
                active.any()
            )
            else 0.0,

        "macro_f1":
            float(
                f1[
                    active
                ].mean()
            )
            if bool(
                active.any()
            )
            else 0.0,

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

        "support":
            int(
                total
            ),

        "active_classes":
            int(
                active.sum()
            ),
    }


# =============================================================================
# GPU-resident minibatching
# =============================================================================

def batches_from_indices(
    indices,
    batch_size,
    shuffle,
):

    if shuffle:

        order = indices[
            torch.randperm(
                len(
                    indices
                ),
                device=indices.device,
            )
        ]

    else:

        order = indices

    for start in range(
        0,
        len(
            order
        ),
        batch_size,
    ):

        yield order[
            start:
            start
            + batch_size
        ]


@torch.no_grad()
def evaluate(
    model,
    protein,
    dna,
    targets,
    indices,
    criterion,
    device_type,
    use_amp,
    number_classes,
    batch_size,
):

    model.eval()

    total_loss = 0.0
    total_count = 0

    all_true = []
    all_predicted = []

    for batch_indices in batches_from_indices(
        indices,
        batch_size,
        shuffle=False,
    ):

        protein_batch = (
            protein
            .index_select(
                0,
                batch_indices,
            )
        )

        dna_batch = (
            dna
            .index_select(
                0,
                batch_indices,
            )
        )

        target_batch = (
            targets
            .index_select(
                0,
                batch_indices,
            )
        )

        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits, _, _ = (
                model(
                    protein_batch,
                    dna_batch,
                )
            )

            loss = criterion(
                logits,
                target_batch,
            )

        count = int(
            target_batch.shape[
                0
            ]
        )

        total_loss += (
            float(
                loss.item()
            )
            * count
        )

        total_count += count

        all_true.extend(
            target_batch
            .detach()
            .cpu()
            .tolist()
        )

        all_predicted.extend(
            logits
            .argmax(
                dim=1
            )
            .detach()
            .cpu()
            .tolist()
        )

    result = classification_metrics(
        np.asarray(
            all_true,
            dtype=np.int64,
        ),
        np.asarray(
            all_predicted,
            dtype=np.int64,
        ),
        number_classes,
    )

    result[
        "loss"
    ] = (
        total_loss
        / max(
            total_count,
            1,
        )
    )

    return result


@torch.no_grad()
def generate_fused(
    model,
    protein,
    dna,
    batch_size,
    device_type,
    use_amp,
    hb,
):

    model.eval()

    number = int(
        protein.shape[
            0
        ]
    )

    fusion_dim = (
        model
        .subtype_classifier
        .in_features
    )

    fused_cpu = torch.empty(
        (
            number,
            fusion_dim,
        ),
        dtype=torch.float32,
    )

    gate_cpu = torch.empty(
        number,
        dtype=torch.float32,
    )

    full_gate_cpu = torch.empty((number, fusion_dim), dtype=torch.float32)

    total_batches = math.ceil(
        number
        / batch_size
    )

    for batch_no, start in enumerate(
        range(
            0,
            number,
            batch_size,
        ),
        start=1,
    ):

        end = min(
            start
            + batch_size,
            number,
        )

        hb.update(
            phase="generate_fused",
            batch=batch_no,
            total_batches=total_batches,
            note=(
                f"records={start:,}-{end:,}"
            ),
        )

        with torch.autocast(
            device_type=device_type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            fused, gate = model.encode(
                protein[
                    start:
                    end
                ],
                dna[
                    start:
                    end
                ],
            )

        full_gate_cpu[start:end] = gate.float().cpu()

        fused_cpu[
            start:
            end
        ] = (
            fused
            .float()
            .cpu()
        )

        gate_cpu[
            start:
            end
        ] = (
            gate
            .float()
            .mean(
                dim=1
            )
            .cpu()
        )

    return (
        fused_cpu
        .contiguous(),

        gate_cpu
        .contiguous(),
        full_gate_cpu.contiguous(),
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()
    print("Device:", "cuda" if torch.cuda.is_available() else "CPU (CUDA required)", flush=True)
    print("Resolved arguments:", vars(args), flush=True)

    args.protein_pt = (
        args.protein_pt
        .expanduser()
        .resolve()
    )

    args.dna_pt = (
        args.dna_pt
        .expanduser()
        .resolve()
    )

    args.train_ids = (
        args.train_ids
        .expanduser()
        .resolve()
    )

    args.test_ids = (
        args.test_ids
        .expanduser()
        .resolve()
    )

    args.output = (
        args.output
        .expanduser()
        .resolve()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable. "
            "This training script does not fall back to CPU."
        )

    if not (
        0.0
        < args.validation_fraction
        < 0.5
    ):

        raise ValueError(
            "--validation-fraction must be between 0 and 0.5"
        )

    if (
        args.batch_size < 1
        or args.eval_batch_size < 1
        or args.epochs < 1
        or args.fusion_dim < 1
    ):

        raise ValueError(
            "Invalid batch/epoch/fusion dimension."
        )

    if (
        args.output.exists()
        and not args.overwrite
    ):

        raise FileExistsError(
            f"Output exists: {args.output}\n"
            "Use --overwrite to replace it."
        )

    structure_path = (
        find_structure(
            args.structure
        )
    )

    torch.cuda.set_device(
        0
    )

    device = torch.device(
        "cuda:0"
    )

    set_seed(
        args.seed,
        args.deterministic,
    )

    # A100 throughput settings.
    torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):

        torch.set_float32_matmul_precision(
            "high"
        )

    use_amp = (
        not args.no_amp
    )

    hb = Heartbeat(
        args.log_interval
    )

    hb.start()

    total_started = time.time()

    try:

        print(
            "=" * 112,
            flush=True,
        )

        print(
            "LONG / FULL-LENGTH SUBTYPE-SUPERVISED GATED FUSION",
            flush=True,
        )

        print(
            "=" * 112,
            flush=True,
        )

        print(
            f"Script version : "
            f"{SCRIPT_VERSION}",
            flush=True,
        )

        print(
            f"Protein PT     : "
            f"{args.protein_pt}",
            flush=True,
        )

        print(
            f"DNA PT         : "
            f"{args.dna_pt}",
            flush=True,
        )

        print(
            f"Train IDs      : "
            f"{args.train_ids}",
            flush=True,
        )

        print(
            f"Test IDs       : "
            f"{args.test_ids}",
            flush=True,
        )

        print(
            f"Structure      : "
            f"{structure_path}",
            flush=True,
        )

        print(
            f"Output         : "
            f"{args.output}",
            flush=True,
        )

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

        print(
            f"AMP            : "
            f"{use_amp}",
            flush=True,
        )

        print(
            f"TF32           : True",
            flush=True,
        )

        print(
            f"Batch size     : "
            f"{args.batch_size:,}",
            flush=True,
        )

        print(
            f"Eval batch     : "
            f"{args.eval_batch_size:,}",
            flush=True,
        )

        print(
            f"Deterministic  : "
            f"{args.deterministic}",
            flush=True,
        )

        hb.update(
            phase="load_and_align",
            note="loading ESM-2/DNABERT-2 PTs and split IDs",
        )

        (
            ids,
            subtypes,
            external_splits,
            protein_cpu,
            dna_cpu,
        ) = align_inputs(
            args,
            structure_path,
        )

        hb.update(
            phase="derive_validation",
            note="validation derived only from external train",
        )

        model_splits = (
            derive_validation(
                ids,
                subtypes,
                external_splits,
                args.validation_fraction,
                args.seed,
            )
        )

        split_indices_np = {
            name:
                np.flatnonzero(
                    np.asarray(
                        model_splits,
                        dtype=object,
                    )
                    == name
                )
            for name
            in (
                "train",
                "validation",
                "test",
            )
        }

        print(
            "\n===== LEAKAGE-FREE SPLITS =====",
            flush=True,
        )

        for name, indices in split_indices_np.items():

            subtype_count = len(
                {
                    subtypes[
                        i
                    ]
                    for i
                    in indices
                }
            )

            print(
                f"{name:10s}: "
                f"records="
                f"{len(indices):,} | "
                f"subtypes="
                f"{subtype_count:,}",
                flush=True,
            )

        # Label map from model-train only.
        train_subtypes = sorted(
            {
                subtypes[
                    i
                ]
                for i
                in split_indices_np[
                    "train"
                ]
            }
        )

        if NONARG in train_subtypes:

            train_subtypes.remove(
                NONARG
            )

            train_subtypes.insert(
                0,
                NONARG,
            )

        subtype_to_index = {
            label:
                index
            for index, label
            in enumerate(
                train_subtypes
            )
        }

        unseen = sorted(
            set(
                subtypes
            )
            - set(
                subtype_to_index
            )
        )

        if unseen:

            raise ValueError(
                "Validation/test contain classes absent from model train: "
                f"{unseen[:20]}"
            )

        targets_cpu = torch.tensor(
            [
                subtype_to_index[
                    label
                ]
                for label
                in subtypes
            ],
            dtype=torch.long,
        )

        # ---------------------------------------------------------------------
        # Train-only normalization.
        # ---------------------------------------------------------------------

        train_idx_cpu = torch.as_tensor(
            split_indices_np[
                "train"
            ],
            dtype=torch.long,
        )

        protein_train = (
            protein_cpu
            .index_select(
                0,
                train_idx_cpu,
            )
        )

        dna_train = (
            dna_cpu
            .index_select(
                0,
                train_idx_cpu,
            )
        )

        protein_mean = (
            protein_train
            .mean(
                dim=0
            )
        )

        protein_std = (
            protein_train
            .std(
                dim=0,
                unbiased=False,
            )
            .clamp_min(
                1e-6
            )
        )

        dna_mean = (
            dna_train
            .mean(
                dim=0
            )
        )

        dna_std = (
            dna_train
            .std(
                dim=0,
                unbiased=False,
            )
            .clamp_min(
                1e-6
            )
        )

        del protein_train
        del dna_train

        # ---------------------------------------------------------------------
        # Move complete dataset to A100 once.
        # ---------------------------------------------------------------------

        hb.update(
            phase="move_to_gpu",
            note="moving full aligned tensors to A100 once",
        )

        protein = (
            protein_cpu
            .to(
                device,
                non_blocking=True,
            )
        )

        dna = (
            dna_cpu
            .to(
                device,
                non_blocking=True,
            )
        )

        targets = (
            targets_cpu
            .to(
                device,
                non_blocking=True,
            )
        )

        del protein_cpu
        del dna_cpu

        split_indices = {
            name:
                torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=device,
                )
            for name, indices
            in split_indices_np.items()
        }

        model = SupervisedGatedFusion(
            protein_dim=int(
                protein.shape[
                    1
                ]
            ),
            dna_dim=int(
                dna.shape[
                    1
                ]
            ),
            fusion_dim=args.fusion_dim,
            number_subtypes=len(
                subtype_to_index
            ),
            dropout=args.dropout,
            protein_mean=(
                protein_mean
                .to(
                    device
                )
            ),
            protein_std=(
                protein_std
                .to(
                    device
                )
            ),
            dna_mean=(
                dna_mean
                .to(
                    device
                )
            ),
            dna_std=(
                dna_std
                .to(
                    device
                )
            ),
        ).to(
            device
        )

        class_weights = (
            calculate_class_weights(
                targets_cpu
                .index_select(
                    0,
                    train_idx_cpu,
                ),
                len(
                    subtype_to_index
                ),
            )
            .to(
                device
            )
        )

        criterion = (
            nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=(
                    args.label_smoothing
                ),
            )
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=(
                    args.weight_decay
                ),
            )
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            )
        )

        try:

            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=use_amp,
            )

        except Exception:

            scaler = torch.cuda.amp.GradScaler(
                enabled=use_amp,
            )

        print(
            "\n===== MODEL =====",
            flush=True,
        )

        print(
            f"Protein dim       : "
            f"{protein.shape[1]}",
            flush=True,
        )

        print(
            f"DNA dim           : "
            f"{dna.shape[1]}",
            flush=True,
        )

        print(
            f"Fusion dim        : "
            f"{args.fusion_dim}",
            flush=True,
        )

        print(
            f"Subtype classes   : "
            f"{len(subtype_to_index):,}",
            flush=True,
        )

        print(
            f"GPU allocated     : "
            f"{torch.cuda.memory_allocated(0)/(1024**3):.3f} GB",
            flush=True,
        )

        print(
            f"GPU reserved      : "
            f"{torch.cuda.memory_reserved(0)/(1024**3):.3f} GB",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------

        best_score = (
            -math.inf
        )

        best_epoch = (
            -1
        )

        best_state = (
            None
        )

        no_improvement = (
            0
        )

        history = []

        train_started = time.time()

        total_train_batches = (
            math.ceil(
                len(
                    split_indices[
                        "train"
                    ]
                )
                / args.batch_size
            )
        )

        print(
            "\n===== TRAIN SUBTYPE-SUPERVISED GATE =====",
            flush=True,
        )

        for epoch in range(
            1,
            args.epochs
            + 1,
        ):

            model.train()

            train_loss_sum = (
                0.0
            )

            train_count = (
                0
            )

            epoch_started = (
                time.time()
            )

            hb.update(
                phase="training",
                epoch=epoch,
                batch=0,
                total_batches=(
                    total_train_batches
                ),
                note="epoch started",
            )

            for batch_no, batch_indices in enumerate(
                batches_from_indices(
                    split_indices[
                        "train"
                    ],
                    args.batch_size,
                    shuffle=True,
                ),
                start=1,
            ):

                hb.update(
                    phase="training",
                    epoch=epoch,
                    batch=batch_no,
                    total_batches=(
                        total_train_batches
                    ),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                protein_batch = (
                    protein
                    .index_select(
                        0,
                        batch_indices,
                    )
                )

                dna_batch = (
                    dna
                    .index_select(
                        0,
                        batch_indices,
                    )
                )

                target_batch = (
                    targets
                    .index_select(
                        0,
                        batch_indices,
                    )
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):

                    logits, _, _ = (
                        model(
                            protein_batch,
                            dna_batch,
                        )
                    )

                    loss = criterion(
                        logits,
                        target_batch,
                    )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                count = int(
                    target_batch.shape[
                        0
                    ]
                )

                train_loss_sum += (
                    float(
                        loss.item()
                    )
                    * count
                )

                train_count += (
                    count
                )

            hb.update(
                phase="validation",
                epoch=epoch,
                batch=0,
                total_batches=0,
                note="evaluating internal validation only",
            )

            val_metrics = evaluate(
                model,
                protein,
                dna,
                targets,
                split_indices[
                    "validation"
                ],
                criterion,
                "cuda",
                use_amp,
                len(
                    subtype_to_index
                ),
                args.eval_batch_size,
            )

            train_loss = (
                train_loss_sum
                / max(
                    train_count,
                    1,
                )
            )

            val_score = float(
                val_metrics[
                    "macro_f1"
                ]
            )

            scheduler.step(
                val_score
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

                    "train_loss":
                        train_loss,

                    "val_loss":
                        val_metrics[
                            "loss"
                        ],

                    "val_accuracy":
                        val_metrics[
                            "accuracy"
                        ],

                    "val_macro_precision":
                        val_metrics[
                            "macro_precision"
                        ],

                    "val_macro_recall":
                        val_metrics[
                            "macro_recall"
                        ],

                    "val_macro_f1":
                        val_metrics[
                            "macro_f1"
                        ],

                    "val_weighted_f1":
                        val_metrics[
                            "weighted_f1"
                        ],

                    "val_mcc":
                        val_metrics[
                            "mcc"
                        ],

                    "val_active_classes":
                        val_metrics[
                            "active_classes"
                        ],

                    "learning_rate":
                        current_lr,

                    "epoch_seconds":
                        epoch_seconds,
                }
            )

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_ACC={val_metrics['accuracy']:.4f} | "
                f"val_P={val_metrics['macro_precision']:.4f} | "
                f"val_R={val_metrics['macro_recall']:.4f} | "
                f"val_macro_F1={val_metrics['macro_f1']:.4f} | "
                f"val_weighted_F1={val_metrics['weighted_f1']:.4f} | "
                f"val_MCC={val_metrics['mcc']:.4f} | "
                f"classes={val_metrics['active_classes']} | "
                f"lr={current_lr:.2e} | "
                f"time={epoch_seconds:.2f}s",
                flush=True,
            )

            if (
                val_score
                > best_score
                + 1e-6
            ):

                best_score = (
                    val_score
                )

                best_epoch = (
                    epoch
                )

                no_improvement = (
                    0
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

                print(
                    f"  -> NEW BEST "
                    f"epoch={best_epoch}, "
                    f"val_macro_F1="
                    f"{best_score:.6f}",
                    flush=True,
                )

            else:

                no_improvement += (
                    1
                )

                if (
                    no_improvement
                    >= args.patience
                ):

                    print(
                        f"Early stopping at "
                        f"epoch {epoch}; "
                        f"best_epoch="
                        f"{best_epoch}",
                        flush=True,
                    )

                    break

        if best_state is None:

            raise RuntimeError(
                "No valid best model produced."
            )

        # ---------------------------------------------------------------------
        # Final evaluation only now touches external test.
        # ---------------------------------------------------------------------

        hb.update(
            phase="load_best_model",
            epoch=best_epoch,
            note="model selection complete; external test can now be evaluated",
        )

        model.load_state_dict(
            best_state
        )

        model.to(
            device
        )

        final_metrics = {}

        for split_name in (
            "train",
            "validation",
            "test",
        ):

            hb.update(
                phase=(
                    f"final_eval_{split_name}"
                ),
                epoch=best_epoch,
                note="final best model",
            )

            final_metrics[
                split_name
            ] = evaluate(
                model,
                protein,
                dna,
                targets,
                split_indices[
                    split_name
                ],
                criterion,
                "cuda",
                use_amp,
                len(
                    subtype_to_index
                ),
                args.eval_batch_size,
            )

        # ---------------------------------------------------------------------
        # Generate all fused embeddings in original PT order.
        # ---------------------------------------------------------------------

        fused_embeddings, protein_gate_mean, protein_gate_weights = (
            generate_fused(
                model,
                protein,
                dna,
                args.eval_batch_size,
                "cuda",
                use_amp,
                hb,
            )
        )

        elapsed_minutes = (
            time.time()
            - train_started
        ) / 60.0

        output = {
            "ids":
                ids,

            "original_ids":
                ids,

            "embeddings":
                fused_embeddings,

            "splits":
                external_splits,

            "model_splits":
                model_splits,

            "subtypes":
                subtypes,

            "subtype_to_index":
                subtype_to_index,

            "gate_mean_protein_weight":
                protein_gate_mean,

            "gate_mean_DNA_weight":
                (
                    1.0
                    - protein_gate_mean
                ),

            "fusion_method":
                "Subtype-supervised gated fusion",

            "fusion_formula":
                (
                    "gate*protein_projection + "
                    "(1-gate)*DNA_projection"
                ),

            "protein_input_dimension":
                int(
                    protein.shape[
                        1
                    ]
                ),

            "DNA_input_dimension":
                int(
                    dna.shape[
                        1
                    ]
                ),

            "embedding_dimension":
                int(
                    fused_embeddings.shape[
                        1
                    ]
                ),

            "num_sequences":
                len(
                    ids
                ),

            "num_subtypes":
                len(
                    subtype_to_index
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "best_validation_macro_f1":
                float(
                    best_score
                ),

            "metrics":
                final_metrics,

            "history":
                history,

            # Store trained gate in the same single PT for reproducibility.
            "fusion_model_state_dict":
                best_state,

            "fusion_configuration":
                {
                    "fusion_dim":
                        args.fusion_dim,

                    "dropout":
                        args.dropout,

                    "label_smoothing":
                        args.label_smoothing,

                    "learning_rate":
                        args.learning_rate,

                    "weight_decay":
                        args.weight_decay,

                    "validation_fraction":
                        args.validation_fraction,

                    "seed":
                        args.seed,

                    "batch_size":
                        args.batch_size,

                    "eval_batch_size":
                        args.eval_batch_size,

                    "AMP":
                        use_amp,

                    "TF32":
                        True,

                    "deterministic":
                        args.deterministic,
                },

            "source_protein_pt":
                str(
                    args.protein_pt
                ),

            "source_DNA_pt":
                str(
                    args.dna_pt
                ),

            "source_train_ids":
                str(
                    args.train_ids
                ),

            "source_test_ids":
                str(
                    args.test_ids
                ),

            "source_structure":
                str(
                    structure_path
                ),

            "script_version":
                SCRIPT_VERSION,
        }

        hb.update(
            phase="saving",
            note=str(
                args.output
            ),
        )

        output['gate_weights_protein'] = protein_gate_weights
        output['gate_weights_DNA'] = 1.0 - protein_gate_weights
        contract.stamp_fusion(output, structure_path, args.train_ids, args.test_ids, args.output)

        safe_save(
            output,
            args.output,
        )

        print(
            "\n"
            + "=" * 112,
            flush=True,
        )

        print(
            "LONG SUPERVISED GATED FUSION COMPLETED",
            flush=True,
        )

        print(
            "=" * 112,
            flush=True,
        )

        print(
            f"Best epoch               : "
            f"{best_epoch}",
            flush=True,
        )

        print(
            f"Best validation macro F1 : "
            f"{best_score:.6f}",
            flush=True,
        )

        print(
            f"Test accuracy            : "
            f"{final_metrics['test']['accuracy']:.6f}",
            flush=True,
        )

        print(
            f"Test macro precision     : "
            f"{final_metrics['test']['macro_precision']:.6f}",
            flush=True,
        )

        print(
            f"Test macro recall        : "
            f"{final_metrics['test']['macro_recall']:.6f}",
            flush=True,
        )

        print(
            f"Test macro F1            : "
            f"{final_metrics['test']['macro_f1']:.6f}",
            flush=True,
        )

        print(
            f"Test weighted F1         : "
            f"{final_metrics['test']['weighted_f1']:.6f}",
            flush=True,
        )

        print(
            f"Test MCC                 : "
            f"{final_metrics['test']['mcc']:.6f}",
            flush=True,
        )

        print(
            f"Fused embedding shape    : "
            f"{tuple(fused_embeddings.shape)}",
            flush=True,
        )

        print(
            f"Mean protein gate        : "
            f"{protein_gate_mean.mean().item():.6f}",
            flush=True,
        )

        print(
            f"Mean DNA gate            : "
            f"{(1.0-protein_gate_mean.mean()).item():.6f}",
            flush=True,
        )

        print(
            f"Output PT                : "
            f"{args.output}",
            flush=True,
        )

        print(
            f"Training minutes         : "
            f"{elapsed_minutes:.3f}",
            flush=True,
        )

        print(
            f"Total script minutes     : "
            f"{(time.time()-total_started)/60.0:.3f}",
            flush=True,
        )

    finally:

        hb.stop()


if __name__ == "__main__":
    main()
