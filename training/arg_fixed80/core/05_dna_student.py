#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the full-length / long DNA-only hierarchical student.

Training-only inputs
--------------------
1. Official DNABERT-2 mean-pooled full-length DNA embeddings [N, 768].
2. Full-length supervised gated-fusion embeddings [N, 512].
3. Residual-GAT refined ARG subtype prototypes [n_ARG_subtypes, 512].

Inference-only input
--------------------
Only one DNABERT-2 DNA embedding [768] is required.

Model
-----
DNA 768D
 -> train-only standardization
 -> Linear(768, 1024) + LN + GELU + Dropout
 -> 2 residual blocks
 -> Linear(1024, 512) + LN
 -> L2 normalization
 -> binary cosine head: nonARG / ARG
 -> ARG subtype cosine head: all ARG subtypes

Training
--------
Stage 1: Fusion teacher
    Fixed 512D gated-fusion representation.
    Train binary + ARG-subtype cosine heads.
    ARG head initialized from refined residual-GAT prototypes.

Stage 2: DNA-only student
    Distillation from the fusion teacher.
    Refined subtype-prototype alignment.
    Separate train-only nonARG center.
    Inference uses DNA only.

Split / leakage policy
----------------------
* train_ids_subtype2.txt and test_ids_subtype2.txt are authoritative.
* validated exact split/model_splits metadata inside Long_fusion_subtype2_newsplit/long_fusion_subtype2_newsplit.pt is consumed from the new fusion artifact.
* external TEST is exactly test_ids_subtype2.txt and is never re-split.
* validation is carved only from train_ids_subtype2.txt.
* for each ARG subtype, at least one external-train sequence remains in model-train.
* the internal validation split is generated with the SAME deterministic logic
  and seed as the new Protein training script, so Protein and DNA use the same
  model-train / validation / test IDs when given the same ID files.
* normalization, class weights and optimization use model-train only.
* teacher/student model selection uses validation only.
* the final binary ARG threshold is fixed to the DNA GRID-best value (0.80).
* external TEST metrics are evaluated only after model/threshold selection.
* raw TEST-ARG Student features may be exported at predefined epochs solely
  for post-hoc UMAP/trajectory visualization and are never used for selection.

Current scientific input definition
-----------------------------------
DNABERT-2-117M official/native tokenizer, final hidden-state mean pooling over
valid tokens, 768D. This script does NOT use the old CLS embedding pipeline.

CUDA/A100 only. No CPU fallback.
"""

from __future__ import annotations

import newsplit_contract as contract

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


ROOT = Path(
    "/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding"
)

DEFAULT_DNA_PT = (
    ROOT
    / "DNABERT2_full_1to1"
    / "full_1to1_dnabert2_mean.pt"
)

DEFAULT_FUSION_PT = (
    ROOT
    / "Long_fusion_subtype2_newsplit"
    / "long_fusion_subtype2_newsplit.pt"
)

DEFAULT_REFINED_PT = (
    ROOT
    / "Refined_subtype_prototypes_subtype2_newsplit"
    / "refined_ARG_subtype_prototypes_residual_GAT.pt"
)

DEFAULT_TRAIN_IDS = ROOT / "train_ids_subtype2.txt"
DEFAULT_TEST_IDS = ROOT / "test_ids_subtype2.txt"

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "DNA_student_subtype2_newsplit_unified"
)

SCRIPT_VERSION = (
    "2026-09-03-DNA-subtype2-authoritative-id-split-v2-refined-filter"
)

NONARG = "nonARG"
FIXED_BINARY_THRESHOLD = 0.80


@dataclass
class Config:
    seed: int = 20260822

    batch_size: int = 128
    eval_batch_size: int = 512

    teacher_epochs: int = 300
    student_epochs: int = 300

    teacher_patience: int = 15
    student_patience: int = 25

    teacher_lr: float = 3.0e-4
    student_lr: float = 2.0e-4

    weight_decay: float = 5.0e-5

    hidden_dim: int = 1024
    embedding_dim: int = 512

    # GRID-best DNA configuration from test_metrics(2).json.
    dropout: float = 0.10
    temperature: float = 3.0

    logit_scale_binary: float = 16.0
    logit_scale_subtype: float = 24.0

    label_smoothing: float = 0.002
    class_weight_power: float = 0.0

    # Fusion-teacher loss.
    teacher_binary_ce: float = 1.0
    teacher_subtype_ce: float = 1.0
    teacher_feature_anchor: float = 0.10
    teacher_refined_anchor: float = 0.05

    # DNA-student loss.
    student_binary_ce: float = 1.0
    student_subtype_ce: float = 1.0
    student_binary_kd: float = 0.20
    student_subtype_kd: float = 0.05
    student_feature_align: float = 0.05
    student_prototype_align: float = 0.05
    student_head_anchor: float = 0.05

    gradient_clip: float = 1.0
    min_delta: float = 1.0e-5


# =============================================================================
# Arguments
# =============================================================================

SCRIPT_VERSION = SCRIPT_VERSION + "-unified-newsplit-v1"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train full-length DNA-only hierarchical student "
            "from gated fusion and refined ARG prototypes."
        )
    )

    parser.add_argument(
        "--dna-pt",
        type=Path,
        default=DEFAULT_DNA_PT,
    )

    parser.add_argument(
        "--fusion-pt",
        type=Path,
        default=DEFAULT_FUSION_PT,
    )

    parser.add_argument(
        "--refined-pt",
        type=Path,
        default=DEFAULT_REFINED_PT,
    )

    parser.add_argument(
        "--train-ids",
        type=Path,
        default=DEFAULT_TRAIN_IDS,
        help=(
            "Authoritative external TRAIN IDs generated by the new subtype>=2 split."
        ),
    )

    parser.add_argument(
        "--test-ids",
        type=Path,
        default=DEFAULT_TEST_IDS,
        help=(
            "Authoritative external TEST IDs generated by the new subtype>=2 split."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction carved from external TRAIN for validation. The exact same "
            "subtype-aware deterministic rule as the Protein new-split script is used."
        ),
    )

    parser.add_argument(
        "--structure",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.batch_size,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=Config.eval_batch_size,
    )

    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=Config.teacher_epochs,
    )

    parser.add_argument(
        "--student-epochs",
        type=int,
        default=Config.student_epochs,
    )

    parser.add_argument(
        "--snapshot-epochs",
        nargs="+",
        type=int,
        default=[0, 20, 50, 100, 150],
        help=(
            "Student epochs whose TEST-ARG 512D representations and "
            "student checkpoints are saved for later UMAP plotting. "
            "The current-run best epoch is always saved automatically."
        ),
    )

    parser.add_argument(
        "--fixed-binary-threshold",
        type=float,
        default=FIXED_BINARY_THRESHOLD,
        help=(
            "GRID-best ARG probability threshold. Default 0.80 from "
            "test_metrics(2).json."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=Config.seed,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
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
        self.epoch = None
        self.note = ""

    def update(
        self,
        phase=None,
        epoch=None,
        note=None,
    ):

        with self.lock:

            if phase is not None:
                self.phase = str(phase)

            if epoch is not None:
                self.epoch = int(epoch)

            if note is not None:
                self.note = str(note)

    def emit(self):

        with self.lock:
            phase = self.phase
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

            peak = (
                torch.cuda.max_memory_allocated(0)
                / (1024 ** 3)
            )

            gpu_text = (
                f"GPU_alloc={allocated:.2f}GB | "
                f"GPU_reserved={reserved:.2f}GB | "
                f"GPU_peak={peak:.2f}GB"
            )

        except Exception:

            gpu_text = "GPU_mem=NA"

        progress = ""

        if epoch is not None:
            progress = f" | epoch={epoch}"

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

    # A100 throughput mode.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if hasattr(
        torch,
        "set_float32_matmul_precision",
    ):
        torch.set_float32_matmul_precision(
            "high"
        )


def canonical_id(value: Any) -> str:

    text = str(
        value
    ).strip()

    return (
        text.split()[0]
        if text
        else ""
    )


def canonical_subtype(value: Any) -> str:

    text = str(
        value
    ).strip()

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


def is_nonarg(value: Any) -> bool:

    return canonical_subtype(
        value
    ) == NONARG


def load_pt(path: Path):

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


def require_tensor_2d(
    value,
    name,
):

    if (
        not isinstance(
            value,
            torch.Tensor,
        )
        or value.ndim != 2
    ):

        raise ValueError(
            f"{name} must be a two-dimensional torch.Tensor."
        )

    value = (
        value
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    if not bool(
        torch.isfinite(
            value
        ).all()
    ):

        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return value


def safe_mcc(
    true,
    predicted,
):

    if (
        len(true) == 0
        or len(
            np.unique(
                true
            )
        ) < 2
        or len(
            np.unique(
                predicted
            )
        ) < 2
    ):

        return 0.0

    return float(
        matthews_corrcoef(
            true,
            predicted,
        )
    )


def json_safe(value):

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key):
                json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item
            in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    return value


def write_csv(
    path: Path,
    rows,
):

    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:
                seen.add(
                    key
                )
                fieldnames.append(
                    key
                )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    os.replace(
        temporary,
        path,
    )


def save_checkpoint(
    checkpoint,
    path: Path,
):

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with temporary.open(
        "wb"
    ) as handle:

        torch.save(
            checkpoint,
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
            f"Temporary checkpoint invalid: {temporary}"
        )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# Structure / type mapping
# =============================================================================

def detect_delimiter(
    path: Path,
):

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

        if (
            not reader.fieldnames
            or "Type" not in reader.fieldnames
            or "Subtype" not in reader.fieldnames
        ):

            raise ValueError(
                f"Structure file must contain Type and Subtype columns: "
                f"{structure_path}; columns={reader.fieldnames}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):

            subtype = canonical_subtype(
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
                continue

            if is_nonarg(
                subtype
            ):

                type_name = NONARG

            if not type_name:

                raise ValueError(
                    f"Empty Type for Subtype={subtype!r} "
                    f"at structure line {line_number}."
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
                    f"{subtype!r}: {previous!r} vs {type_name!r}"
                )

            mapping[
                subtype
            ] = type_name

    # Supports ARG-only structure files as well.
    mapping[
        NONARG
    ] = NONARG

    return mapping


# =============================================================================
# Load and align current full-length data
# =============================================================================

def extract_embedding_pt(
    path: Path,
    description: str,
):

    obj = load_pt(
        path
    )

    if not isinstance(
        obj,
        dict,
    ):

        raise TypeError(
            f"{description} PT must contain a dictionary: {path}"
        )

    id_key = next(
        (
            key
            for key
            in (
                "ids",
                "sequence_ids",
                "fragment_ids",
            )
            if key in obj
        ),
        None,
    )

    emb_key = next(
        (
            key
            for key
            in (
                "embeddings",
                "mean_embeddings",
                "features",
            )
            if key in obj
        ),
        None,
    )

    if (
        id_key is None
        or emb_key is None
    ):

        raise KeyError(
            f"{description} PT requires IDs and embeddings; "
            f"keys={sorted(obj.keys())}"
        )

    ids = [
        canonical_id(
            value
        )
        for value
        in obj[
            id_key
        ]
    ]

    embeddings = require_tensor_2d(
        obj[
            emb_key
        ],
        f"{description} embeddings",
    )

    if len(
        ids
    ) != embeddings.shape[
        0
    ]:

        raise ValueError(
            f"{description} row mismatch: "
            f"IDs={len(ids)}, "
            f"embeddings={embeddings.shape[0]}"
        )

    if len(
        ids
    ) != len(
        set(
            ids
        )
    ):

        raise ValueError(
            f"{description} contains duplicate IDs."
        )

    return (
        ids,
        embeddings,
        obj,
    )


def read_id_file(path: Path):
    ids = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            seq_id = canonical_id(line)
            if not seq_id:
                continue
            if seq_id in seen:
                raise ValueError(
                    f"Duplicate ID in {path.name} at line {line_number}: {seq_id}"
                )
            seen.add(seq_id)
            ids.append(seq_id)
    if not ids:
        raise ValueError(f"Empty split ID file: {path}")
    return ids


def write_lines(path: Path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_model_splits_from_new_ids(
    ids,
    subtypes,
    train_ids_set,
    test_ids_set,
    validation_fraction,
    seed,
):
    """Build model-train/validation/test from the new authoritative ID files.

    This is intentionally the SAME algorithm used by the new Protein training
    script. External TEST is never touched. Validation is carved only from
    external TRAIN. Each ARG subtype keeps at least one sequence in model-train.
    """
    if not (0.0 < float(validation_fraction) < 0.5):
        raise ValueError(
            f"--validation-fraction must be between 0 and 0.5; "
            f"got {validation_fraction}"
        )

    rng = random.Random(int(seed))

    train_positions_by_subtype = defaultdict(list)
    train_nonarg_positions = []

    for index, (seq_id, subtype) in enumerate(zip(ids, subtypes)):
        if seq_id in test_ids_set:
            continue
        if seq_id not in train_ids_set:
            raise RuntimeError(
                f"Selected ID is in neither new train nor new test: {seq_id}"
            )

        if is_nonarg(subtype):
            train_nonarg_positions.append(index)
        else:
            train_positions_by_subtype[subtype].append(index)

    model_split = [None] * len(ids)
    external_split = [None] * len(ids)

    for index, seq_id in enumerate(ids):
        if seq_id in test_ids_set:
            external_split[index] = "test"
            model_split[index] = "test"
        elif seq_id in train_ids_set:
            external_split[index] = "train"
        else:
            raise RuntimeError(f"Unassigned selected ID: {seq_id}")

    # ARG validation: stratify within subtype, always leaving >=1 model-train row.
    for subtype in sorted(train_positions_by_subtype):
        positions = list(train_positions_by_subtype[subtype])
        rng.shuffle(positions)
        n = len(positions)

        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * float(validation_fraction)))
            n_val = max(1, n_val)
            n_val = min(n - 1, n_val)

        val_positions = set(positions[:n_val])
        for index in positions:
            model_split[index] = (
                "validation" if index in val_positions else "train"
            )

    # Match nonARG validation size to ARG validation size.
    arg_validation_count = sum(
        1
        for i, value in enumerate(model_split)
        if value == "validation" and not is_nonarg(subtypes[i])
    )

    nonarg_positions = list(train_nonarg_positions)
    rng.shuffle(nonarg_positions)
    n_nonarg = len(nonarg_positions)

    if arg_validation_count >= n_nonarg:
        raise ValueError(
            "Not enough external-train nonARG rows to match ARG validation size: "
            f"ARG validation={arg_validation_count}, nonARG train={n_nonarg}"
        )

    nonarg_val_positions = set(nonarg_positions[:arg_validation_count])
    for index in nonarg_positions:
        model_split[index] = (
            "validation" if index in nonarg_val_positions else "train"
        )

    if any(value is None for value in model_split):
        missing = [ids[i] for i, value in enumerate(model_split) if value is None]
        raise RuntimeError(
            f"Failed to assign model split to {len(missing)} IDs; "
            f"examples={missing[:10]}"
        )

    model_test_ids = {
        ids[i] for i, value in enumerate(model_split) if value == "test"
    }
    if model_test_ids != set(test_ids_set):
        raise RuntimeError(
            "Model TEST IDs do not exactly match test_ids_subtype2.txt."
        )

    leaked_test = {
        ids[i]
        for i, value in enumerate(model_split)
        if value in {"train", "validation"} and ids[i] in test_ids_set
    }
    if leaked_test:
        raise RuntimeError(
            f"TEST leakage detected; examples={sorted(leaked_test)[:10]}"
        )

    return model_split, external_split


def load_and_align_data(
    dna_path: Path,
    fusion_path: Path,
    refined_path: Path,
    structure_path: Path,
    train_ids_path: Path,
    test_ids_path: Path,
    validation_fraction: float,
    split_seed: int,
):
    dna_ids, dna_embeddings, dna_obj = extract_embedding_pt(dna_path, "DNA")
    fusion_ids, fusion_embeddings, fusion_obj = extract_embedding_pt(
        fusion_path, "fusion"
    )

    if "subtypes" not in fusion_obj:
        raise KeyError("Fusion PT missing required metadata: subtypes")

    all_subtypes = [
        canonical_subtype(value) for value in fusion_obj["subtypes"]
    ]
    if len(fusion_ids) != len(all_subtypes):
        raise ValueError("Fusion IDs/subtypes length mismatch.")

    if not set(fusion_ids).issubset(set(dna_ids)):
        raise ValueError(
            "DNA and fusion ID sets differ: "
            f"DNA={len(dna_ids):,}, fusion={len(fusion_ids):,}, "
            f"DNA_only={len(set(dna_ids)-set(fusion_ids)):,}, "
            f"fusion_only={len(set(fusion_ids)-set(dna_ids)):,}"
        )

    dna_index = {seq_id: index for index, seq_id in enumerate(dna_ids)}
    alignment = torch.tensor(
        [dna_index[seq_id] for seq_id in fusion_ids], dtype=torch.long
    )
    dna_embeddings = dna_embeddings.index_select(0, alignment).contiguous()

    if dna_embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected official DNABERT-2 768D mean embeddings; "
            f"found {dna_embeddings.shape[1]}."
        )
    if fusion_embeddings.shape[1] != 512:
        raise ValueError(
            f"Expected gated-fusion dimension 512; "
            f"found {fusion_embeddings.shape[1]}."
        )

    # ---------------------------------------------------------------------
    # NEW authoritative external train/test split.
    # Ignore any old splits/model_splits metadata inside long_fusion_subtype2_newsplit.pt.
    # ---------------------------------------------------------------------
    new_train_ids = read_id_file(train_ids_path)
    new_test_ids = read_id_file(test_ids_path)
    train_ids_set = set(new_train_ids)
    test_ids_set = set(new_test_ids)

    overlap = train_ids_set & test_ids_set
    if overlap:
        raise ValueError(
            f"New train/test ID files overlap by {len(overlap)} IDs; "
            f"examples={sorted(overlap)[:10]}"
        )

    requested_ids = train_ids_set | test_ids_set
    contract.validate_fusion(fusion_obj, train_ids_path, test_ids_path)
    available_ids = set(fusion_ids)
    missing_requested = requested_ids - available_ids
    if missing_requested:
        raise ValueError(
            f"{len(missing_requested)} IDs from the new train/test files are "
            f"absent from DNA/fusion embeddings; "
            f"examples={sorted(missing_requested)[:20]}"
        )

    excluded_source_ids = available_ids - requested_ids

    # Keep original fusion order, but retain ONLY IDs explicitly present in the
    # new train/test files. This automatically excludes singleton ARGs and the
    # excess nonARG rows removed during the new dataset split.
    selected_positions = [
        i for i, seq_id in enumerate(fusion_ids) if seq_id in requested_ids
    ]
    select_index = torch.tensor(selected_positions, dtype=torch.long)

    ids = [fusion_ids[i] for i in selected_positions]
    subtypes = [all_subtypes[i] for i in selected_positions]
    dna_embeddings = dna_embeddings.index_select(0, select_index).contiguous()
    fusion_embeddings = fusion_embeddings.index_select(0, select_index).contiguous()

    if set(ids) != requested_ids:
        raise RuntimeError(
            "Internal filtering error: selected embedding IDs do not equal "
            "new train U new test IDs."
        )

    # Strong audit: every retained ARG subtype must exist in BOTH external train
    # and external test. This is the intended subtype>=2 split policy.
    train_arg_subtypes = defaultdict(int)
    test_arg_subtypes = defaultdict(int)
    total_arg_subtypes = defaultdict(int)
    for seq_id, subtype in zip(ids, subtypes):
        if is_nonarg(subtype):
            continue
        total_arg_subtypes[subtype] += 1
        if seq_id in train_ids_set:
            train_arg_subtypes[subtype] += 1
        elif seq_id in test_ids_set:
            test_arg_subtypes[subtype] += 1

    bad_subtypes = [
        subtype
        for subtype in total_arg_subtypes
        if train_arg_subtypes[subtype] < 1 or test_arg_subtypes[subtype] < 1
    ]
    if bad_subtypes:
        examples = [
            (
                subtype,
                total_arg_subtypes[subtype],
                train_arg_subtypes[subtype],
                test_arg_subtypes[subtype],
            )
            for subtype in sorted(bad_subtypes)[:20]
        ]
        raise ValueError(
            f"{len(bad_subtypes)} retained ARG subtype(s) are not represented "
            f"in both external train and external test. Examples "
            f"(subtype,total,train,test)={examples}"
        )

    model_splits, external_splits = contract.use_saved_splits(
        obj=fusion_obj,
        ids=ids,
        subtypes=subtypes,
        train_ids_set=train_ids_set,
        test_ids_set=test_ids_set,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    refined_obj = load_pt(refined_path)
    contract.validate_derived(refined_obj, fusion_obj, fusion_path, 'refined')

    if not isinstance(refined_obj, dict):
        raise TypeError("Refined prototype PT must contain a dictionary.")

    required_refined = {
        "arg_subtype_names",
        "refined_arg_subtype_prototypes",
    }
    missing_refined_keys = sorted(required_refined - set(refined_obj))
    if missing_refined_keys:
        raise KeyError(f"Refined PT missing keys: {missing_refined_keys}")

    all_refined_names = [
        canonical_subtype(value) for value in refined_obj["arg_subtype_names"]
    ]
    all_refined = require_tensor_2d(
        refined_obj["refined_arg_subtype_prototypes"],
        "refined ARG prototypes",
    )
    if len(all_refined_names) != all_refined.shape[0] or all_refined.shape[1] != 512:
        raise ValueError(
            f"Refined prototype mismatch: names={len(all_refined_names)}, "
            f"shape={tuple(all_refined.shape)}"
        )
    if any(is_nonarg(name) for name in all_refined_names):
        raise ValueError("nonARG must not be present in ARG refined prototypes.")
    if len(all_refined_names) != len(set(all_refined_names)):
        raise ValueError("Duplicate ARG subtype names in refined prototype PT.")

    retained_arg_set = {subtype for subtype in subtypes if not is_nonarg(subtype)}
    refined_name_to_index = {
        name: index for index, name in enumerate(all_refined_names)
    }
    missing_refined = sorted(retained_arg_set - set(refined_name_to_index))
    if missing_refined:
        raise ValueError(
            f"{len(missing_refined):,} retained ARG subtype(s) lack refined "
            f"prototypes; examples={missing_refined[:20]}"
        )

    # Preserve the existing refined-prototype order, but only keep subtypes in
    # the NEW dataset.
    arg_names = [name for name in all_refined_names if name in retained_arg_set]
    refined_indices = torch.tensor(
        [refined_name_to_index[name] for name in arg_names], dtype=torch.long
    )
    refined = all_refined.index_select(0, refined_indices).contiguous()
    refined = F.normalize(refined, p=2, dim=1)

    if set(arg_names) != retained_arg_set:
        raise ValueError("Retained ARG subtype set and filtered prototype set differ.")

    print(
        "Refined prototype filtering:",
        flush=True,
    )
    print(
        f"  source refined subtypes : {len(all_refined_names):,}",
        flush=True,
    )
    print(
        f"  retained split subtypes : {len(retained_arg_set):,}",
        flush=True,
    )
    print(
        f"  selected prototypes     : {len(arg_names):,}",
        flush=True,
    )
    print(
        f"  ignored old prototypes : {len(all_refined_names) - len(arg_names):,}",
        flush=True,
    )

    if NONARG not in set(subtypes):
        raise ValueError("No retained nonARG records found.")

    subtype_to_type = load_subtype_to_type(structure_path)
    if any(subtype_to_type.get(s) != t for s, t in fusion_obj['subtype_to_type'].items()):
        raise ValueError('Student structure Type mapping differs from fusion')

    missing_type_mapping = sorted(set(subtypes) - set(subtype_to_type))
    if missing_type_mapping:
        raise ValueError(
            f"{len(missing_type_mapping):,} retained subtype(s) have no Type "
            f"mapping; examples={missing_type_mapping[:20]}"
        )

    type_names = sorted(
        {subtype_to_type[name] for name in arg_names},
        key=lambda value: value.lower(),
    )
    arg_to_index = {name: index for index, name in enumerate(arg_names)}

    binary_targets = torch.tensor(
        [0 if is_nonarg(subtype) else 1 for subtype in subtypes],
        dtype=torch.long,
    )
    arg_targets = torch.tensor(
        [-1 if is_nonarg(subtype) else arg_to_index[subtype] for subtype in subtypes],
        dtype=torch.long,
    )

    split_indices = {
        split: torch.tensor(
            [index for index, value in enumerate(model_splits) if value == split],
            dtype=torch.long,
        )
        for split in ("train", "validation", "test")
    }
    if any(indices.numel() == 0 for indices in split_indices.values()):
        raise ValueError(
            "train/validation/test must all be non-empty: "
            f"{ {k: int(v.numel()) for k, v in split_indices.items()} }"
        )

    train_arg_names = {
        subtypes[index]
        for index in split_indices["train"].tolist()
        if not is_nonarg(subtypes[index])
    }
    missing_train_arg = sorted(retained_arg_set - train_arg_names)
    if missing_train_arg:
        raise ValueError(
            f"{len(missing_train_arg):,} retained ARG subtype(s) absent from "
            f"model-train; examples={missing_train_arg[:20]}"
        )

    test_arg_names = {
        subtypes[index]
        for index in split_indices["test"].tolist()
        if not is_nonarg(subtypes[index])
    }
    missing_test_arg = sorted(retained_arg_set - test_arg_names)
    if missing_test_arg:
        raise ValueError(
            f"{len(missing_test_arg):,} retained ARG subtype(s) absent from "
            f"test; examples={missing_test_arg[:20]}"
        )

    split_manifest = []
    for local_index, seq_id in enumerate(ids):
        subtype = subtypes[local_index]
        split_manifest.append(
            {
                "sequence_id": seq_id,
                "Subtype": subtype,
                "Type": subtype_to_type[subtype],
                "external_split": external_splits[local_index],
                "model_split": model_splits[local_index],
            }
        )

    print("\n===== NEW AUTHORITATIVE DNA SPLIT =====", flush=True)
    print(f"External train IDs      : {len(train_ids_set):,} | {train_ids_path}", flush=True)
    print(f"External test IDs       : {len(test_ids_set):,} | {test_ids_path}", flush=True)
    print(f"Selected total          : {len(ids):,}", flush=True)
    print(f"Excluded source rows    : {len(excluded_source_ids):,}", flush=True)
    print(f"Retained ARG subtypes   : {len(arg_names):,}", flush=True)
    print(f"ARG Types               : {len(type_names):,}", flush=True)
    print(
        f"Validation fraction     : {float(validation_fraction):.3f} "
        "(carved ONLY from external train)",
        flush=True,
    )

    for split, indices in split_indices.items():
        local_binary = binary_targets.index_select(0, indices)
        arg_subtype_count = len(
            {
                subtypes[index]
                for index in indices.tolist()
                if not is_nonarg(subtypes[index])
            }
        )
        print(
            f"{split:10s}: sequences={indices.numel():,}, "
            f"ARG={int((local_binary==1).sum()):,}, "
            f"nonARG={int((local_binary==0).sum()):,}, "
            f"ARG_subtypes={arg_subtype_count:,}",
            flush=True,
        )

    model_train_ids = [
        ids[i] for i, value in enumerate(model_splits) if value == "train"
    ]
    validation_ids = [
        ids[i] for i, value in enumerate(model_splits) if value == "validation"
    ]
    model_test_ids = [
        ids[i] for i, value in enumerate(model_splits) if value == "test"
    ]

    return {
        "ids": ids,
        "subtypes": subtypes,
        "model_splits": model_splits,
        "external_splits": external_splits,
        "dna": dna_embeddings,
        "fusion": fusion_embeddings,
        "refined": refined,
        "arg_names": arg_names,
        "type_names": type_names,
        "subtype_to_type": subtype_to_type,
        "binary_targets": binary_targets,
        "arg_targets": arg_targets,
        "split_indices": split_indices,
        "source_dna": dna_obj,
        "source_fusion": fusion_obj,
        "source_refined": refined_obj,
        "split_manifest": split_manifest,
        "external_train_ids": list(new_train_ids),
        "external_test_ids": list(new_test_ids),
        "model_train_ids": model_train_ids,
        "validation_ids": validation_ids,
        "model_test_ids": model_test_ids,
        "excluded_source_ids": sorted(excluded_source_ids),
        "validation_fraction": float(validation_fraction),
        "split_seed": int(split_seed),
        "original_sequence_count": len(fusion_ids),
    }


def save_split_reference_artifacts(data, output_dir: Path):
    """Save the exact split actually used by this DNA run."""
    write_csv(output_dir / "split_manifest.csv", data["split_manifest"])
    write_lines(output_dir / "model_train_ids.txt", data["model_train_ids"])
    write_lines(output_dir / "validation_ids.txt", data["validation_ids"])
    write_lines(output_dir / "test_ids.txt", data["model_test_ids"])
    write_lines(output_dir / "external_train_ids.txt", data["external_train_ids"])
    write_lines(output_dir / "external_test_ids.txt", data["external_test_ids"])
    write_lines(output_dir / "excluded_source_ids.txt", data["excluded_source_ids"])


# =============================================================================
# Model definitions
# =============================================================================

def mild_class_weights(
    targets,
    number_classes,
    power,
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
            f"Class weight calculation found absent class(es): "
            f"{missing[:20]}"
        )

    weights = (
        counts.mean()
        / counts
    ).pow(
        power
    )

    weights = (
        weights
        / weights.mean()
    )

    return weights.clamp(
        0.70,
        2.50,
    )


class CosineHead(
    nn.Module
):

    def __init__(
        self,
        number_classes,
        dimension,
        scale,
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                number_classes,
                dimension,
            )
        )

        nn.init.xavier_uniform_(
            self.weight
        )

        self.scale = float(
            scale
        )

    def forward(
        self,
        features,
    ):

        return self.scale * F.linear(
            F.normalize(
                features,
                p=2,
                dim=1,
            ),
            F.normalize(
                self.weight,
                p=2,
                dim=1,
            ),
        )

    def normalized_weight(self):

        return F.normalize(
            self.weight.float(),
            p=2,
            dim=1,
        )


class FusionTeacher(
    nn.Module
):

    def __init__(
        self,
        refined,
        config,
    ):

        super().__init__()

        dimension = config.embedding_dim

        self.binary_head = CosineHead(
            2,
            dimension,
            config.logit_scale_binary,
        )

        self.arg_head = CosineHead(
            refined.shape[
                0
            ],
            dimension,
            config.logit_scale_subtype,
        )

        with torch.no_grad():

            self.arg_head.weight.copy_(
                refined
            )

    def encode(
        self,
        fusion,
    ):

        return F.normalize(
            fusion,
            p=2,
            dim=1,
        )

    def forward(
        self,
        fusion,
    ):

        z = self.encode(
            fusion
        )

        return (
            self.binary_head(
                z
            ),
            self.arg_head(
                z
            ),
            z,
        )


class ResidualBlock(
    nn.Module
):

    def __init__(
        self,
        dimension,
        dropout,
    ):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dimension,
                dimension,
            ),
            nn.LayerNorm(
                dimension
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                dimension,
                dimension,
            ),
            nn.LayerNorm(
                dimension
            ),
        )

    def forward(
        self,
        x,
    ):

        return F.gelu(
            x
            + self.net(
                x
            )
        )


class DNAStudent(
    nn.Module
):

    def __init__(
        self,
        dna_dimension,
        dna_mean,
        dna_std,
        number_arg_subtypes,
        config,
    ):

        super().__init__()

        self.register_buffer(
            "dna_mean",
            dna_mean.clone(),
        )

        self.register_buffer(
            "dna_std",
            dna_std.clone(),
        )

        self.input = nn.Sequential(
            nn.Linear(
                dna_dimension,
                config.hidden_dim,
            ),
            nn.LayerNorm(
                config.hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
        )

        self.blocks = nn.Sequential(
            ResidualBlock(
                config.hidden_dim,
                config.dropout,
            ),
            ResidualBlock(
                config.hidden_dim,
                config.dropout,
            ),
        )

        self.output = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.embedding_dim,
            ),
            nn.LayerNorm(
                config.embedding_dim
            ),
        )

        self.binary_head = CosineHead(
            2,
            config.embedding_dim,
            config.logit_scale_binary,
        )

        self.arg_head = CosineHead(
            number_arg_subtypes,
            config.embedding_dim,
            config.logit_scale_subtype,
        )

    def encode(
        self,
        dna,
    ):

        dna = (
            dna
            - self.dna_mean
        ) / self.dna_std

        hidden = self.blocks(
            self.input(
                dna
            )
        )

        return F.normalize(
            self.output(
                hidden
            ),
            p=2,
            dim=1,
        )

    def forward(
        self,
        dna,
    ):

        z = self.encode(
            dna
        )

        return (
            self.binary_head(
                z
            ),
            self.arg_head(
                z
            ),
            z,
        )


# =============================================================================
# Loss helpers
# =============================================================================

def amp_context(
    enabled,
):

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=enabled,
    )


def make_grad_scaler(
    enabled,
):

    try:

        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )

    except Exception:

        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def cosine_pull(
    features,
    targets,
):

    return (
        1.0
        - (
            F.normalize(
                features.float(),
                p=2,
                dim=1,
            )
            * F.normalize(
                targets.float(),
                p=2,
                dim=1,
            )
        ).sum(
            dim=1
        )
    ).mean()


def kd_loss(
    student,
    teacher,
    temperature,
):

    log_prob = F.log_softmax(
        student.float()
        / temperature,
        dim=1,
    )

    target_prob = F.softmax(
        teacher.float().detach()
        / temperature,
        dim=1,
    )

    return (
        F.kl_div(
            log_prob,
            target_prob,
            reduction="batchmean",
        )
        * temperature
        ** 2
    )


# =============================================================================
# GPU-resident batches
# =============================================================================

def batch_indices(
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


# =============================================================================
# Metrics
# =============================================================================

def metric_dict(
    true,
    predicted,
):

    if len(
        true
    ) == 0:

        return {
            "n": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "weighted_precision": 0.0,
            "micro_precision": 0.0,
            "macro_recall": 0.0,
            "weighted_recall": 0.0,
            "micro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "micro_f1": 0.0,
            "mcc": 0.0,
        }

    return {
        "n":
            int(
                len(
                    true
                )
            ),

        "accuracy":
            float(
                accuracy_score(
                    true,
                    predicted,
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    true,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_precision":
            float(
                precision_score(
                    true,
                    predicted,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "micro_precision":
            float(
                precision_score(
                    true,
                    predicted,
                    average="micro",
                    zero_division=0,
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    true,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_recall":
            float(
                recall_score(
                    true,
                    predicted,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "micro_recall":
            float(
                recall_score(
                    true,
                    predicted,
                    average="micro",
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    true,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    true,
                    predicted,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "micro_f1":
            float(
                f1_score(
                    true,
                    predicted,
                    average="micro",
                    zero_division=0,
                )
            ),

        "mcc":
            safe_mcc(
                true,
                predicted,
            ),
    }


def summary_rows(metrics):
    order = (
        ("ARG_nonARG", "binary"),
        ("ARG_Subtype", "ARG_subtype_oracle_binary"),
        ("End_to_end", "end_to_end_all_classes"),
    )
    rows = []
    for display_name, internal_name in order:
        value = metrics[internal_name]
        rows.append({
            "task": display_name,
            "n": int(value["n"]),
            "accuracy": float(value["accuracy"]),
            "macro_precision": float(value["macro_precision"]),
            "weighted_precision": float(value["weighted_precision"]),
            "micro_precision": float(value["micro_precision"]),
            "macro_recall": float(value["macro_recall"]),
            "weighted_recall": float(value["weighted_recall"]),
            "micro_recall": float(value["micro_recall"]),
            "macro_f1": float(value["macro_f1"]),
            "weighted_f1": float(value["weighted_f1"]),
            "micro_f1": float(value["micro_f1"]),
            "mcc": float(value["mcc"]),
        })
    return rows


@torch.inference_mode()
def collect_model_outputs(
    model,
    input_embeddings,
    indices,
    batch_size,
    use_amp,
):

    model.eval()

    binary_logits = []
    arg_logits = []
    features = []
    output_indices = []

    for local_indices in batch_indices(
        indices,
        batch_size,
        shuffle=False,
    ):

        xb = input_embeddings.index_select(
            0,
            local_indices,
        )

        with amp_context(
            use_amp
        ):

            binary, subtype, z = model(
                xb
            )

        binary_logits.append(
            binary
            .float()
            .cpu()
        )

        arg_logits.append(
            subtype
            .float()
            .cpu()
        )

        features.append(
            z
            .float()
            .cpu()
        )

        output_indices.append(
            local_indices
            .cpu()
        )

    return {
        "indices":
            torch.cat(
                output_indices
            ),

        "binary_logits":
            torch.cat(
                binary_logits
            ),

        "arg_logits":
            torch.cat(
                arg_logits
            ),

        "features":
            torch.cat(
                features
            ),
    }


def predictions_and_metrics(
    outputs,
    binary_targets_cpu,
    arg_targets_cpu,
    binary_threshold=0.50,
):

    indices = outputs[
        "indices"
    ]

    true_binary = (
        binary_targets_cpu
        .index_select(
            0,
            indices,
        )
        .numpy()
    )

    true_arg = (
        arg_targets_cpu
        .index_select(
            0,
            indices,
        )
        .numpy()
    )

    binary_prob = (
        torch.softmax(
            outputs[
                "binary_logits"
            ],
            dim=1,
        )
        .numpy()
    )

    arg_prob = (
        torch.softmax(
            outputs[
                "arg_logits"
            ],
            dim=1,
        )
        .numpy()
    )

    pred_binary = (
        binary_prob[
            :,
            1
        ]
        >= binary_threshold
    ).astype(
        np.int64
    )

    pred_arg = arg_prob.argmax(
        axis=1
    )

    true_full = np.where(
        true_binary
        == 0,
        0,
        true_arg
        + 1,
    )

    predicted_full = np.where(
        pred_binary
        == 0,
        0,
        pred_arg
        + 1,
    )

    arg_mask = (
        true_binary
        == 1
    )

    metrics = {
        "binary":
            metric_dict(
                true_binary,
                pred_binary,
            ),

        "ARG_subtype_oracle_binary":
            metric_dict(
                true_arg[
                    arg_mask
                ],
                pred_arg[
                    arg_mask
                ],
            ),

        "end_to_end_all_classes":
            metric_dict(
                true_full,
                predicted_full,
            ),
    }

    arrays = {
        "indices":
            indices.numpy(),

        "true_binary":
            true_binary,

        "pred_binary":
            pred_binary,

        "true_arg":
            true_arg,

        "pred_arg":
            pred_arg,

        "true_full":
            true_full,

        "predicted_full":
            predicted_full,

        "binary_ARG_probability":
            binary_prob[
                :,
                1
            ],

        "ARG_subtype_probability":
            arg_prob.max(
                axis=1
            ),
    }

    return (
        metrics,
        arrays,
    )


def validation_score(
    metrics,
):

    full = metrics[
        "end_to_end_all_classes"
    ]

    subtype = metrics[
        "ARG_subtype_oracle_binary"
    ]

    binary = metrics[
        "binary"
    ]

    # Keep the original precision-priority selection strategy.
    return (
        0.45
        * float(
            subtype[
                "weighted_precision"
            ]
        )
        + 0.25
        * float(
            full[
                "weighted_precision"
            ]
        )
        + 0.15
        * float(
            full[
                "accuracy"
            ]
        )
        + 0.10
        * float(
            full[
                "mcc"
            ]
        )
        + 0.05
        * float(
            binary[
                "weighted_precision"
            ]
        )
    )


def evaluate_for_selection(
    model,
    inputs,
    validation_indices,
    binary_targets_cpu,
    arg_targets_cpu,
    config,
    use_amp,
):

    outputs = collect_model_outputs(
        model,
        inputs,
        validation_indices,
        config.eval_batch_size,
        use_amp,
    )

    metrics, _ = predictions_and_metrics(
        outputs,
        binary_targets_cpu,
        arg_targets_cpu,
        binary_threshold=0.50,
    )

    return (
        validation_score(
            metrics
        ),
        metrics,
    )


# =============================================================================
# Teacher
# =============================================================================

def initialize_binary_centers(
    model,
    fusion,
    binary_targets,
    train_indices,
):

    train_fusion = F.normalize(
        fusion.index_select(
            0,
            train_indices,
        ),
        p=2,
        dim=1,
    )

    train_binary = binary_targets.index_select(
        0,
        train_indices,
    )

    centers = []

    for label in (
        0,
        1,
    ):

        selected = train_fusion[
            train_binary
            == label
        ]

        if selected.shape[
            0
        ] == 0:

            raise ValueError(
                f"Binary class {label} absent from model-train."
            )

        center = selected.mean(
            dim=0
        )

        centers.append(
            F.normalize(
                center,
                p=2,
                dim=0,
            )
        )

    with torch.no_grad():

        model.binary_head.weight.copy_(
            torch.stack(
                centers
            )
        )


def train_teacher(
    data,
    config,
    use_amp,
    hb,
    progress_dir=None,
):

    train_indices = data[
        "split_indices_gpu"
    ][
        "train"
    ]

    validation_indices = data[
        "split_indices_gpu"
    ][
        "validation"
    ]

    train_arg = data[
        "arg_targets_gpu"
    ].index_select(
        0,
        train_indices,
    )

    arg_only = train_arg[
        train_arg
        >= 0
    ]

    subtype_weights = mild_class_weights(
        arg_only,
        len(
            data[
                "arg_names"
            ]
        ),
        config.class_weight_power,
    )

    model = FusionTeacher(
        data[
            "refined_gpu"
        ],
        config,
    ).cuda()

    initialize_binary_centers(
        model,
        data[
            "fusion_gpu"
        ],
        data[
            "binary_targets_gpu"
        ],
        train_indices,
    )

    refined = data[
        "refined_gpu"
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.teacher_lr,
        weight_decay=(
            config.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1.0e-6,
        )
    )

    scaler = make_grad_scaler(
        use_amp
    )

    history = []

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    no_improvement = 0

    total_batches = math.ceil(
        len(
            train_indices
        )
        / config.batch_size
    )

    print(
        "\n===== TRAIN FUSION TEACHER =====",
        flush=True,
    )

    for epoch in range(
        1,
        config.teacher_epochs
        + 1,
    ):

        hb.update(
            phase="teacher",
            epoch=epoch,
            note=(
                f"batches={total_batches}"
            ),
        )

        epoch_started = time.time()

        model.train()

        totals = defaultdict(
            float
        )

        seen = 0

        for local_indices in batch_indices(
            train_indices,
            config.batch_size,
            shuffle=True,
        ):

            fusion = data[
                "fusion_gpu"
            ].index_select(
                0,
                local_indices,
            )

            binary_y = data[
                "binary_targets_gpu"
            ].index_select(
                0,
                local_indices,
            )

            arg_y = data[
                "arg_targets_gpu"
            ].index_select(
                0,
                local_indices,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                use_amp
            ):

                (
                    binary_logits,
                    arg_logits,
                    z,
                ) = model(
                    fusion
                )

                loss_binary = F.cross_entropy(
                    binary_logits.float(),
                    binary_y,
                    label_smoothing=(
                        config.label_smoothing
                    ),
                )

                arg_mask = (
                    arg_y
                    >= 0
                )

                if not bool(
                    arg_mask.any()
                ):

                    raise RuntimeError(
                        "Teacher batch has no ARG samples."
                    )

                loss_subtype = F.cross_entropy(
                    arg_logits[
                        arg_mask
                    ].float(),
                    arg_y[
                        arg_mask
                    ],
                    weight=subtype_weights,
                    label_smoothing=(
                        config.label_smoothing
                    ),
                )

                loss_feature = cosine_pull(
                    z,
                    F.normalize(
                        fusion,
                        p=2,
                        dim=1,
                    ),
                )

                loss_refined = (
                    1.0
                    - (
                        model.arg_head.normalized_weight()
                        * refined
                    ).sum(
                        dim=1
                    )
                ).mean()

                loss = (
                    config.teacher_binary_ce
                    * loss_binary
                    + config.teacher_subtype_ce
                    * loss_subtype
                    + config.teacher_feature_anchor
                    * loss_feature
                    + config.teacher_refined_anchor
                    * loss_refined
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_n = int(
                binary_y.shape[
                    0
                ]
            )

            seen += batch_n

            for key, value in (
                (
                    "loss",
                    loss,
                ),
                (
                    "binary",
                    loss_binary,
                ),
                (
                    "subtype",
                    loss_subtype,
                ),
                (
                    "feature",
                    loss_feature,
                ),
                (
                    "refined",
                    loss_refined,
                ),
            ):

                totals[
                    key
                ] += (
                    float(
                        value.detach()
                    )
                    * batch_n
                )

        score, validation = evaluate_for_selection(
            model,
            data[
                "fusion_gpu"
            ],
            validation_indices,
            data[
                "binary_targets_cpu"
            ],
            data[
                "arg_targets_cpu"
            ],
            config,
            use_amp,
        )

        scheduler.step(
            score
        )

        row = {
            "stage":
                "teacher",

            "epoch":
                epoch,

            "loss":
                totals[
                    "loss"
                ]
                / seen,

            "binary_loss":
                totals[
                    "binary"
                ]
                / seen,

            "subtype_loss":
                totals[
                    "subtype"
                ]
                / seen,

            "feature_loss":
                totals[
                    "feature"
                ]
                / seen,

            "refined_loss":
                totals[
                    "refined"
                ]
                / seen,

            "validation_score":
                score,

            "validation_weighted_precision":
                validation[
                    "end_to_end_all_classes"
                ][
                    "weighted_precision"
                ],

            "validation_accuracy":
                validation[
                    "end_to_end_all_classes"
                ][
                    "accuracy"
                ],

            "validation_macro_f1":
                validation[
                    "end_to_end_all_classes"
                ][
                    "macro_f1"
                ],

            "validation_mcc":
                validation[
                    "end_to_end_all_classes"
                ][
                    "mcc"
                ],

            "lr":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],

            "epoch_seconds":
                time.time()
                - epoch_started,
        }

        history.append(
            row
        )

        if progress_dir is not None:
            write_csv(
                Path(progress_dir) / "teacher_training_history.csv",
                history,
            )
            write_run_status(
                Path(progress_dir),
                "teacher_training",
                epoch=int(epoch),
                validation_score=float(score),
            )

        print(
            f"Teacher {epoch:03d}/{config.teacher_epochs} | "
            f"loss={row['loss']:.5f} | "
            f"binary={row['binary_loss']:.5f} | "
            f"subtype={row['subtype_loss']:.5f} | "
            f"val_score={score:.5f} | "
            f"val_ACC={row['validation_accuracy']:.5f} | "
            f"val_weighted_P="
            f"{row['validation_weighted_precision']:.5f} | "
            f"val_macro_F1={row['validation_macro_f1']:.5f} | "
            f"val_MCC={row['validation_mcc']:.5f} | "
            f"lr={row['lr']:.2e} | "
            f"time={row['epoch_seconds']:.2f}s",
            flush=True,
        )

        if (
            score
            > best_score
            + config.min_delta
        ):

            best_score = score
            best_epoch = epoch

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

            no_improvement = 0

            print(
                f"  -> NEW BEST TEACHER: "
                f"epoch={epoch}, "
                f"score={score:.6f}",
                flush=True,
            )

            if progress_dir is not None:
                save_checkpoint(
                    {
                        "script_version": SCRIPT_VERSION,
                        "epoch": int(best_epoch),
                        "validation_score": float(best_score),
                        "teacher_state_dict": best_state,
                        "configuration": asdict(config),
                        "arg_subtype_names": list(data["arg_names"]),
                        "validation_metrics": copy.deepcopy(validation),
                    },
                    Path(progress_dir) / "fusion_teacher_best_so_far.pt",
                )

        else:

            no_improvement += 1

        if (
            no_improvement
            >= config.teacher_patience
        ):

            print(
                f"Teacher early stopping at "
                f"epoch {epoch}; "
                f"best_epoch={best_epoch}",
                flush=True,
            )

            break

    if best_state is None:

        raise RuntimeError(
            "Fusion teacher did not produce a checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    return (
        model,
        history,
        best_epoch,
        best_score,
    )


# =============================================================================
# TEST-ARG representation snapshots for UMAP / trajectory analysis
# =============================================================================

@torch.inference_mode()
def save_test_arg_snapshot(
    model,
    data,
    config,
    use_amp,
    snapshot_dir: Path,
    epoch: int,
    tag: str,
    dna_mean,
    dna_std,
    validation_metrics=None,
    validation_score_value=None,
):
    """Save the same TEST-ARG 512D Student representations at each epoch.

    These snapshots are exported only for post-hoc visualization. They are not
    used for optimization, epoch selection, threshold selection or scheduling.
    """

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_indices_cpu = data[
        "split_indices_cpu"
    ][
        "test"
    ]

    test_binary = data[
        "binary_targets_cpu"
    ].index_select(
        0,
        test_indices_cpu,
    )

    arg_indices_cpu = test_indices_cpu[
        test_binary == 1
    ]

    arg_indices_gpu = arg_indices_cpu.cuda(
        non_blocking=True
    )

    outputs = collect_model_outputs(
        model,
        data[
            "dna_gpu"
        ],
        arg_indices_gpu,
        config.eval_batch_size,
        use_amp,
    )

    global_indices = outputs[
        "indices"
    ].tolist()

    sequence_ids = [
        data[
            "ids"
        ][index]
        for index in global_indices
    ]

    true_subtypes = [
        data[
            "subtypes"
        ][index]
        for index in global_indices
    ]

    true_types = [
        data[
            "subtype_to_type"
        ][subtype]
        for subtype in true_subtypes
    ]

    features = outputs[
        "features"
    ].float().contiguous()

    if features.ndim != 2 or features.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Snapshot feature shape mismatch: {tuple(features.shape)}; "
            f"expected (*, {config.embedding_dim})."
        )

    feature_path = (
        snapshot_dir
        / f"{tag}_test_ARG_type_features.pt"
    )

    save_checkpoint(
        {
            "epoch": int(epoch),
            "tag": str(tag),
            "split": "test",
            "ARG_only": True,
            "sequence_ids": sequence_ids,
            "true_subtypes": true_subtypes,
            "true_types": true_types,
            "features": features,
            "binary_logits": outputs[
                "binary_logits"
            ].float().contiguous(),
            "arg_subtype_logits": outputs[
                "arg_logits"
            ].float().contiguous(),
            "arg_subtype_names": list(
                data[
                    "arg_names"
                ]
            ),
            "subtype_to_type": dict(
                data[
                    "subtype_to_type"
                ]
            ),
            "feature_dimension": int(
                features.shape[1]
            ),
            "num_sequences": int(
                features.shape[0]
            ),
            "representation": (
                "DNAStudent normalized 512D z from full-length "
                "DNABERT-2 mean embedding"
            ),
            "note": (
                "Same TEST-ARG sequence set is saved at every requested "
                "epoch. No UMAP/t-SNE/PCA has been applied. These raw 512D "
                "features are for post-hoc visualization only."
            ),
            "script_version": SCRIPT_VERSION,
        },
        feature_path,
    )

    checkpoint_path = (
        snapshot_dir
        / f"{tag}_student_checkpoint.pt"
    )

    save_checkpoint(
        {
            "epoch": int(epoch),
            "tag": str(tag),
            "student_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "dna_train_mean": dna_mean.detach().cpu(),
            "dna_train_std": dna_std.detach().cpu(),
            "configuration": asdict(config),
            "fixed_binary_ARG_probability_threshold": float(
                FIXED_BINARY_THRESHOLD
            ),
            "arg_subtype_names": list(
                data[
                    "arg_names"
                ]
            ),
            "subtype_to_type": dict(
                data[
                    "subtype_to_type"
                ]
            ),
            "validation_metrics": validation_metrics,
            "validation_score": (
                None
                if validation_score_value is None
                else float(validation_score_value)
            ),
            "script_version": SCRIPT_VERSION,
        },
        checkpoint_path,
    )

    metrics_path = (
        snapshot_dir
        / f"{tag}_validation_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(
                {
                    "epoch": int(epoch),
                    "tag": str(tag),
                    "validation_score": validation_score_value,
                    "validation_metrics": validation_metrics,
                }
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[SNAPSHOT] {tag} | epoch={epoch} | "
        f"TEST ARG={features.shape[0]:,} | dim={features.shape[1]} | "
        f"features={feature_path.name}",
        flush=True,
    )

    return {
        "epoch": int(epoch),
        "tag": str(tag),
        "feature_file": str(feature_path),
        "checkpoint_file": str(checkpoint_path),
        "validation_metrics_file": str(metrics_path),
        "n_test_ARG": int(features.shape[0]),
        "dimension": int(features.shape[1]),
    }


# =============================================================================
# Student
# =============================================================================

def train_student(
    teacher,
    data,
    config,
    use_amp,
    hb,
    snapshot_epochs,
    snapshot_dir,
):

    train_indices_cpu = data[
        "split_indices_cpu"
    ][
        "train"
    ]

    train_indices = data[
        "split_indices_gpu"
    ][
        "train"
    ]

    validation_indices = data[
        "split_indices_gpu"
    ][
        "validation"
    ]

    # Train-only normalization from original FP32 DNA embeddings.
    dna_train_cpu = data[
        "dna_cpu"
    ].index_select(
        0,
        train_indices_cpu,
    )

    dna_mean = dna_train_cpu.mean(
        dim=0
    )

    dna_std = (
        dna_train_cpu
        .std(
            dim=0,
            unbiased=False,
        )
        .clamp_min(
            1.0e-6
        )
    )

    del dna_train_cpu

    train_arg_cpu = data[
        "arg_targets_cpu"
    ].index_select(
        0,
        train_indices_cpu,
    )

    arg_only_cpu = train_arg_cpu[
        train_arg_cpu
        >= 0
    ]

    subtype_weights = mild_class_weights(
        arg_only_cpu,
        len(
            data[
                "arg_names"
            ]
        ),
        config.class_weight_power,
    ).cuda()

    model = DNAStudent(
        data[
            "dna_gpu"
        ].shape[
            1
        ],
        dna_mean.cuda(),
        dna_std.cuda(),
        len(
            data[
                "arg_names"
            ]
        ),
        config,
    ).cuda()

    with torch.no_grad():

        model.binary_head.weight.copy_(
            teacher.binary_head.weight
        )

        model.arg_head.weight.copy_(
            teacher.arg_head.weight
        )

    teacher_binary_anchor = (
        teacher.binary_head
        .normalized_weight()
        .detach()
    )

    teacher_arg_anchor = (
        teacher.arg_head
        .normalized_weight()
        .detach()
    )

    refined = data[
        "refined_gpu"
    ]

    # Train-only nonARG center in teacher/fusion space.
    train_fusion = F.normalize(
        data[
            "fusion_gpu"
        ].index_select(
            0,
            train_indices,
        ),
        p=2,
        dim=1,
    )

    train_binary = data[
        "binary_targets_gpu"
    ].index_select(
        0,
        train_indices,
    )

    nonarg_features = train_fusion[
        train_binary
        == 0
    ]

    if nonarg_features.shape[
        0
    ] == 0:

        raise ValueError(
            "No model-train nonARG sequences."
        )

    nonarg_center = F.normalize(
        nonarg_features.mean(
            dim=0
        ),
        p=2,
        dim=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.student_lr,
        weight_decay=(
            config.weight_decay
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1.0e-6,
        )
    )

    scaler = make_grad_scaler(
        use_amp
    )

    history = []

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    no_improvement = 0
    snapshot_records = []
    early_stop_notified = False

    requested_snapshot_epochs = sorted(
        set(int(value) for value in snapshot_epochs)
    )

    # Persist progress incrementally. This prevents a walltime termination or
    # later evaluation error from leaving a completed training run with no
    # result files.
    progress_dir = snapshot_dir.parent
    progress_dir.mkdir(parents=True, exist_ok=True)

    teacher.eval()

    total_batches = math.ceil(
        len(
            train_indices
        )
        / config.batch_size
    )

    print(
        "\n===== TRAIN DNA-ONLY STUDENT (FULL TRAJECTORY) =====",
        flush=True,
    )

    print(
        f"Representation snapshots: {requested_snapshot_epochs} "
        "+ current-run best epoch",
        flush=True,
    )

    print(
        "Student early-stop criterion is monitored but does not terminate "
        "this trajectory run; training continues through student_epochs so "
        "all requested epoch features can be saved.",
        flush=True,
    )

    if 0 in requested_snapshot_epochs:
        snapshot_records.append(
            save_test_arg_snapshot(
                model=model,
                data=data,
                config=config,
                use_amp=use_amp,
                snapshot_dir=snapshot_dir,
                epoch=0,
                tag="epoch_000",
                dna_mean=dna_mean,
                dna_std=dna_std,
                validation_metrics=None,
                validation_score_value=None,
            )
        )

    for epoch in range(
        1,
        config.student_epochs
        + 1,
    ):

        hb.update(
            phase="student",
            epoch=epoch,
            note=(
                f"batches={total_batches}"
            ),
        )

        epoch_started = time.time()

        model.train()

        totals = defaultdict(
            float
        )

        seen = 0

        for local_indices in batch_indices(
            train_indices,
            config.batch_size,
            shuffle=True,
        ):

            dna = data[
                "dna_gpu"
            ].index_select(
                0,
                local_indices,
            )

            fusion = data[
                "fusion_gpu"
            ].index_select(
                0,
                local_indices,
            )

            binary_y = data[
                "binary_targets_gpu"
            ].index_select(
                0,
                local_indices,
            )

            arg_y = data[
                "arg_targets_gpu"
            ].index_select(
                0,
                local_indices,
            )

            with torch.no_grad():

                with amp_context(
                    use_amp
                ):

                    (
                        teacher_binary,
                        teacher_arg,
                        teacher_z,
                    ) = teacher(
                        fusion
                    )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                use_amp
            ):

                (
                    student_binary,
                    student_arg,
                    student_z,
                ) = model(
                    dna
                )

                loss_binary = F.cross_entropy(
                    student_binary.float(),
                    binary_y,
                    label_smoothing=(
                        config.label_smoothing
                    ),
                )

                arg_mask = (
                    arg_y
                    >= 0
                )

                if not bool(
                    arg_mask.any()
                ):

                    raise RuntimeError(
                        "Student batch has no ARG samples."
                    )

                loss_subtype = F.cross_entropy(
                    student_arg[
                        arg_mask
                    ].float(),
                    arg_y[
                        arg_mask
                    ],
                    weight=subtype_weights,
                    label_smoothing=(
                        config.label_smoothing
                    ),
                )

                loss_binary_kd = kd_loss(
                    student_binary,
                    teacher_binary,
                    config.temperature,
                )

                loss_subtype_kd = kd_loss(
                    student_arg[
                        arg_mask
                    ],
                    teacher_arg[
                        arg_mask
                    ],
                    config.temperature,
                )

                loss_feature = cosine_pull(
                    student_z,
                    teacher_z,
                )

                target_prototypes = torch.empty_like(
                    student_z
                )

                target_prototypes[
                    arg_mask
                ] = refined.index_select(
                    0,
                    arg_y[
                        arg_mask
                    ],
                )

                target_prototypes[
                    ~arg_mask
                ] = nonarg_center

                loss_prototype = cosine_pull(
                    student_z,
                    target_prototypes,
                )

                loss_head = (
                    1.0
                    - (
                        model.binary_head.normalized_weight()
                        * teacher_binary_anchor
                    ).sum(
                        dim=1
                    )
                ).mean() + (
                    1.0
                    - (
                        model.arg_head.normalized_weight()
                        * teacher_arg_anchor
                    ).sum(
                        dim=1
                    )
                ).mean()

                loss = (
                    config.student_binary_ce
                    * loss_binary
                    + config.student_subtype_ce
                    * loss_subtype
                    + config.student_binary_kd
                    * loss_binary_kd
                    + config.student_subtype_kd
                    * loss_subtype_kd
                    + config.student_feature_align
                    * loss_feature
                    + config.student_prototype_align
                    * loss_prototype
                    + config.student_head_anchor
                    * loss_head
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_n = int(
                binary_y.shape[
                    0
                ]
            )

            seen += batch_n

            for key, value in (
                (
                    "loss",
                    loss,
                ),
                (
                    "binary",
                    loss_binary,
                ),
                (
                    "subtype",
                    loss_subtype,
                ),
                (
                    "binary_kd",
                    loss_binary_kd,
                ),
                (
                    "subtype_kd",
                    loss_subtype_kd,
                ),
                (
                    "feature",
                    loss_feature,
                ),
                (
                    "prototype",
                    loss_prototype,
                ),
                (
                    "head",
                    loss_head,
                ),
            ):

                totals[
                    key
                ] += (
                    float(
                        value.detach()
                    )
                    * batch_n
                )

        score, validation = evaluate_for_selection(
            model,
            data[
                "dna_gpu"
            ],
            validation_indices,
            data[
                "binary_targets_cpu"
            ],
            data[
                "arg_targets_cpu"
            ],
            config,
            use_amp,
        )

        scheduler.step(
            score
        )

        row = {
            "stage":
                "student",

            "epoch":
                epoch,

            "loss":
                totals[
                    "loss"
                ]
                / seen,

            "binary_loss":
                totals[
                    "binary"
                ]
                / seen,

            "subtype_loss":
                totals[
                    "subtype"
                ]
                / seen,

            "binary_kd_loss":
                totals[
                    "binary_kd"
                ]
                / seen,

            "subtype_kd_loss":
                totals[
                    "subtype_kd"
                ]
                / seen,

            "feature_loss":
                totals[
                    "feature"
                ]
                / seen,

            "prototype_loss":
                totals[
                    "prototype"
                ]
                / seen,

            "head_loss":
                totals[
                    "head"
                ]
                / seen,

            "validation_score":
                score,

            "validation_weighted_precision":
                validation[
                    "end_to_end_all_classes"
                ][
                    "weighted_precision"
                ],

            "validation_accuracy":
                validation[
                    "end_to_end_all_classes"
                ][
                    "accuracy"
                ],

            "validation_macro_f1":
                validation[
                    "end_to_end_all_classes"
                ][
                    "macro_f1"
                ],

            "validation_mcc":
                validation[
                    "end_to_end_all_classes"
                ][
                    "mcc"
                ],

            "lr":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],

            "epoch_seconds":
                time.time()
                - epoch_started,
        }

        history.append(
            row
        )

        # Incremental training history: updated every epoch.
        write_csv(
            progress_dir / "student_training_history.csv",
            history,
        )

        print(
            f"Student {epoch:03d}/{config.student_epochs} | "
            f"loss={row['loss']:.5f} | "
            f"binary={row['binary_loss']:.5f} | "
            f"subtype={row['subtype_loss']:.5f} | "
            f"KD_bin={row['binary_kd_loss']:.5f} | "
            f"KD_sub={row['subtype_kd_loss']:.5f} | "
            f"val_score={score:.5f} | "
            f"val_ACC={row['validation_accuracy']:.5f} | "
            f"val_weighted_P="
            f"{row['validation_weighted_precision']:.5f} | "
            f"val_macro_F1={row['validation_macro_f1']:.5f} | "
            f"val_MCC={row['validation_mcc']:.5f} | "
            f"lr={row['lr']:.2e} | "
            f"time={row['epoch_seconds']:.2f}s",
            flush=True,
        )

        if (
            score
            > best_score
            + config.min_delta
        ):

            best_score = score
            best_epoch = epoch

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

            no_improvement = 0

            print(
                f"  -> NEW BEST STUDENT: "
                f"epoch={epoch}, "
                f"score={score:.6f}",
                flush=True,
            )

            # Save the current validation-selected best model immediately.
            save_checkpoint(
                {
                    "model_name": "DNA Student best-so-far during trajectory",
                    "script_version": SCRIPT_VERSION,
                    "epoch": int(best_epoch),
                    "validation_score": float(best_score),
                    "student_state_dict": best_state,
                    "dna_train_mean": dna_mean.detach().cpu(),
                    "dna_train_std": dna_std.detach().cpu(),
                    "configuration": asdict(config),
                    "fixed_binary_ARG_probability_threshold": float(FIXED_BINARY_THRESHOLD),
                    "arg_subtype_names": list(data["arg_names"]),
                    "subtype_to_type": dict(data["subtype_to_type"]),
                    "validation_metrics": copy.deepcopy(validation),
                },
                progress_dir / "DNA_student_best_so_far.pt",
            )

            with (progress_dir / "best_so_far_validation_metrics.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    json_safe({
                        "best_epoch": int(best_epoch),
                        "best_validation_score": float(best_score),
                        "validation_metrics": validation,
                    }),
                    handle, ensure_ascii=False, indent=2,
                )

        else:

            no_improvement += 1

        if epoch in requested_snapshot_epochs:
            snapshot_records.append(
                save_test_arg_snapshot(
                    model=model,
                    data=data,
                    config=config,
                    use_amp=use_amp,
                    snapshot_dir=snapshot_dir,
                    epoch=epoch,
                    tag=f"epoch_{epoch:03d}",
                    dna_mean=dna_mean,
                    dna_std=dna_std,
                    validation_metrics=copy.deepcopy(validation),
                    validation_score_value=float(score),
                )
            )

        if (
            no_improvement
            >= config.student_patience
            and not early_stop_notified
        ):

            print(
                f"Student early-stop criterion first reached at epoch {epoch}; "
                f"best_epoch={best_epoch}. Continuing to epoch "
                f"{config.student_epochs} for trajectory snapshots.",
                flush=True,
            )

            early_stop_notified = True

    if best_state is None:

        raise RuntimeError(
            "DNA student did not produce a checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    best_validation_score_value, best_validation_metrics = evaluate_for_selection(
        model,
        data[
            "dna_gpu"
        ],
        validation_indices,
        data[
            "binary_targets_cpu"
        ],
        data[
            "arg_targets_cpu"
        ],
        config,
        use_amp,
    )

    snapshot_records.append(
        save_test_arg_snapshot(
            model=model,
            data=data,
            config=config,
            use_amp=use_amp,
            snapshot_dir=snapshot_dir,
            epoch=best_epoch,
            tag=f"run_best_epoch_{best_epoch:03d}",
            dna_mean=dna_mean,
            dna_std=dna_std,
            validation_metrics=copy.deepcopy(best_validation_metrics),
            validation_score_value=float(best_validation_score_value),
        )
    )

    return (
        model,
        history,
        best_epoch,
        best_score,
        dna_mean,
        dna_std,
        snapshot_records,
    )


# =============================================================================
# Binary threshold
# =============================================================================

def select_binary_threshold_on_validation(
    model,
    data,
    config,
    use_amp,
    hb,
):

    hb.update(
        phase="threshold_selection",
        note="validation only",
    )

    outputs = collect_model_outputs(
        model,
        data[
            "dna_gpu"
        ],
        data[
            "split_indices_gpu"
        ][
            "validation"
        ],
        config.eval_batch_size,
        use_amp,
    )

    indices = outputs[
        "indices"
    ]

    true_binary = (
        data[
            "binary_targets_cpu"
        ]
        .index_select(
            0,
            indices,
        )
        .numpy()
    )

    probability = (
        torch.softmax(
            outputs[
                "binary_logits"
            ],
            dim=1,
        )[
            :,
            1
        ]
        .numpy()
    )

    rows = []

    best_threshold = 0.50
    best_score = -math.inf

    for threshold in np.round(
        np.arange(
            0.30,
            0.901,
            0.01,
        ),
        2,
    ):

        predicted = (
            probability
            >= threshold
        ).astype(
            np.int64
        )

        arg_precision = float(
            precision_score(
                true_binary,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        )

        arg_recall = float(
            recall_score(
                true_binary,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        )

        accuracy = float(
            accuracy_score(
                true_binary,
                predicted,
            )
        )

        mcc = safe_mcc(
            true_binary,
            predicted,
        )

        score = (
            0.70
            * arg_precision
            + 0.20
            * accuracy
            + 0.10
            * mcc
        )

        rows.append(
            {
                "threshold":
                    float(
                        threshold
                    ),

                "ARG_precision":
                    arg_precision,

                "ARG_recall":
                    arg_recall,

                "accuracy":
                    accuracy,

                "mcc":
                    mcc,

                "selection_score":
                    score,

                "coverage":
                    1.0,
            }
        )

        if (
            score
            > best_score
            + 1.0e-12
        ):

            best_score = score
            best_threshold = float(
                threshold
            )

    print(
        f"Selected binary ARG threshold "
        f"on validation: "
        f"{best_threshold:.2f} | "
        f"precision-priority score="
        f"{best_score:.6f} | "
        f"coverage=1.0",
        flush=True,
    )

    return (
        best_threshold,
        rows,
    )


# =============================================================================
# Final evaluation
# =============================================================================

def evaluate_split(
    model,
    split,
    data,
    config,
    use_amp,
    binary_threshold,
):

    outputs = collect_model_outputs(
        model,
        data[
            "dna_gpu"
        ],
        data[
            "split_indices_gpu"
        ][
            split
        ],
        config.eval_batch_size,
        use_amp,
    )

    metrics, arrays = predictions_and_metrics(
        outputs,
        data[
            "binary_targets_cpu"
        ],
        data[
            "arg_targets_cpu"
        ],
        binary_threshold=(
            binary_threshold
        ),
    )

    full_names = [
        NONARG
    ] + data[
        "arg_names"
    ]

    rows = []

    for local, global_index in enumerate(
        arrays[
            "indices"
        ]
    ):

        true_subtype = full_names[
            arrays[
                "true_full"
            ][
                local
            ]
        ]

        predicted_subtype = full_names[
            arrays[
                "predicted_full"
            ][
                local
            ]
        ]

        rows.append(
            {
                "split":
                    split,

                "sequence_level":
                    "full_length",

                "sequence_id":
                    data[
                        "ids"
                    ][
                        global_index
                    ],

                "true_subtype":
                    true_subtype,

                "predicted_subtype":
                    predicted_subtype,

                "true_type":
                    data[
                        "subtype_to_type"
                    ][
                        true_subtype
                    ],

                "predicted_type":
                    data[
                        "subtype_to_type"
                    ][
                        predicted_subtype
                    ],

                "true_ARG":
                    int(
                        arrays[
                            "true_binary"
                        ][
                            local
                        ]
                    ),

                "predicted_ARG":
                    int(
                        arrays[
                            "pred_binary"
                        ][
                            local
                        ]
                    ),

                "ARG_probability":
                    float(
                        arrays[
                            "binary_ARG_probability"
                        ][
                            local
                        ]
                    ),

                "ARG_subtype_max_probability":
                    float(
                        arrays[
                            "ARG_subtype_probability"
                        ][
                            local
                        ]
                    ),

                "binary_threshold":
                    float(
                        binary_threshold
                    ),

                "coverage":
                    1.0,
            }
        )

    return (
        metrics,
        rows,
    )


def one_vs_rest_class_metrics(
    true_labels,
    predicted_labels,
    class_names,
    class_column,
    split,
):

    true_array = np.asarray(
        true_labels,
        dtype=object,
    )

    predicted_array = np.asarray(
        predicted_labels,
        dtype=object,
    )

    number = len(
        true_array
    )

    rows = []

    for class_name in class_names:

        true_positive_mask = (
            true_array
            == class_name
        )

        predicted_positive_mask = (
            predicted_array
            == class_name
        )

        tp = int(
            np.sum(
                true_positive_mask
                & predicted_positive_mask
            )
        )

        fp = int(
            np.sum(
                ~true_positive_mask
                & predicted_positive_mask
            )
        )

        fn = int(
            np.sum(
                true_positive_mask
                & ~predicted_positive_mask
            )
        )

        tn = int(
            number
            - tp
            - fp
            - fn
        )

        precision = (
            tp
            / (
                tp
                + fp
            )
            if (
                tp
                + fp
            )
            else 0.0
        )

        recall = (
            tp
            / (
                tp
                + fn
            )
            if (
                tp
                + fn
            )
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            )
            else 0.0
        )

        rows.append(
            {
                class_column:
                    class_name,

                "split":
                    split,

                "sequence_level":
                    "full_length",

                "ACC":
                    (
                        (
                            tp
                            + tn
                        )
                        / number
                        if number
                        else 0.0
                    ),

                "Precision":
                    precision,

                "Recall":
                    recall,

                "F1":
                    f1,

                "MCC":
                    safe_mcc(
                        true_positive_mask.astype(
                            np.int64
                        ),
                        predicted_positive_mask.astype(
                            np.int64
                        ),
                    ),

                "support":
                    int(
                        tp
                        + fn
                    ),

                "predicted_count":
                    int(
                        tp
                        + fp
                    ),

                "correct_count":
                    tp,

                "wrong_count":
                    int(
                        fp
                        + fn
                    ),

                "TP":
                    tp,

                "FP":
                    fp,

                "FN":
                    fn,

                "TN":
                    tn,

                "total_samples":
                    number,

                "coverage":
                    1.0,
            }
        )

    return rows


def build_level_metric_tables(
    prediction_rows,
    data,
):

    subtype_rows = []
    type_rows = []
    binary_rows = []

    for split in (
        "train",
        "validation",
        "test",
    ):

        selected = [
            row
            for row
            in prediction_rows
            if row[
                "split"
            ] == split
        ]

        true_subtype = [
            row[
                "true_subtype"
            ]
            for row
            in selected
        ]

        predicted_subtype = [
            row[
                "predicted_subtype"
            ]
            for row
            in selected
        ]

        subtype_rows.extend(
            one_vs_rest_class_metrics(
                true_subtype,
                predicted_subtype,
                data[
                    "arg_names"
                ],
                "Subtype",
                split,
            )
        )

        true_type = [
            row[
                "true_type"
            ]
            for row
            in selected
        ]

        predicted_type = [
            row[
                "predicted_type"
            ]
            for row
            in selected
        ]

        type_rows.extend(
            one_vs_rest_class_metrics(
                true_type,
                predicted_type,
                data[
                    "type_names"
                ],
                "Type",
                split,
            )
        )

        true_binary = [
            (
                NONARG
                if is_nonarg(
                    value
                )
                else "ARG"
            )
            for value
            in true_subtype
        ]

        predicted_binary = [
            (
                NONARG
                if is_nonarg(
                    value
                )
                else "ARG"
            )
            for value
            in predicted_subtype
        ]

        binary_rows.extend(
            one_vs_rest_class_metrics(
                true_binary,
                predicted_binary,
                [
                    "ARG",
                    NONARG,
                ],
                "Class",
                split,
            )
        )

    return (
        subtype_rows,
        type_rows,
        binary_rows,
    )


# =============================================================================
# Run-status helper
# =============================================================================

def write_run_status(output_dir: Path, phase: str, **extra):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": str(phase),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": SCRIPT_VERSION,
        **extra,
    }
    path = output_dir / "run_status.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()
    print("Device:", "cuda" if torch.cuda.is_available() else "CPU (CUDA required)", flush=True)
    print("Resolved arguments:", vars(args), flush=True)

    args.dna_pt = (
        args.dna_pt
        .expanduser()
        .resolve()
    )

    args.fusion_pt = (
        args.fusion_pt
        .expanduser()
        .resolve()
    )

    args.refined_pt = (
        args.refined_pt
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

    args.structure = (
        args.structure
        .expanduser()
        .resolve()
    )

    args.output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    for path, name in (
        (
            args.dna_pt,
            "DNA embedding PT",
        ),
        (
            args.fusion_pt,
            "gated-fusion PT",
        ),
        (
            args.refined_pt,
            "refined prototype PT",
        ),
        (
            args.structure,
            "structure file",
        ),
        (
            args.train_ids,
            "new external train ID file",
        ),
        (
            args.test_ids,
            "new external test ID file",
        ),
    ):

        if not path.is_file():

            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

    if (
        args.output_dir.exists()
        and any(
            args.output_dir.iterdir()
        )
    ):

        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: "
                f"{args.output_dir}\n"
                "Use --overwrite to replace it."
            )

        print(
            f"[OVERWRITE] removing old output directory: {args.output_dir}",
            flush=True,
        )
        shutil.rmtree(
            args.output_dir
        )

    if (
        args.batch_size < 1
        or args.eval_batch_size < 1
        or args.teacher_epochs < 1
        or args.student_epochs < 1
    ):

        raise ValueError(
            "Batch sizes and epoch counts must be positive."
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU unavailable. "
            "CPU fallback is disabled."
        )

    torch.cuda.set_device(
        0
    )

    # Scientific hyperparameters are fixed to the DNA GRID-best values
    # reported in test_metrics(2).json. CLI values only override resource /
    # trajectory controls (batch sizes, epoch counts, seed).
    config = Config(
        seed=args.seed,
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        teacher_epochs=int(args.teacher_epochs),
        student_epochs=max(
            int(args.student_epochs),
            max(int(value) for value in args.snapshot_epochs),
        ),
    )

    set_seed(
        config.seed
    )

    use_amp = (
        not args.no_amp
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (args.output_dir / "RUN_STARTED.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"script_version={SCRIPT_VERSION}\n")
        handle.flush()
        os.fsync(handle.fileno())

    write_run_status(
        args.output_dir,
        "startup",
        dna_pt=str(args.dna_pt),
        fusion_pt=str(args.fusion_pt),
        refined_pt=str(args.refined_pt),
        structure=str(args.structure),
        train_ids=str(args.train_ids),
        test_ids=str(args.test_ids),
        validation_fraction=float(args.validation_fraction),
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
            "LONG / FULL-LENGTH DNA-ONLY HIERARCHICAL STUDENT",
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
            f"DNA PT         : "
            f"{args.dna_pt}",
            flush=True,
        )

        print(
            f"Fusion PT      : "
            f"{args.fusion_pt}",
            flush=True,
        )

        print(
            f"Refined PT     : "
            f"{args.refined_pt}",
            flush=True,
        )

        print(
            f"External train : {args.train_ids}",
            flush=True,
        )

        print(
            f"External test  : {args.test_ids}",
            flush=True,
        )

        print(
            f"Validation frac: {args.validation_fraction:.3f}",
            flush=True,
        )

        print(
            f"Structure      : "
            f"{args.structure}",
            flush=True,
        )

        print(
            f"Output dir     : "
            f"{args.output_dir}",
            flush=True,
        )

        print(
            "DNA definition : "
            "official DNABERT-2 final hidden-state mean, 768D",
            flush=True,
        )

        print(
            "Teacher input  : "
            "full-length supervised gated fusion, 512D",
            flush=True,
        )

        print(
            "Prototype prior: "
            "residual-GAT refined ARG prototypes, 512D",
            flush=True,
        )

        print(
            "Inference input : "
            "DNA embedding only",
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
            f"AMP            : "
            f"{use_amp}",
            flush=True,
        )

        print(
            "TF32           : True",
            flush=True,
        )

        print(
            f"Train batch    : "
            f"{config.batch_size:,}",
            flush=True,
        )

        print(
            f"Eval batch     : "
            f"{config.eval_batch_size:,}",
            flush=True,
        )

        print(
            f"Snapshot epochs: {sorted(set(args.snapshot_epochs))} "
            "+ current-run best epoch",
            flush=True,
        )

        print(
            f"Fixed ARG thr  : {args.fixed_binary_threshold:.2f}",
            flush=True,
        )

        print(
            "Best config    : dropout=0.10, class_weight_power=0.0, "
            "student_subtype_ce=1.0, binary_kd=0.20, subtype_kd=0.05, "
            "feature_align=0.05, prototype_align=0.05",
            flush=True,
        )

        print(
            "CPU fallback   : disabled",
            flush=True,
        )

        hb.update(
            phase="load_and_align",
            note="DNA + fusion + refined prototypes + labels",
        )
        write_run_status(args.output_dir, "load_and_align")

        data = load_and_align_data(
            args.dna_pt,
            args.fusion_pt,
            args.refined_pt,
            args.structure,
            args.train_ids,
            args.test_ids,
            args.validation_fraction,
            config.seed,
        )

        # Save the exact external and internal split used by DNA.
        save_split_reference_artifacts(
            data,
            args.output_dir,
        )

        if (
            config.embedding_dim
            != data[
                "fusion"
            ].shape[
                1
            ]
        ):

            raise ValueError(
                "Config embedding_dim must equal fusion dimension."
            )

        # Keep CPU copies for final reporting/checkpoint.
        data[
            "dna_cpu"
        ] = data[
            "dna"
        ]

        data[
            "binary_targets_cpu"
        ] = data[
            "binary_targets"
        ]

        data[
            "arg_targets_cpu"
        ] = data[
            "arg_targets"
        ]

        data[
            "split_indices_cpu"
        ] = data[
            "split_indices"
        ]

        # Move repeatedly used full dataset to GPU exactly once.
        hb.update(
            phase="move_to_gpu",
            note="moving full aligned tensors to A100 once",
        )

        data[
            "dna_gpu"
        ] = data[
            "dna"
        ].cuda(
            non_blocking=True
        )

        data[
            "fusion_gpu"
        ] = data[
            "fusion"
        ].cuda(
            non_blocking=True
        )

        data[
            "refined_gpu"
        ] = data[
            "refined"
        ].cuda(
            non_blocking=True
        )

        data[
            "binary_targets_gpu"
        ] = data[
            "binary_targets"
        ].cuda(
            non_blocking=True
        )

        data[
            "arg_targets_gpu"
        ] = data[
            "arg_targets"
        ].cuda(
            non_blocking=True
        )

        data[
            "split_indices_gpu"
        ] = {
            split:
                indices.cuda(
                    non_blocking=True
                )
            for split, indices
            in data[
                "split_indices"
            ].items()
        }

        torch.cuda.synchronize()

        print(
            f"GPU memory before teacher: "
            f"allocated="
            f"{torch.cuda.memory_allocated(0)/(1024**3):.3f} GiB, "
            f"reserved="
            f"{torch.cuda.memory_reserved(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Stage 1: fusion teacher.
        # ---------------------------------------------------------------------

        (
            teacher,
            teacher_history,
            teacher_best_epoch,
            teacher_best_score,
        ) = train_teacher(
            data,
            config,
            use_amp,
            hb,
            progress_dir=args.output_dir,
        )

        # Persist Teacher results before Student training starts.
        write_csv(
            args.output_dir / "teacher_training_history.csv",
            teacher_history,
        )
        save_checkpoint(
            {
                "script_version": SCRIPT_VERSION,
                "teacher_state_dict": {
                    key: value.detach().cpu()
                    for key, value in teacher.state_dict().items()
                },
                "teacher_best_epoch": int(teacher_best_epoch),
                "teacher_best_validation_score": float(teacher_best_score),
                "configuration": asdict(config),
                "arg_subtype_names": list(data["arg_names"]),
            },
            args.output_dir / "fusion_teacher_best.pt",
        )

        torch.cuda.empty_cache()

        print(
            f"GPU memory after teacher : "
            f"allocated="
            f"{torch.cuda.memory_allocated(0)/(1024**3):.3f} GiB, "
            f"reserved="
            f"{torch.cuda.memory_reserved(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Stage 2: DNA-only student.
        # ---------------------------------------------------------------------

        write_run_status(
            args.output_dir,
            "student_training",
            teacher_best_epoch=int(teacher_best_epoch),
            teacher_best_validation_score=float(teacher_best_score),
        )

        snapshot_dir = (
            args.output_dir
            / "epoch_features"
        )

        (
            student,
            student_history,
            student_best_epoch,
            student_best_score,
            dna_mean,
            dna_std,
            snapshot_records,
        ) = train_student(
            teacher,
            data,
            config,
            use_amp,
            hb,
            snapshot_epochs=args.snapshot_epochs,
            snapshot_dir=snapshot_dir,
        )

        write_csv(
            args.output_dir
            / "snapshot_manifest.csv",
            snapshot_records,
        )

        # Training has completed at this point. Save the selected Student
        # immediately, before any final train/validation/test reporting.
        write_csv(
            args.output_dir / "student_training_history.csv",
            student_history,
        )
        save_checkpoint(
            {
                "model_name": "DNA Student validation-selected best model before final reporting",
                "script_version": SCRIPT_VERSION,
                "student_state_dict": {
                    key: value.detach().cpu()
                    for key, value in student.state_dict().items()
                },
                "dna_train_mean": dna_mean.detach().cpu(),
                "dna_train_std": dna_std.detach().cpu(),
                "student_best_epoch": int(student_best_epoch),
                "student_best_validation_score": float(student_best_score),
                "configuration": asdict(config),
                "selected_binary_ARG_probability_threshold": float(args.fixed_binary_threshold),
                "arg_subtype_names": list(data["arg_names"]),
                "subtype_to_type": dict(data["subtype_to_type"]),
            },
            args.output_dir / "DNA_student_best_model_before_test.pt",
        )

        torch.cuda.empty_cache()

        print(
            f"GPU memory after student : "
            f"allocated="
            f"{torch.cuda.memory_allocated(0)/(1024**3):.3f} GiB, "
            f"reserved="
            f"{torch.cuda.memory_reserved(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Fixed GRID-best threshold from the DNA hyperparameter search.
        # ---------------------------------------------------------------------

        binary_threshold = float(
            args.fixed_binary_threshold
        )

        threshold_rows = [
            {
                "threshold": binary_threshold,
                "source": "DNA GRID-best test_metrics(2).json",
                "selection_policy": (
                    "fixed before this trajectory run; not recalibrated on TEST"
                ),
            }
        ]

        print(
            f"Using fixed GRID-best ARG threshold: {binary_threshold:.2f}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Final metric evaluation.
        # TEST trajectory features may already have been exported above, but
        # TEST labels/metrics are not used for model or threshold selection.
        # ---------------------------------------------------------------------

        all_metrics = {}
        prediction_rows = []

        for split in (
            "train",
            "validation",
            "test",
        ):

            hb.update(
                phase=(
                    f"final_eval_{split}"
                ),
                note=(
                    "test only occurs after all selection"
                    if split == "test"
                    else "final selected model"
                ),
            )

            metrics, rows = evaluate_split(
                student,
                split,
                data,
                config,
                use_amp,
                binary_threshold,
            )

            all_metrics[
                split
            ] = metrics

            prediction_rows.extend(
                rows
            )

            # Save each split immediately. In particular, test_metrics.csv and
            # test_sequence_predictions.csv exist as soon as TEST evaluation
            # finishes, rather than waiting for all downstream tables.
            write_csv(
                args.output_dir / f"{split}_metrics.csv",
                summary_rows(metrics),
            )
            write_csv(
                args.output_dir / f"{split}_sequence_predictions.csv",
                rows,
            )

            print(
                f"\n===== DNA-ONLY {split.upper()} RESULTS =====",
                flush=True,
            )

            full = metrics[
                "end_to_end_all_classes"
            ]

            binary = metrics[
                "binary"
            ]

            subtype = metrics[
                "ARG_subtype_oracle_binary"
            ]

            print(
                f"end-to-end | "
                f"ACC={full['accuracy']:.6f} | "
                f"weighted_P={full['weighted_precision']:.6f} | "
                f"macro_P={full['macro_precision']:.6f} | "
                f"weighted_F1={full['weighted_f1']:.6f} | "
                f"macro_F1={full['macro_f1']:.6f} | "
                f"MCC={full['mcc']:.6f}",
                flush=True,
            )

            print(
                f"binary     | "
                f"ACC={binary['accuracy']:.6f} | "
                f"weighted_P={binary['weighted_precision']:.6f} | "
                f"weighted_F1={binary['weighted_f1']:.6f} | "
                f"MCC={binary['mcc']:.6f}",
                flush=True,
            )

            print(
                f"ARG subtype oracle | "
                f"ACC={subtype['accuracy']:.6f} | "
                f"weighted_P={subtype['weighted_precision']:.6f} | "
                f"weighted_F1={subtype['weighted_f1']:.6f} | "
                f"macro_F1={subtype['macro_f1']:.6f} | "
                f"MCC={subtype['mcc']:.6f}",
                flush=True,
            )

        (
            subtype_level_rows,
            type_level_rows,
            binary_level_rows,
        ) = build_level_metric_tables(
            prediction_rows,
            data,
        )

        # Protein-style TEST-only class-level result files.
        write_csv(
            args.output_dir / "test_subtype_level_metrics.csv",
            [row for row in subtype_level_rows if row.get("split") == "test"],
        )
        write_csv(
            args.output_dir / "test_type_level_metrics.csv",
            [row for row in type_level_rows if row.get("split") == "test"],
        )
        write_csv(
            args.output_dir / "test_ARG_nonARG_level_metrics.csv",
            [row for row in binary_level_rows if row.get("split") == "test"],
        )

        overall_rows = []

        for split, split_metrics in all_metrics.items():

            for task, values in split_metrics.items():

                overall_rows.append(
                    {
                        "split":
                            split,

                        "sequence_level":
                            "full_length",

                        "task":
                            task,

                        **values,

                        "coverage":
                            1.0,
                    }
                )

        # ---------------------------------------------------------------------
        # Final checkpoint.
        # ---------------------------------------------------------------------

        checkpoint = {
            "model_name":
                (
                    "Full-length DNA-only hierarchical "
                    "fusion-teacher refined-prototype student"
                ),

            "script_version":
                SCRIPT_VERSION,

            "dataset_definition":
                "full-length ARG:nonARG balanced 1:1",

            "DNA_embedding_definition":
                (
                    "DNABERT-2-117M native tokenizer; "
                    "final hidden-state mean pooling over valid tokens; "
                    "768 dimensions; full-length DNA input"
                ),

            "student_state_dict":
                {
                    key:
                        value
                        .detach()
                        .cpu()
                    for key, value
                    in student
                    .state_dict()
                    .items()
                },

            "teacher_state_dict":
                {
                    key:
                        value
                        .detach()
                        .cpu()
                    for key, value
                    in teacher
                    .state_dict()
                    .items()
                },

            "dna_input_dimension":
                int(
                    data[
                        "dna_cpu"
                    ].shape[
                        1
                    ]
                ),

            "embedding_dimension":
                int(
                    config.embedding_dim
                ),

            "arg_subtype_names":
                data[
                    "arg_names"
                ],

            "full_class_names":
                [
                    NONARG
                ] + data[
                    "arg_names"
                ],

            "num_ARG_subtypes":
                len(
                    data[
                        "arg_names"
                    ]
                ),

            "num_full_classes":
                (
                    len(
                        data[
                            "arg_names"
                        ]
                    )
                    + 1
                ),

            "nonARG_class_index":
                0,

            "selected_binary_ARG_probability_threshold":
                float(
                    binary_threshold
                ),

            "refined_ARG_subtype_prototypes":
                data[
                    "refined"
                ]
                .detach()
                .cpu(),

            "dna_train_mean":
                dna_mean
                .detach()
                .cpu(),

            "dna_train_std":
                dna_std
                .detach()
                .cpu(),

            "teacher_best_epoch":
                int(
                    teacher_best_epoch
                ),

            "teacher_best_validation_score":
                float(
                    teacher_best_score
                ),

            "student_best_epoch":
                int(
                    student_best_epoch
                ),

            "student_best_validation_score":
                float(
                    student_best_score
                ),

            "metrics":
                all_metrics,

            "configuration":
                asdict(
                    config
                ),

            "GRID_best_source":
                "test_metrics(2).json",

            "epoch_representation_snapshots":
                snapshot_records,

            "subtype_to_type":
                data[
                    "subtype_to_type"
                ],

            "type_names":
                data[
                    "type_names"
                ],

            "inference_rule":
                (
                    "official DNABERT-2 mean embedding -> DNA student -> "
                    "binary nonARG/ARG head; if ARG, ARG-subtype cosine head"
                ),

            "training_rule":
                (
                    "model-train optimizes fusion teacher and DNA student; "
                    "validation selects teacher/student epochs; the binary threshold is fixed "
                    "to the DNA GRID-best value before TEST evaluation; "
                    "external test evaluated only after all selection; "
                    "deployment/inference uses DNA only"
                ),

            "source_DNA_pt":
                str(
                    args.dna_pt
                ),

            "source_fusion_pt":
                str(
                    args.fusion_pt
                ),

            "source_refined_pt":
                str(
                    args.refined_pt
                ),

            "source_structure":
                str(
                    args.structure
                ),

            "source_train_ids":
                str(
                    args.train_ids
                ),

            "source_test_ids":
                str(
                    args.test_ids
                ),

            "validation_fraction_from_external_train":
                float(
                    args.validation_fraction
                ),

            "split_policy":
                (
                    "train_ids_subtype2.txt and test_ids_subtype2.txt are the "
                    "authoritative external split; validation is carved only "
                    "from external train with the same deterministic subtype-aware "
                    "algorithm and seed as the Protein new-split training script; "
                    "old long_fusion split metadata is ignored"
                ),
        }

        checkpoint_path = (
            args.output_dir
            / "DNA_student_long_model.pt"
        )

        hb.update(
            phase="saving",
            note=str(
                checkpoint_path
            ),
        )

        save_checkpoint(
            checkpoint,
            checkpoint_path,
        )

        write_csv(
            args.output_dir
            / "training_history.csv",
            teacher_history
            + student_history,
        )

        write_csv(
            args.output_dir
            / "sequence_predictions.csv",
            prediction_rows,
        )

        write_csv(
            args.output_dir
            / "binary_ARG_threshold_validation_sweep.csv",
            threshold_rows,
        )

        write_csv(
            args.output_dir
            / "overall_metrics.csv",
            overall_rows,
        )

        write_csv(
            args.output_dir
            / "subtype_level_metrics.csv",
            subtype_level_rows,
        )

        write_csv(
            args.output_dir
            / "type_level_metrics.csv",
            type_level_rows,
        )

        write_csv(
            args.output_dir
            / "ARG_nonARG_level_metrics.csv",
            binary_level_rows,
        )

        metrics_path = (
            args.output_dir
            / "metrics.json"
        )

        temporary_json = metrics_path.with_suffix(
            ".json.tmp"
        )

        with temporary_json.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                json_safe(
                    all_metrics
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temporary_json,
            metrics_path,
        )

        test_metrics_path = (
            args.output_dir
            / "test_metrics.json"
        )

        with test_metrics_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                json_safe(
                    {
                        "test": all_metrics[
                            "test"
                        ],
                        "selected_binary_ARG_probability_threshold": (
                            binary_threshold
                        ),
                        "best_parameters": asdict(config),
                        "student_best_epoch": int(student_best_epoch),
                        "student_best_validation_score": float(
                            student_best_score
                        ),
                    }
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        best_epoch_summary_path = (
            args.output_dir
            / "current_run_best_epoch_summary.json"
        )

        with best_epoch_summary_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                json_safe(
                    {
                        "current_run_best_epoch": int(student_best_epoch),
                        "current_run_best_validation_score": float(
                            student_best_score
                        ),
                        "student_epochs_completed": int(
                            config.student_epochs
                        ),
                        "requested_snapshot_epochs": sorted(
                            set(int(x) for x in args.snapshot_epochs)
                        ),
                        "best_epoch_snapshot": (
                            f"run_best_epoch_{student_best_epoch:03d}_"
                            "test_ARG_type_features.pt"
                        ),
                    }
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "\n"
            + "=" * 125,
            flush=True,
        )

        print(
            "LONG DNA STUDENT TRAINING COMPLETED",
            flush=True,
        )

        print(
            "=" * 125,
            flush=True,
        )

        print(
            f"Teacher best epoch     : "
            f"{teacher_best_epoch}",
            flush=True,
        )

        print(
            f"Teacher best score     : "
            f"{teacher_best_score:.6f}",
            flush=True,
        )

        print(
            f"Student best epoch     : "
            f"{student_best_epoch}",
            flush=True,
        )

        print(
            f"Student best score     : "
            f"{student_best_score:.6f}",
            flush=True,
        )

        print(
            f"Binary ARG threshold   : "
            f"{binary_threshold:.2f}",
            flush=True,
        )

        print(
            f"ARG subtype count      : "
            f"{len(data['arg_names']):,}",
            flush=True,
        )

        print(
            f"Model checkpoint       : "
            f"{checkpoint_path}",
            flush=True,
        )

        print(
            f"TEST metrics           : "
            f"{args.output_dir / 'test_metrics.csv'}",
            flush=True,
        )
        print(
            f"TEST predictions       : "
            f"{args.output_dir / 'test_sequence_predictions.csv'}",
            flush=True,
        )
        print(
            f"Overall metrics        : "
            f"{args.output_dir / 'overall_metrics.csv'}",
            flush=True,
        )

        print(
            f"Subtype metrics        : "
            f"{args.output_dir / 'subtype_level_metrics.csv'}",
            flush=True,
        )

        print(
            f"Type metrics           : "
            f"{args.output_dir / 'type_level_metrics.csv'}",
            flush=True,
        )

        print(
            f"ARG/nonARG metrics     : "
            f"{args.output_dir / 'ARG_nonARG_level_metrics.csv'}",
            flush=True,
        )

        print(
            f"Epoch UMAP features    : "
            f"{args.output_dir / 'epoch_features'}",
            flush=True,
        )

        print(
            f"Snapshot manifest      : "
            f"{args.output_dir / 'snapshot_manifest.csv'}",
            flush=True,
        )

        print(
            f"Peak GPU memory        : "
            f"{torch.cuda.max_memory_allocated(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        print(
            f"Elapsed minutes        : "
            f"{(time.time()-started)/60.0:.3f}",
            flush=True,
        )

        write_run_status(
            args.output_dir,
            "completed",
            teacher_best_epoch=int(teacher_best_epoch),
            student_best_epoch=int(student_best_epoch),
            binary_threshold=float(binary_threshold),
            elapsed_minutes=float((time.time()-started)/60.0),
        )

    except Exception as exc:
        try:
            write_run_status(
                args.output_dir,
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_minutes=float((time.time()-started)/60.0),
            )
        finally:
            raise

    finally:

        hb.stop()


if __name__ == "__main__":
    main()
