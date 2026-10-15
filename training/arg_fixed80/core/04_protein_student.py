#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARGScope full-length (100%) Protein Student epoch-trajectory feature run.

Based on the Protein GRID-best training pipeline. A new FusionTeacher
is trained on the saved exact model-train split; the Protein Student is retrained on the
100% full-length ESM-2 dataset using the exact selected hyperparameters from
best_parameters_and_test_results.csv. TEST-ARG 512D Student representations
are saved at Epoch 0/20/50/100/150 and the reference GRID-best epoch.

ARGScope full-length Protein-only hierarchical Student.

Current inputs
--------------
1. ESM-2 full-length protein embeddings:
      ESM2_full_1to1/full_1to1_esm2_mean.pt
      expected dimension = 1280

2. Supervised long gated-fusion embeddings:
      Long_fusion_subtype2_newsplit/long_fusion_subtype2_newsplit.pt
      expected dimension = 512

3. Residual-GAT refined ARG subtype prototypes:
      Refined_subtype_prototypes_subtype2_newsplit/
      refined_ARG_subtype_prototypes_residual_GAT.pt
      expected dimension = 512

4. Structure table with authoritative Type/Subtype mapping.

Training
--------
A fixed fusion representation is used to train a lightweight FusionTeacher
(binary ARG/nonARG + ARG subtype cosine heads). The subtype head is initialized
from the refined residual-GAT prototypes.

The ProteinStudent receives ONLY ESM-2 embeddings and learns from:
  * binary supervised loss,
  * subtype supervised loss,
  * binary knowledge distillation from FusionTeacher,
  * subtype knowledge distillation from FusionTeacher,
  * feature alignment to the fusion teacher representation,
  * refined-prototype alignment,
  * classifier-head anchor.

Inference
---------
Only an ESM-2 1280D protein embedding is required.

Leakage policy
--------------
* train_ids_subtype2.txt and test_ids_subtype2.txt are authoritative.
* validated exact split/model_splits metadata inside Long_fusion_subtype2_newsplit/long_fusion_subtype2_newsplit.pt is consumed from the new fusion artifact.
* validation is carved only from the new external train set.
* all normalization/class weighting/optimization use model-train only.
* validation is used only for epoch selection and binary-threshold calibration.
* TEST is evaluated only after teacher/student/threshold selection is complete.
* no silent subtype intersection filtering is allowed.
* no CPU fallback.

Output policy
-------------
training_loss.csv:
    optimization losses only.

train_metrics.csv:
    final selected model performance on model-train, reporting only.

test_metrics.csv:
    final TEST results only:
      ARG_nonARG
      ARG_Subtype
      End_to_end

All overall metrics include:
    accuracy,
    macro/weighted/micro precision,
    macro/weighted/micro recall,
    macro/weighted/micro F1,
    MCC.
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


SCRIPT_VERSION = (
    "2026-09-03-protein-subtype2-authoritative-id-split-v2-refined-filter"
)

NONARG = "nonARG"

ROOT = Path(
    "/scratch/users/ntu/pan.jie/work/ARGscope_final_model/Dataset_embedding"
)

DEFAULT_PROTEIN_PT = (
    ROOT
    / "ESM2_full_1to1"
    / "full_1to1_esm2_mean.pt"
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

DEFAULT_GRID_DIR = (
    ROOT
    / "2.Protein_student_long_gridsearch"
)

DEFAULT_BEST_PARAMS = (
    DEFAULT_GRID_DIR
    / "best_parameters_and_test_results.csv"
)

DEFAULT_TEACHER_CHECKPOINT = (
    DEFAULT_GRID_DIR
    / "best_fusion_teacher.pt"
)

DEFAULT_TRAIN_IDS = ROOT / "train_ids_subtype2.txt"
DEFAULT_TEST_IDS = ROOT / "test_ids_subtype2.txt"

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "Protein_student_subtype2_newsplit_unified"
)


@dataclass
class Config:
    # These defaults match best_parameters_and_test_results.csv from the
    # selected Protein GRID-best pipeline. The CSV is still loaded at runtime
    # and is authoritative.
    seed: int = 20260822

    batch_size: int = 128
    eval_batch_size: int = 512

    teacher_epochs: int = 300
    student_epochs: int = 300

    teacher_patience: int = 15
    student_patience: int = 30

    teacher_lr: float = 2.0e-4
    student_lr: float = 1.0e-4
    weight_decay: float = 3.0e-5

    protein_hidden_1: int = 1536
    protein_hidden_2: int = 1024
    embedding_dim: int = 512

    dropout: float = 0.10

    logit_scale_binary: float = 16.0
    logit_scale_subtype: float = 22.0

    label_smoothing: float = 0.002

    # Student class-weight power from the final GRID-best configuration.
    class_weight_power: float = 0.0

    # Selected FusionTeacher used class-weight power 0.18. It is stored
    # separately because the final Student uses 0.0.
    teacher_class_weight_power: float = 0.18

    temperature: float = 3.0

    # Selected Fusion Teacher.
    teacher_binary_ce: float = 1.0
    teacher_subtype_ce: float = 1.0
    teacher_feature_anchor: float = 0.0
    teacher_refined_anchor: float = 0.10

    # Selected Protein Student.
    student_binary_ce: float = 1.0
    student_subtype_ce: float = 1.30
    student_binary_kd: float = 0.0
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

    p = argparse.ArgumentParser(
        description=(
            "Train full-length Protein-only ARG/nonARG + ARG-subtype Student "
            "using ESM-2, long gated fusion and refined subtype prototypes."
        )
    )

    p.add_argument(
        "--protein-pt",
        type=Path,
        default=DEFAULT_PROTEIN_PT,
    )

    p.add_argument(
        "--fusion-pt",
        type=Path,
        default=DEFAULT_FUSION_PT,
    )

    p.add_argument(
        "--refined-pt",
        type=Path,
        default=DEFAULT_REFINED_PT,
    )

    p.add_argument(
        "--structure",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--train-ids",
        type=Path,
        default=DEFAULT_TRAIN_IDS,
        help=(
            "Authoritative external TRAIN IDs. Default: "
            "Dataset_embedding/train_ids_subtype2.txt"
        ),
    )

    p.add_argument(
        "--test-ids",
        type=Path,
        default=DEFAULT_TEST_IDS,
        help=(
            "Authoritative external TEST IDs. Default: "
            "Dataset_embedding/test_ids_subtype2.txt"
        ),
    )

    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction carved from the NEW external train set for validation. "
            "For ARG subtypes, at least one sequence is always kept in model-train; "
            "subtypes represented by only one external-train sequence are never moved "
            "to validation. Default: 0.20."
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )


    p.add_argument(
        "--best-params",
        type=Path,
        default=DEFAULT_BEST_PARAMS,
        help="Protein GRID-best parameter CSV; first data row is authoritative.",
    )

    p.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=DEFAULT_TEACHER_CHECKPOINT,
        help=("Deprecated/ignored in this new-split run. The FusionTeacher is retrained "
              "from scratch on the new model-train subset to avoid old-split leakage."),
    )

    p.add_argument(
        "--snapshot-epochs",
        nargs="+",
        type=int,
        default=[0, 20, 50, 100, 150],
        help=(
            "Student epochs whose TEST-ARG 512D representations are saved. "
            "After training, the best epoch from THIS run is saved automatically."
        ),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=Config.batch_size,
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=Config.eval_batch_size,
    )

    p.add_argument(
        "--teacher-epochs",
        type=int,
        default=Config.teacher_epochs,
    )

    p.add_argument(
        "--student-epochs",
        type=int,
        default=Config.student_epochs,
    )

    p.add_argument(
        "--teacher-lr",
        type=float,
        default=Config.teacher_lr,
    )

    p.add_argument(
        "--student-lr",
        type=float,
        default=Config.student_lr,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=Config.weight_decay,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=Config.dropout,
    )

    p.add_argument(
        "--class-weight-power",
        type=float,
        default=Config.class_weight_power,
    )

    p.add_argument(
        "--student-subtype-ce",
        type=float,
        default=Config.student_subtype_ce,
    )

    p.add_argument(
        "--student-binary-kd",
        type=float,
        default=Config.student_binary_kd,
    )

    p.add_argument(
        "--student-subtype-kd",
        type=float,
        default=Config.student_subtype_kd,
    )

    p.add_argument(
        "--student-feature-align",
        type=float,
        default=Config.student_feature_align,
    )

    p.add_argument(
        "--student-prototype-align",
        type=float,
        default=Config.student_prototype_align,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=Config.seed,
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
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()



# =============================================================================
# GRID-best parameter / teacher loading
# =============================================================================

def _parse_csv_scalar(value):
    text = str(value).strip()
    if text == "":
        return None
    try:
        if any(ch in text.lower() for ch in (".", "e")):
            return float(text)
        return int(text)
    except Exception:
        return text


def load_grid_best_parameters(path: Path, config: Config):
    """Load the first row of the selected Protein GRID-best CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Empty GRID-best parameter CSV: {path}")

    # The file contains one row per reporting task; all parameter columns
    # must be identical. Use the first row and audit the rest.
    first = rows[0]
    parameter_columns = [
        key for key in first
        if key.startswith("param_") or key.startswith("selected_teacher_")
    ]

    for other in rows[1:]:
        for key in parameter_columns:
            if str(other.get(key, "")).strip() != str(first.get(key, "")).strip():
                raise ValueError(
                    f"GRID-best parameter file has inconsistent {key}: "
                    f"{first.get(key)!r} vs {other.get(key)!r}"
                )

    mapping = {
        "param_seed": "seed",
        "param_batch_size": "batch_size",
        "param_eval_batch_size": "eval_batch_size",
        "param_teacher_epochs": "teacher_epochs",
        "param_student_epochs": "student_epochs",
        "param_teacher_patience": "teacher_patience",
        "param_student_patience": "student_patience",
        "param_teacher_lr": "teacher_lr",
        "param_student_lr": "student_lr",
        "param_weight_decay": "weight_decay",
        "param_protein_hidden_1": "protein_hidden_1",
        "param_protein_hidden_2": "protein_hidden_2",
        "param_embedding_dim": "embedding_dim",
        "param_dropout": "dropout",
        "param_logit_scale_binary": "logit_scale_binary",
        "param_logit_scale_subtype": "logit_scale_subtype",
        "param_label_smoothing": "label_smoothing",
        "param_class_weight_power": "class_weight_power",
        "param_temperature": "temperature",
        "param_teacher_binary_ce": "teacher_binary_ce",
        "param_teacher_subtype_ce": "teacher_subtype_ce",
        "param_teacher_feature_anchor": "teacher_feature_anchor",
        "param_teacher_refined_anchor": "teacher_refined_anchor",
        "param_student_binary_ce": "student_binary_ce",
        "param_student_subtype_ce": "student_subtype_ce",
        "param_student_binary_kd": "student_binary_kd",
        "param_student_subtype_kd": "student_subtype_kd",
        "param_student_feature_align": "student_feature_align",
        "param_student_prototype_align": "student_prototype_align",
        "param_student_head_anchor": "student_head_anchor",
        "param_gradient_clip": "gradient_clip",
        "param_min_delta": "min_delta",
    }

    for source, target in mapping.items():
        if source in first and str(first[source]).strip() != "":
            current = getattr(config, target)
            value = _parse_csv_scalar(first[source])
            if isinstance(current, int) and not isinstance(current, bool):
                value = int(float(value))
            elif isinstance(current, float):
                value = float(value)
            setattr(config, target, value)

    # These are independently selected Teacher settings and are the actual
    # Teacher values used by the final grid pipeline.
    if str(first.get("selected_teacher_lr", "")).strip():
        config.teacher_lr = float(first["selected_teacher_lr"])
    if str(first.get("selected_teacher_refined_anchor", "")).strip():
        config.teacher_refined_anchor = float(first["selected_teacher_refined_anchor"])
    if str(first.get("selected_teacher_feature_anchor", "")).strip():
        config.teacher_feature_anchor = float(first["selected_teacher_feature_anchor"])
    if str(first.get("selected_teacher_class_weight_power", "")).strip():
        config.teacher_class_weight_power = float(
            first["selected_teacher_class_weight_power"]
        )

    metadata = {
        "reference_student_best_epoch": int(float(first.get("student_best_epoch", 0))),
        "reference_teacher_best_epoch": int(float(first.get("teacher_best_epoch", 0))),
        "reference_binary_threshold": float(
            first.get("selected_binary_ARG_probability_threshold", 0.8)
        ),
        "reference_student_best_validation_score": float(
            first.get("student_best_validation_score", "nan")
        ),
        "reference_teacher_best_validation_score": float(
            first.get("teacher_best_validation_score", "nan")
        ),
    }

    return config, metadata, first


def load_selected_fusion_teacher(path: Path, data, config: Config):
    checkpoint = load_pt(path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Teacher checkpoint must be a dict: {path}")

    state = checkpoint.get("state_dict")
    if state is None:
        state = checkpoint.get("teacher_state_dict")
    if state is None:
        raise KeyError(
            f"Teacher checkpoint lacks state_dict/teacher_state_dict; "
            f"keys={sorted(checkpoint.keys())}"
        )

    teacher = FusionTeacher(
        data["refined_gpu"],
        config,
    ).cuda()
    teacher.load_state_dict(state, strict=True)
    teacher.eval()

    stored_config = checkpoint.get("configuration", {})
    if stored_config:
        print("Selected Teacher checkpoint configuration:", flush=True)
        for key in (
            "teacher_lr",
            "teacher_refined_anchor",
            "teacher_feature_anchor",
            "class_weight_power",
        ):
            if key in stored_config:
                print(f"  {key} = {stored_config[key]}", flush=True)

    return teacher, checkpoint

# =============================================================================
# Heartbeat
# =============================================================================

class Heartbeat:

    def __init__(self, interval=30):
        self.interval = max(
            5,
            int(interval),
        )
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.phase = "startup"
        self.epoch = None
        self.note = ""
        self.thread = None

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

        epoch_text = (
            f" | epoch={epoch}"
            if epoch is not None
            else ""
        )

        note_text = (
            f" | {note}"
            if note
            else ""
        )

        print(
            f"[HEARTBEAT] "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"phase={phase}"
            f"{epoch_text} | "
            f"{gpu_text}"
            f"{note_text}",
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
                timeout=5
            )
        self.emit()


# =============================================================================
# General utilities
# =============================================================================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass


def canonical_id(value: Any):

    text = str(
        value
    ).strip()

    return (
        text.split()[0]
        if text
        else ""
    )


def canonical_subtype(value: Any):

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


def is_nonarg(value: Any):

    return (
        canonical_subtype(
            value
        )
        == NONARG
    )


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
            f"{name} must be a 2D torch.Tensor."
        )

    value = (
        value.detach()
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
            f"{name} contains NaN/Inf."
        )

    return value


def extract_embedding_pt(
    path,
    description,
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
            f"{description} PT requires IDs + embeddings; "
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
            f"{description} ID/embedding mismatch: "
            f"{len(ids)} vs {embeddings.shape[0]}"
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


def detect_delimiter(path):

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


def load_subtype_to_type(path):

    delimiter = detect_delimiter(
        path
    )

    mapping = {}

    with path.open(
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
            or "Subtype" not in reader.fieldnames
            or "Type" not in reader.fieldnames
        ):
            raise ValueError(
                f"Structure requires Subtype and Type columns. "
                f"Found={reader.fieldnames}"
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
                    f"Empty Type at line {line_number}: "
                    f"Subtype={subtype!r}"
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
                    f"{subtype}: {previous} vs {type_name}"
                )

            mapping[
                subtype
            ] = type_name

    mapping[
        NONARG
    ] = NONARG

    return mapping


def write_csv(
    path,
    rows,
):

    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(
                    key
                )
                seen.add(
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


def json_safe(value):

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k):
                json_safe(v)
            for k, v
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(v)
            for v
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


def safe_save(
    obj,
    path,
):

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

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
            f"Invalid temporary checkpoint: {temporary}"
        )

    os.replace(
        temporary,
        path,
    )


# =============================================================================
# Data loading/alignment
# =============================================================================

def read_id_file(path: Path):
    ids = []
    seen = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            seq_id = canonical_id(raw)
            if not seq_id:
                continue
            if seq_id in seen:
                raise ValueError(
                    f"Duplicate ID in {path} at line {line_number}: {seq_id}"
                )
            seen.add(seq_id)
            ids.append(seq_id)

    if not ids:
        raise ValueError(f"ID file is empty: {path}")

    return ids


def build_model_splits_from_new_ids(
    ids,
    subtypes,
    train_ids_set,
    test_ids_set,
    validation_fraction,
    seed,
):
    """
    Build model-train/validation/test from the NEW authoritative ID files.

    External TEST is never touched. Validation is carved only from external TRAIN.
    For every ARG subtype, at least one external-train sequence is guaranteed to
    remain in model-train. Therefore, an ARG subtype with only one sequence on the
    train side stays entirely in model-train and contributes zero validation rows.
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

    # ARG validation: stratify within each subtype while guaranteeing >=1 train row.
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

    # nonARG validation is matched to the ARG validation count.
    # Because the supplied external train is ARG:nonARG = 1:1, this keeps
    # BOTH model-train and validation approximately/exactly 1:1 at binary level.
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

    n_nonarg_val = int(arg_validation_count)
    nonarg_val_positions = set(nonarg_positions[:n_nonarg_val])
    for index in nonarg_positions:
        model_split[index] = (
            "validation" if index in nonarg_val_positions else "train"
        )

    if any(value is None for value in model_split):
        missing = [ids[i] for i, value in enumerate(model_split) if value is None]
        raise RuntimeError(
            f"Failed to assign model split to {len(missing)} IDs; examples={missing[:10]}"
        )

    # Strong checks: external test is exactly model test, and no external test leaks.
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
    protein_path,
    fusion_path,
    refined_path,
    structure_path,
    train_ids_path,
    test_ids_path,
    validation_fraction,
    split_seed,
):

    protein_ids, protein, protein_obj = (
        extract_embedding_pt(
            protein_path,
            "ESM-2 protein",
        )
    )

    fusion_ids, fusion, fusion_obj = (
        extract_embedding_pt(
            fusion_path,
            "long fusion",
        )
    )

    if protein.shape[
        1
    ] != 1280:
        raise ValueError(
            f"Expected ESM-2 650M mean embedding dimension 1280; "
            f"found {protein.shape[1]}."
        )

    if fusion.shape[
        1
    ] != 512:
        raise ValueError(
            f"Expected long fusion dimension 512; "
            f"found {fusion.shape[1]}."
        )

    if not set(fusion_ids).issubset(set(protein_ids)):
        raise ValueError(
            "Protein and fusion ID sets differ. "
            f"Protein={len(protein_ids):,}; "
            f"Fusion={len(fusion_ids):,}; "
            f"protein_only={len(set(protein_ids)-set(fusion_ids)):,}; "
            f"fusion_only={len(set(fusion_ids)-set(protein_ids)):,}"
        )

    protein_index = {
        seq_id:
            index
        for index, seq_id
        in enumerate(
            protein_ids
        )
    }

    alignment = torch.tensor(
        [
            protein_index[
                seq_id
            ]
            for seq_id
            in fusion_ids
        ],
        dtype=torch.long,
    )

    protein = (
        protein.index_select(
            0,
            alignment,
        )
        .contiguous()
    )

    # ---------------------------------------------------------------------
    # NEW authoritative external train/test split.
    # Ignore any validated exact split/model_splits metadata stored inside long_fusion_subtype2_newsplit.pt.
    # ---------------------------------------------------------------------
    if "subtypes" not in fusion_obj:
        raise KeyError(
            "long_fusion_subtype2_newsplit.pt missing required key: subtypes"
        )

    all_subtypes = [
        canonical_subtype(value)
        for value in fusion_obj["subtypes"]
    ]

    if len(all_subtypes) != len(fusion_ids):
        raise ValueError(
            "Fusion IDs/subtypes length mismatch: "
            f"{len(fusion_ids)} vs {len(all_subtypes)}"
        )

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
            f"{len(missing_requested)} IDs from the new train/test files are absent "
            f"from protein/fusion embeddings; examples={sorted(missing_requested)[:20]}"
        )

    # Filter away all excluded singleton ARG / excess nonARG rows.
    selected_positions = [
        i for i, seq_id in enumerate(fusion_ids) if seq_id in requested_ids
    ]
    select_index = torch.tensor(selected_positions, dtype=torch.long)

    fusion_ids = [fusion_ids[i] for i in selected_positions]
    subtypes = [all_subtypes[i] for i in selected_positions]
    protein = protein.index_select(0, select_index).contiguous()
    fusion = fusion.index_select(0, select_index).contiguous()

    if set(fusion_ids) != requested_ids:
        raise RuntimeError(
            "Internal filtering error: selected embedding IDs do not equal "
            "new train U new test IDs."
        )

    model_splits, external_splits = contract.use_saved_splits(
        obj=fusion_obj,
        ids=fusion_ids,
        subtypes=subtypes,
        train_ids_set=train_ids_set,
        test_ids_set=test_ids_set,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    n = len(fusion_ids)

    print(
        "NEW authoritative split loaded from ID files:",
        flush=True,
    )
    print(
        f"  external train IDs : {len(train_ids_set):,} | {train_ids_path}",
        flush=True,
    )
    print(
        f"  external test IDs  : {len(test_ids_set):,} | {test_ids_path}",
        flush=True,
    )
    print(
        f"  selected total     : {n:,}",
        flush=True,
    )
    print(
        f"  validation fraction: {float(validation_fraction):.3f} "
        "(carved ONLY from external train)",
        flush=True,
    )

    refined_obj = load_pt(
        refined_path
    )
    contract.validate_derived(refined_obj, fusion_obj, fusion_path, 'refined')


    if not isinstance(
        refined_obj,
        dict,
    ):
        raise TypeError(
            "Refined PT must be a dictionary."
        )

    if (
        "arg_subtype_names" not in refined_obj
        or
        "refined_arg_subtype_prototypes" not in refined_obj
    ):
        raise KeyError(
            "Refined PT requires arg_subtype_names and "
            "refined_arg_subtype_prototypes."
        )

    # The refined prototype PT was generated from the unified new-split pipeline and
    # therefore still contains prototypes for all original ARG subtypes
    # (2,824 in the current project).  The new authoritative train/test ID
    # files contain only the retained subtype>=2 dataset (820 ARG subtypes).
    #
    # Therefore, do NOT require exact equality between the refined PT subtype
    # set and the current selected dataset.  Instead:
    #   1) every retained/observed ARG subtype MUST have a refined prototype;
    #   2) extra prototypes belonging to removed singleton subtypes are ignored;
    #   3) prototype order is preserved from the refined PT.
    all_refined_arg_names = [
        canonical_subtype(
            value
        )
        for value
        in refined_obj[
            "arg_subtype_names"
        ]
    ]

    all_refined = require_tensor_2d(
        refined_obj[
            "refined_arg_subtype_prototypes"
        ],
        "refined prototypes",
    )

    if (
        len(
            all_refined_arg_names
        )
        != all_refined.shape[
            0
        ]
        or all_refined.shape[
            1
        ] != 512
    ):
        raise ValueError(
            f"Refined prototype mismatch: "
            f"names={len(all_refined_arg_names)}, "
            f"shape={tuple(all_refined.shape)}"
        )

    if len(
        all_refined_arg_names
    ) != len(
        set(
            all_refined_arg_names
        )
    ):
        raise ValueError(
            "Duplicate subtype names in refined PT."
        )

    if any(
        is_nonarg(
            value
        )
        for value
        in all_refined_arg_names
    ):
        raise ValueError(
            "nonARG must not be included in ARG refined prototypes."
        )

    observed_arg = {
        value
        for value
        in subtypes
        if not is_nonarg(
            value
        )
    }

    all_refined_arg_set = set(
        all_refined_arg_names
    )

    missing_refined = sorted(
        observed_arg
        - all_refined_arg_set
    )

    if missing_refined:
        raise ValueError(
            f"{len(missing_refined):,} retained ARG subtype(s) lack "
            f"refined prototypes; examples={missing_refined[:20]}"
        )

    refined_name_to_index = {
        name:
            index
        for index, name
        in enumerate(
            all_refined_arg_names
        )
    }

    # Preserve the original refined-prototype order, but keep only the ARG
    # subtypes present in the NEW authoritative split.
    arg_names = [
        name
        for name
        in all_refined_arg_names
        if name in observed_arg
    ]

    refined_indices = torch.tensor(
        [
            refined_name_to_index[
                name
            ]
            for name
            in arg_names
        ],
        dtype=torch.long,
    )

    refined = (
        all_refined
        .index_select(
            0,
            refined_indices,
        )
        .contiguous()
    )

    refined = F.normalize(
        refined,
        p=2,
        dim=1,
    )

    refined_arg = set(
        arg_names
    )

    if refined_arg != observed_arg:
        raise RuntimeError(
            "Internal refined-prototype filtering error: "
            f"observed={len(observed_arg):,}, "
            f"selected_refined={len(refined_arg):,}"
        )

    print(
        "Refined prototype filtering:",
        flush=True,
    )
    print(
        f"  source refined subtypes : {len(all_refined_arg_names):,}",
        flush=True,
    )
    print(
        f"  retained split subtypes : {len(observed_arg):,}",
        flush=True,
    )
    print(
        f"  selected prototypes     : {len(arg_names):,}",
        flush=True,
    )
    print(
        f"  ignored old prototypes : "
        f"{len(all_refined_arg_names) - len(arg_names):,}",
        flush=True,
    )

    if NONARG not in set(
        subtypes
    ):
        raise ValueError(
            "No nonARG records found."
        )

    subtype_to_type = load_subtype_to_type(
        structure_path
    )
    if any(subtype_to_type.get(s) != t for s, t in fusion_obj['subtype_to_type'].items()):
        raise ValueError('Student structure Type mapping differs from fusion')


    missing_type = sorted(
        set(
            subtypes
        )
        - set(
            subtype_to_type
        )
    )

    if missing_type:
        raise ValueError(
            f"{len(missing_type)} subtype(s) lack Type mapping; "
            f"examples={missing_type[:10]}"
        )

    type_names = sorted(
        {
            subtype_to_type[
                name
            ]
            for name
            in arg_names
        },
        key=lambda value:
            value.lower(),
    )

    arg_to_index = {
        value:
            index
        for index, value
        in enumerate(
            arg_names
        )
    }

    binary_targets = torch.tensor(
        [
            0
            if is_nonarg(
                value
            )
            else 1
            for value
            in subtypes
        ],
        dtype=torch.long,
    )

    arg_targets = torch.tensor(
        [
            -1
            if is_nonarg(
                value
            )
            else arg_to_index[
                value
            ]
            for value
            in subtypes
        ],
        dtype=torch.long,
    )

    split_indices = {
        split:
            torch.tensor(
                [
                    index
                    for index, value
                    in enumerate(
                        model_splits
                    )
                    if value == split
                ],
                dtype=torch.long,
            )
        for split
        in (
            "train",
            "validation",
            "test",
        )
    }

    if any(
        values.numel() == 0
        for values
        in split_indices.values()
    ):
        raise ValueError(
            "train/validation/test must all be non-empty: "
            f"{ {k:int(v.numel()) for k,v in split_indices.items()} }"
        )

    # No silent class dropping.
    train_arg_names = {
        subtypes[
            index
        ]
        for index
        in split_indices[
            "train"
        ].tolist()
        if not is_nonarg(
            subtypes[
                index
            ]
        )
    }

    missing_in_train = sorted(
        refined_arg
        - train_arg_names
    )

    if missing_in_train:
        raise ValueError(
            f"{len(missing_in_train)} ARG subtype(s) absent from model-train; "
            f"examples={missing_in_train[:20]}"
        )

    print(
        f"ESM-2 shape             : {tuple(protein.shape)}",
        flush=True,
    )
    print(
        f"Fusion shape            : {tuple(fusion.shape)}",
        flush=True,
    )
    print(
        f"Refined prototype shape : {tuple(refined.shape)}",
        flush=True,
    )
    print(
        f"ARG subtype count       : {len(arg_names):,}",
        flush=True,
    )
    print(
        f"ARG Type count          : {len(type_names):,}",
        flush=True,
    )
    print(
        f"ARG total               : {int((binary_targets==1).sum()):,}",
        flush=True,
    )
    print(
        f"nonARG total            : {int((binary_targets==0).sum()):,}",
        flush=True,
    )

    for split, indices in split_indices.items():

        local_binary = binary_targets.index_select(
            0,
            indices,
        )

        subtype_count = len(
            {
                subtypes[
                    index
                ]
                for index
                in indices.tolist()
                if not is_nonarg(
                    subtypes[
                        index
                    ]
                )
            }
        )

        print(
            f"{split:10s}: "
            f"n={indices.numel():,}, "
            f"ARG={int((local_binary==1).sum()):,}, "
            f"nonARG={int((local_binary==0).sum()):,}, "
            f"ARG_subtypes={subtype_count:,}",
            flush=True,
        )

    return {
        "ids":
            fusion_ids,
        "subtypes":
            subtypes,
        "model_splits":
            model_splits,
        "external_splits":
            external_splits,
        "protein":
            protein,
        "fusion":
            fusion,
        "refined":
            F.normalize(
                refined,
                p=2,
                dim=1,
            ),
        "arg_names":
            arg_names,
        "type_names":
            type_names,
        "subtype_to_type":
            subtype_to_type,
        "binary_targets":
            binary_targets,
        "arg_targets":
            arg_targets,
        "split_indices":
            split_indices,
        "source_protein":
            protein_obj,
        "source_fusion":
            fusion_obj,
        "source_refined":
            refined_obj,
    }


# =============================================================================
# Model
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
            f"Missing model-train class(es): {missing[:20]}"
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

        self.binary_head = CosineHead(
            2,
            config.embedding_dim,
            config.logit_scale_binary,
        )

        self.arg_head = CosineHead(
            refined.shape[
                0
            ],
            config.embedding_dim,
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


class ProteinStudent(
    nn.Module
):

    def __init__(
        self,
        protein_dimension,
        protein_mean,
        protein_std,
        number_arg_subtypes,
        config,
    ):

        super().__init__()

        self.register_buffer(
            "protein_mean",
            protein_mean.clone(),
        )

        self.register_buffer(
            "protein_std",
            protein_std.clone(),
        )

        # Keeps the useful high-capacity shape of the old LongAA mapper,
        # but ends directly in the current 512D fusion/prototype space.
        self.input = nn.Sequential(
            nn.Linear(
                protein_dimension,
                config.protein_hidden_1,
            ),
            nn.LayerNorm(
                config.protein_hidden_1
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),

            nn.Linear(
                config.protein_hidden_1,
                config.protein_hidden_2,
            ),
            nn.LayerNorm(
                config.protein_hidden_2
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
        )

        self.blocks = nn.Sequential(
            ResidualBlock(
                config.protein_hidden_2,
                config.dropout,
            ),
            ResidualBlock(
                config.protein_hidden_2,
                config.dropout,
            ),
        )

        self.output = nn.Sequential(
            nn.Linear(
                config.protein_hidden_2,
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
        protein,
    ):

        protein = (
            protein
            - self.protein_mean
        ) / self.protein_std

        hidden = self.input(
            protein
        )

        hidden = self.blocks(
            hidden
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
        protein,
    ):

        z = self.encode(
            protein
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
# Training helpers
# =============================================================================

def amp_context(enabled):

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=enabled,
    )


def make_grad_scaler(enabled):

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except Exception:
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


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

        centers.append(
            F.normalize(
                selected.mean(
                    dim=0
                ),
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


# =============================================================================
# Metrics
# =============================================================================

def safe_mcc(
    true,
    predicted,
):

    if (
        len(
            true
        ) == 0
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


@torch.inference_mode()
def collect_outputs(
    model,
    inputs,
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

        xb = inputs.index_select(
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
            binary.float().cpu()
        )
        arg_logits.append(
            subtype.float().cpu()
        )
        features.append(
            z.float().cpu()
        )
        output_indices.append(
            local_indices.cpu()
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
        binary_targets_cpu.index_select(
            0,
            indices,
        ).numpy()
    )

    true_arg = (
        arg_targets_cpu.index_select(
            0,
            indices,
        ).numpy()
    )

    binary_prob = (
        torch.softmax(
            outputs[
                "binary_logits"
            ],
            dim=1,
        ).numpy()
    )

    arg_prob = (
        torch.softmax(
            outputs[
                "arg_logits"
            ],
            dim=1,
        ).numpy()
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

    pred_full = np.where(
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
                pred_full,
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
        "pred_full":
            pred_full,
        "ARG_probability":
            binary_prob[
                :,
                1
            ],
        "ARG_subtype_max_probability":
            arg_prob.max(
                axis=1
            ),
    }

    return (
        metrics,
        arrays,
    )


def validation_score(metrics):

    full = metrics[
        "end_to_end_all_classes"
    ]

    subtype = metrics[
        "ARG_subtype_oracle_binary"
    ]

    binary = metrics[
        "binary"
    ]

    # Precision-priority selection, consistent with the current final pipeline.
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



def metrics_from_prediction_arrays(
    true_binary,
    true_arg,
    predicted_binary,
    predicted_arg,
):
    true_full = np.where(
        true_binary == 0,
        0,
        true_arg + 1,
    )
    predicted_full = np.where(
        predicted_binary == 0,
        0,
        predicted_arg + 1,
    )
    arg_mask = true_binary == 1

    return {
        "binary": metric_dict(true_binary, predicted_binary),
        "ARG_subtype_oracle_binary": metric_dict(
            true_arg[arg_mask],
            predicted_arg[arg_mask],
        ),
        "end_to_end_all_classes": metric_dict(
            true_full,
            predicted_full,
        ),
    }


def calibrate_threshold_from_outputs(
    outputs,
    data,
    minimum_arg_recall=0.70,
):
    """Exact threshold-calibration logic used by the Protein grid search."""
    indices = outputs["indices"]
    true_binary = (
        data["binary_targets_cpu"]
        .index_select(0, indices)
        .numpy()
    )
    true_arg = (
        data["arg_targets_cpu"]
        .index_select(0, indices)
        .numpy()
    )

    binary_probability = (
        torch.softmax(outputs["binary_logits"], dim=1)[:, 1]
        .numpy()
    )
    predicted_arg = outputs["arg_logits"].argmax(dim=1).numpy()

    rows = []
    for threshold in np.round(np.arange(0.30, 0.901, 0.01), 2):
        predicted_binary = (
            binary_probability >= threshold
        ).astype(np.int64)

        metrics = metrics_from_prediction_arrays(
            true_binary,
            true_arg,
            predicted_binary,
            predicted_arg,
        )

        arg_precision = float(
            precision_score(
                true_binary,
                predicted_binary,
                pos_label=1,
                zero_division=0,
            )
        )
        arg_recall = float(
            recall_score(
                true_binary,
                predicted_binary,
                pos_label=1,
                zero_division=0,
            )
        )
        binary_accuracy = float(
            accuracy_score(true_binary, predicted_binary)
        )
        binary_mcc = safe_mcc(true_binary, predicted_binary)
        threshold_score = (
            0.70 * arg_precision
            + 0.20 * binary_accuracy
            + 0.10 * binary_mcc
        )

        rows.append(
            {
                "threshold": float(threshold),
                "ARG_precision": arg_precision,
                "ARG_recall": arg_recall,
                "binary_accuracy": binary_accuracy,
                "binary_mcc": binary_mcc,
                "threshold_score": threshold_score,
                "recall_floor_satisfied": bool(
                    arg_recall >= minimum_arg_recall
                ),
                "metrics": metrics,
            }
        )

    feasible = [
        row for row in rows
        if row["recall_floor_satisfied"]
    ]
    used_recall_fallback = False
    if not feasible:
        feasible = rows
        used_recall_fallback = True

    best = max(
        feasible,
        key=lambda row: (
            row["threshold_score"],
            row["ARG_precision"],
            row["binary_accuracy"],
            row["binary_mcc"],
            row["ARG_recall"],
            -abs(row["threshold"] - 0.50),
        ),
    )

    return {
        "threshold": float(best["threshold"]),
        "ARG_precision": float(best["ARG_precision"]),
        "ARG_recall": float(best["ARG_recall"]),
        "used_recall_fallback": bool(used_recall_fallback),
        "metrics": best["metrics"],
    }


def evaluate_for_selection(
    model,
    inputs,
    validation_indices,
    data,
    config,
    use_amp,
):

    outputs = collect_outputs(
        model,
        inputs,
        validation_indices,
        config.eval_batch_size,
        use_amp,
    )

    metrics, _ = predictions_and_metrics(
        outputs,
        data[
            "binary_targets_cpu"
        ],
        data[
            "arg_targets_cpu"
        ],
        binary_threshold=0.50,
    )

    return (
        validation_score(
            metrics
        ),
        metrics,
    )


# =============================================================================
# Train Fusion Teacher
# =============================================================================

def train_teacher(
    data,
    config,
    use_amp,
    hb,
):

    train_indices = data[
        "split_indices_gpu"
    ][
        "train"
    ]

    val_indices = data[
        "split_indices_gpu"
    ][
        "validation"
    ]

    train_arg = (
        data[
            "arg_targets_gpu"
        ].index_select(
            0,
            train_indices,
        )
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
        config.teacher_class_weight_power,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.teacher_lr,
        weight_decay=config.weight_decay,
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

    refined = data[
        "refined_gpu"
    ]

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

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
        )

        started = time.time()

        model.train()

        totals = defaultdict(
            float
        )

        seen = 0

        for indices in batch_indices(
            train_indices,
            config.batch_size,
            shuffle=True,
        ):

            fusion = data[
                "fusion_gpu"
            ].index_select(
                0,
                indices,
            )

            binary_y = data[
                "binary_targets_gpu"
            ].index_select(
                0,
                indices,
            )

            arg_y = data[
                "arg_targets_gpu"
            ].index_select(
                0,
                indices,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with amp_context(
                use_amp
            ):

                binary_logits, arg_logits, z = model(
                    fusion
                )

                loss_binary = F.cross_entropy(
                    binary_logits.float(),
                    binary_y,
                    label_smoothing=config.label_smoothing,
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
                    label_smoothing=config.label_smoothing,
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

            n = int(
                binary_y.shape[
                    0
                ]
            )

            seen += n

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
                    * n
                )

        score, validation = evaluate_for_selection(
            model,
            data[
                "fusion_gpu"
            ],
            val_indices,
            data,
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
            "lr":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],
            "epoch_seconds":
                time.time()
                - started,
        }

        history.append(
            row
        )

        print(
            f"Teacher {epoch:03d}/{config.teacher_epochs} | "
            f"loss={row['loss']:.5f} | "
            f"val_score={score:.5f} | "
            f"val_ACC="
            f"{validation['end_to_end_all_classes']['accuracy']:.5f} | "
            f"val_weighted_P="
            f"{validation['end_to_end_all_classes']['weighted_precision']:.5f} | "
            f"val_MCC="
            f"{validation['end_to_end_all_classes']['mcc']:.5f} | "
            f"lr={row['lr']:.2e}",
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
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict().items()
            }

            no_improve = 0

        else:

            no_improve += 1

        if (
            no_improve
            >= config.teacher_patience
        ):
            print(
                f"Teacher early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Teacher did not produce a checkpoint."
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
# TEST-ARG representation snapshots
# =============================================================================

@torch.inference_mode()
def save_test_arg_representation_snapshot(
    model,
    data,
    config,
    use_amp,
    snapshot_dir: Path,
    epoch: int,
    tag: str,
):
    """
    Save the 512D ProteinStudent representation for the SAME external TEST ARG
    sequences at every requested epoch. nonARG is intentionally excluded
    because the target figure is colored by ARG Type.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    test_indices_cpu = data["split_indices_cpu"]["test"]
    test_binary = data["binary_targets_cpu"].index_select(0, test_indices_cpu)
    arg_indices_cpu = test_indices_cpu[test_binary == 1]
    arg_indices_gpu = arg_indices_cpu.cuda(non_blocking=True)

    outputs = collect_outputs(
        model,
        data["protein_gpu"],
        arg_indices_gpu,
        config.eval_batch_size,
        use_amp,
    )

    global_indices = outputs["indices"].tolist()
    sequence_ids = [data["ids"][index] for index in global_indices]
    true_subtypes = [data["subtypes"][index] for index in global_indices]
    true_types = [data["subtype_to_type"][name] for name in true_subtypes]

    features = outputs["features"].float().contiguous()

    if features.shape[1] != config.embedding_dim:
        raise ValueError(
            f"Snapshot dimension mismatch: {tuple(features.shape)}; "
            f"expected (*, {config.embedding_dim})"
        )

    path = snapshot_dir / f"{tag}_test_ARG_type_features.pt"

    safe_save(
        {
            "epoch": int(epoch),
            "tag": str(tag),
            "split": "test",
            "ARG_only": True,
            "sequence_ids": sequence_ids,
            "true_subtypes": true_subtypes,
            "true_types": true_types,
            "features": features,
            "binary_logits": outputs["binary_logits"].float().contiguous(),
            "arg_subtype_logits": outputs["arg_logits"].float().contiguous(),
            "arg_subtype_names": list(data["arg_names"]),
            "subtype_to_type": dict(data["subtype_to_type"]),
            "feature_dimension": int(features.shape[1]),
            "num_sequences": int(features.shape[0]),
            "representation": "ProteinStudent normalized 512D z",
            "note": (
                "Same TEST-ARG sequence set is used for every saved epoch; "
                "no UMAP/t-SNE/PCA has been applied."
            ),
            "script_version": SCRIPT_VERSION,
        },
        path,
    )

    print(
        f"[SNAPSHOT] {tag} | epoch={epoch} | "
        f"TEST ARG={features.shape[0]:,} | dim={features.shape[1]} | {path}",
        flush=True,
    )

    return {
        "epoch": int(epoch),
        "tag": str(tag),
        "path": str(path),
        "n": int(features.shape[0]),
        "dimension": int(features.shape[1]),
    }


# =============================================================================
# Train Protein Student
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

    val_indices = data[
        "split_indices_gpu"
    ][
        "validation"
    ]

    protein_train = (
        data[
            "protein_cpu"
        ].index_select(
            0,
            train_indices_cpu,
        )
    )

    protein_mean = protein_train.mean(
        dim=0
    )

    protein_std = (
        protein_train.std(
            dim=0,
            unbiased=False,
        ).clamp_min(
            1.0e-6
        )
    )

    del protein_train

    train_arg = (
        data[
            "arg_targets_cpu"
        ].index_select(
            0,
            train_indices_cpu,
        )
    )

    subtype_weights = mild_class_weights(
        train_arg[
            train_arg
            >= 0
        ],
        len(
            data[
                "arg_names"
            ]
        ),
        config.class_weight_power,
    ).cuda()

    model = ProteinStudent(
        protein_dimension=1280,
        protein_mean=protein_mean.cuda(),
        protein_std=protein_std.cuda(),
        number_arg_subtypes=len(
            data[
                "arg_names"
            ]
        ),
        config=config,
    ).cuda()

    with torch.no_grad():
        model.binary_head.weight.copy_(
            teacher.binary_head.weight
        )
        model.arg_head.weight.copy_(
            teacher.arg_head.weight
        )

    teacher_binary_anchor = (
        teacher.binary_head.normalized_weight()
        .detach()
    )

    teacher_arg_anchor = (
        teacher.arg_head.normalized_weight()
        .detach()
    )

    refined = data[
        "refined_gpu"
    ]

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

    train_binary = (
        data[
            "binary_targets_gpu"
        ].index_select(
            0,
            train_indices,
        )
    )

    nonarg_fusion = train_fusion[
        train_binary
        == 0
    ]

    if nonarg_fusion.shape[
        0
    ] == 0:
        raise ValueError(
            "No train nonARG sequences."
        )

    nonarg_center = F.normalize(
        nonarg_fusion.mean(
            dim=0
        ),
        p=2,
        dim=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.student_lr,
        weight_decay=config.weight_decay,
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

    teacher.eval()

    best_score = -math.inf
    best_precision = -math.inf
    best_accuracy = -math.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    snapshot_records = []
    early_stop_notified = False

    requested_snapshot_epochs = sorted(
        set(int(value) for value in snapshot_epochs)
    )

    # Epoch 0: after GRID-best Teacher heads initialize the Student heads,
    # before any Student optimization step.
    if 0 in requested_snapshot_epochs:
        snapshot_records.append(
            save_test_arg_representation_snapshot(
                model,
                data,
                config,
                use_amp,
                snapshot_dir,
                epoch=0,
                tag="epoch_000",
            )
        )

    print(
        "\n===== TRAIN PROTEIN-ONLY STUDENT (FULL TRAJECTORY) =====",
        flush=True,
    )
    print(
        f"Representation snapshots: {requested_snapshot_epochs}",
        flush=True,
    )
    print(
        "Student early-stop criterion is monitored but NOT used to stop this "
        "trajectory run. Training continues through the configured epochs so "
        "all snapshots are available and the best epoch is determined from "
        "THIS run.",
        flush=True,
    )

    for epoch in range(
        1,
        config.student_epochs
        + 1,
    ):

        hb.update(
            phase="protein_student",
            epoch=epoch,
        )

        started = time.time()

        model.train()

        totals = defaultdict(
            float
        )

        seen = 0

        for indices in batch_indices(
            train_indices,
            config.batch_size,
            shuffle=True,
        ):

            protein = data[
                "protein_gpu"
            ].index_select(
                0,
                indices,
            )

            fusion = data[
                "fusion_gpu"
            ].index_select(
                0,
                indices,
            )

            binary_y = data[
                "binary_targets_gpu"
            ].index_select(
                0,
                indices,
            )

            arg_y = data[
                "arg_targets_gpu"
            ].index_select(
                0,
                indices,
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
                    protein
                )

                loss_binary = F.cross_entropy(
                    student_binary.float(),
                    binary_y,
                    label_smoothing=config.label_smoothing,
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
                    label_smoothing=config.label_smoothing,
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

            n = int(
                binary_y.shape[
                    0
                ]
            )

            seen += n

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
                    * n
                )

        # Exact validation/calibration logic from the Protein grid search.
        validation_outputs = collect_outputs(
            model,
            data["protein_gpu"],
            val_indices,
            config.eval_batch_size,
            use_amp,
        )

        calibration = calibrate_threshold_from_outputs(
            validation_outputs,
            data,
            minimum_arg_recall=0.70,
        )

        validation = calibration["metrics"]
        score = validation_score(validation)

        scheduler.step(score)

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
            "lr":
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ],
            "validation_threshold":
                float(calibration["threshold"]),
            "validation_ARG_precision":
                float(calibration["ARG_precision"]),
            "validation_ARG_recall":
                float(calibration["ARG_recall"]),
            "validation_selection_score":
                float(score),
            "validation_end_accuracy":
                float(validation["end_to_end_all_classes"]["accuracy"]),
            "validation_end_weighted_precision":
                float(validation["end_to_end_all_classes"]["weighted_precision"]),
            "validation_end_weighted_f1":
                float(validation["end_to_end_all_classes"]["weighted_f1"]),
            "validation_end_mcc":
                float(validation["end_to_end_all_classes"]["mcc"]),
            "validation_subtype_weighted_precision":
                float(validation["ARG_subtype_oracle_binary"]["weighted_precision"]),
            "validation_binary_weighted_precision":
                float(validation["binary"]["weighted_precision"]),
            "epoch_seconds":
                time.time()
                - started,
        }

        history.append(
            row
        )

        print(
            f"Protein Student {epoch:03d}/{config.student_epochs} | "
            f"loss={row['loss']:.5f} | "
            f"binary={row['binary_loss']:.5f} | "
            f"subtype={row['subtype_loss']:.5f} | "
            f"KD_bin={row['binary_kd_loss']:.5f} | "
            f"KD_sub={row['subtype_kd_loss']:.5f} | "
            f"val_score={score:.5f} | "
            f"val_thr={calibration['threshold']:.2f} | "
            f"ARG_P={calibration['ARG_precision']:.5f} | "
            f"ARG_R={calibration['ARG_recall']:.5f} | "
            f"val_ACC="
            f"{validation['end_to_end_all_classes']['accuracy']:.5f} | "
            f"val_weighted_P="
            f"{validation['end_to_end_all_classes']['weighted_precision']:.5f} | "
            f"val_MCC="
            f"{validation['end_to_end_all_classes']['mcc']:.5f} | "
            f"lr={row['lr']:.2e}",
            flush=True,
        )

        current_precision = float(
            validation["end_to_end_all_classes"]["weighted_precision"]
        )
        current_accuracy = float(
            validation["end_to_end_all_classes"]["accuracy"]
        )

        improved = (
            score > best_score + config.min_delta
            or (
                abs(score - best_score) <= config.min_delta
                and current_precision > best_precision + 1.0e-8
            )
            or (
                abs(score - best_score) <= config.min_delta
                and abs(current_precision - best_precision) <= 1.0e-8
                and current_accuracy > best_accuracy + 1.0e-8
            )
        )

        if improved:
            best_score = float(score)
            best_precision = current_precision
            best_accuracy = current_accuracy
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if epoch in requested_snapshot_epochs:
            snapshot_records.append(
                save_test_arg_representation_snapshot(
                    model,
                    data,
                    config,
                    use_amp,
                    snapshot_dir,
                    epoch=epoch,
                    tag=f"epoch_{epoch:03d}",
                )
            )

        if (
            no_improve
            >= config.student_patience
            and not early_stop_notified
        ):
            print(
                f"Protein Student early-stop criterion first reached at "
                f"epoch {epoch}; best_epoch={best_epoch}. "
                "Continuing through the configured epochs for representation "
                "trajectory and current-run best-epoch determination.",
                flush=True,
            )
            early_stop_notified = True

    if best_state is None:
        raise RuntimeError(
            "Protein Student did not produce a checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    snapshot_records.append(
        save_test_arg_representation_snapshot(
            model,
            data,
            config,
            use_amp,
            snapshot_dir,
            epoch=best_epoch,
            tag=f"run_best_epoch_{best_epoch:03d}",
        )
    )

    return (
        model,
        history,
        best_epoch,
        best_score,
        protein_mean,
        protein_std,
        snapshot_records,
    )


# =============================================================================
# Threshold / evaluation
# =============================================================================

def select_binary_threshold(
    model,
    data,
    config,
    use_amp,
    hb,
):

    hb.update(
        phase="validation_threshold",
        note="validation only",
    )

    outputs = collect_outputs(
        model,
        data[
            "protein_gpu"
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
        ].index_select(
            0,
            indices,
        ).numpy()
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
        ].numpy()
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
        f"Selected validation ARG threshold: "
        f"{best_threshold:.2f} | "
        f"score={best_score:.6f}",
        flush=True,
    )

    return (
        best_threshold,
        rows,
    )


def evaluate_split(
    model,
    split,
    data,
    config,
    use_amp,
    threshold,
):

    outputs = collect_outputs(
        model,
        data[
            "protein_gpu"
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
        binary_threshold=threshold,
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

        pred_subtype = full_names[
            arrays[
                "pred_full"
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
                    pred_subtype,
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
                        pred_subtype
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
                            "ARG_probability"
                        ][
                            local
                        ]
                    ),
                "ARG_subtype_max_probability":
                    float(
                        arrays[
                            "ARG_subtype_max_probability"
                        ][
                            local
                        ]
                    ),
                "binary_threshold":
                    float(
                        threshold
                    ),
                "coverage":
                    1.0,
            }
        )

    return (
        metrics,
        rows,
    )


def summary_rows(metrics):

    order = (
        (
            "ARG_nonARG",
            "binary",
        ),
        (
            "ARG_Subtype",
            "ARG_subtype_oracle_binary",
        ),
        (
            "End_to_end",
            "end_to_end_all_classes",
        ),
    )

    rows = []

    for display_name, internal_name in order:

        value = metrics[
            internal_name
        ]

        rows.append(
            {
                "task":
                    display_name,
                "n":
                    int(
                        value[
                            "n"
                        ]
                    ),
                "accuracy":
                    float(
                        value[
                            "accuracy"
                        ]
                    ),
                "macro_precision":
                    float(
                        value[
                            "macro_precision"
                        ]
                    ),
                "weighted_precision":
                    float(
                        value[
                            "weighted_precision"
                        ]
                    ),
                "micro_precision":
                    float(
                        value[
                            "micro_precision"
                        ]
                    ),
                "macro_recall":
                    float(
                        value[
                            "macro_recall"
                        ]
                    ),
                "weighted_recall":
                    float(
                        value[
                            "weighted_recall"
                        ]
                    ),
                "micro_recall":
                    float(
                        value[
                            "micro_recall"
                        ]
                    ),
                "macro_f1":
                    float(
                        value[
                            "macro_f1"
                        ]
                    ),
                "weighted_f1":
                    float(
                        value[
                            "weighted_f1"
                        ]
                    ),
                "micro_f1":
                    float(
                        value[
                            "micro_f1"
                        ]
                    ),
                "mcc":
                    float(
                        value[
                            "mcc"
                        ]
                    ),
            }
        )

    return rows


def one_vs_rest_metrics(
    true_labels,
    pred_labels,
    class_names,
    class_column,
    split,
):

    true_array = np.asarray(
        true_labels,
        dtype=object,
    )

    pred_array = np.asarray(
        pred_labels,
        dtype=object,
    )

    n = len(
        true_array
    )

    rows = []

    for class_name in class_names:

        t = (
            true_array
            == class_name
        )

        p = (
            pred_array
            == class_name
        )

        tp = int(
            np.sum(
                t
                & p
            )
        )
        fp = int(
            np.sum(
                ~t
                & p
            )
        )
        fn = int(
            np.sum(
                t
                & ~p
            )
        )
        tn = int(
            n
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
                        / n
                        if n
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
                        t.astype(
                            np.int64
                        ),
                        p.astype(
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
                    n,
                "coverage":
                    1.0,
            }
        )

    return rows


def build_test_level_tables(
    prediction_rows,
    data,
):

    true_subtype = [
        row[
            "true_subtype"
        ]
        for row
        in prediction_rows
    ]

    pred_subtype = [
        row[
            "predicted_subtype"
        ]
        for row
        in prediction_rows
    ]

    subtype_rows = one_vs_rest_metrics(
        true_subtype,
        pred_subtype,
        data[
            "arg_names"
        ],
        "Subtype",
        "test",
    )

    true_type = [
        row[
            "true_type"
        ]
        for row
        in prediction_rows
    ]

    pred_type = [
        row[
            "predicted_type"
        ]
        for row
        in prediction_rows
    ]

    type_rows = one_vs_rest_metrics(
        true_type,
        pred_type,
        data[
            "type_names"
        ],
        "Type",
        "test",
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

    pred_binary = [
        (
            NONARG
            if is_nonarg(
                value
            )
            else "ARG"
        )
        for value
        in pred_subtype
    ]

    binary_rows = one_vs_rest_metrics(
        true_binary,
        pred_binary,
        [
            "ARG",
            NONARG,
        ],
        "Class",
        "test",
    )

    return (
        subtype_rows,
        type_rows,
        binary_rows,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()
    print("Device:", "cuda" if torch.cuda.is_available() else "CPU (CUDA required)", flush=True)
    print("Resolved arguments:", vars(args), flush=True)

    for name in (
        "protein_pt",
        "fusion_pt",
        "refined_pt",
        "structure",
        "train_ids",
        "test_ids",
        "best_params",
        "teacher_checkpoint",
        "output_dir",
    ):
        setattr(
            args,
            name,
            getattr(
                args,
                name,
            ).expanduser().resolve(),
        )

    for path, label in (
        (
            args.protein_pt,
            "ESM-2 PT",
        ),
        (
            args.fusion_pt,
            "Fusion PT",
        ),
        (
            args.refined_pt,
            "Refined PT",
        ),
        (
            args.structure,
            "Structure",
        ),
        (
            args.train_ids,
            "New TRAIN ID file",
        ),
        (
            args.test_ids,
            "New TEST ID file",
        ),
        (
            args.best_params,
            "GRID-best parameters",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}\n"
                "Use --overwrite only when you intentionally want to replace it."
            )
        print(
            f"[OVERWRITE] removing old output directory: {args.output_dir}",
            flush=True,
        )
        shutil.rmtree(args.output_dir)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. CPU fallback is disabled."
        )

    torch.cuda.set_device(
        0
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Runtime configuration is loaded from the actual selected GRID-best CSV.
    config = Config()
    config, grid_meta, grid_row = load_grid_best_parameters(
        args.best_params,
        config,
    )

    # Explicit CLI overrides for resource/trajectory controls only.
    # Scientific hyperparameters remain those in the GRID-best CSV.
    config.batch_size = int(config.batch_size)
    config.eval_batch_size = int(config.eval_batch_size)
    config.student_epochs = max(
        int(config.student_epochs),
        max(int(x) for x in args.snapshot_epochs),
    )

    # Previous GRID-best epoch is intentionally NOT reused as the best epoch
    # of this new trajectory run. Only the selected scientific parameters and
    # binary threshold are reused.
    reference_binary_threshold = float(
        grid_meta["reference_binary_threshold"]
    )


    seed_everything(
        config.seed
    )

    use_amp = (
        not args.no_amp
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
            "ARGScope FULL-LENGTH PROTEIN-ONLY STUDENT",
            flush=True,
        )
        print(
            "=" * 125,
            flush=True,
        )
        print(
            f"Script version : {SCRIPT_VERSION}",
            flush=True,
        )
        print(
            f"Protein PT     : {args.protein_pt}",
            flush=True,
        )
        print(
            f"Fusion PT      : {args.fusion_pt}",
            flush=True,
        )
        print(
            f"Refined PT     : {args.refined_pt}",
            flush=True,
        )
        print(
            f"Structure      : {args.structure}",
            flush=True,
        )
        print(
            f"Output         : {args.output_dir}",
            flush=True,
        )

        print(
            f"New TRAIN IDs  : {args.train_ids}",
            flush=True,
        )
        print(
            f"New TEST IDs   : {args.test_ids}",
            flush=True,
        )
        print(
            f"Validation frac: {args.validation_fraction:.3f} (from NEW train only)",
            flush=True,
        )
        print(
            f"Best params    : {args.best_params}",
            flush=True,
        )
        print(
            "FusionTeacher  : RETRAINED from scratch on NEW model-train",
            flush=True,
        )
        print(
            f"Snapshot epochs: {sorted(set(args.snapshot_epochs))} "
            "+ current-run best epoch after training",
            flush=True,
        )
        print(
            f"GRID threshold : {reference_binary_threshold:.2f}",
            flush=True,
        )
        print(
            "Protein input  : ESM-2 650M official residue mean, 1280D",
            flush=True,
        )
        print(
            "Teacher input  : supervised long gated fusion, 512D",
            flush=True,
        )
        print(
            "Prototype prior: residual-GAT refined ARG prototypes, 512D",
            flush=True,
        )
        print(
            "Inference      : ESM-2 protein embedding only",
            flush=True,
        )
        print(
            f"GPU            : {torch.cuda.get_device_name(0)}",
            flush=True,
        )
        print(
            f"CUDA           : {torch.version.cuda}",
            flush=True,
        )
        print(
            f"AMP            : {use_amp}",
            flush=True,
        )
        print(
            "TF32           : enabled",
            flush=True,
        )
        print(
            "CPU fallback   : disabled",
            flush=True,
        )

        print("\n===== GRID-BEST SCIENTIFIC CONFIGURATION =====", flush=True)
        for key, value in asdict(config).items():
            print(f"{key:30s}: {value}", flush=True)
        print(
            f"Selected Teacher class weight power: "
            f"{config.teacher_class_weight_power}",
            flush=True,
        )

        hb.update(
            phase="load_and_align",
        )

        data = load_and_align_data(
            args.protein_pt,
            args.fusion_pt,
            args.refined_pt,
            args.structure,
            args.train_ids,
            args.test_ids,
            args.validation_fraction,
            config.seed,
        )

        # Keep the reporting copies on CPU.
        data[
            "protein_cpu"
        ] = data[
            "protein"
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

        hb.update(
            phase="move_to_gpu",
        )

        # Dataset is small enough to stay on A100 and avoid CPU batch transfer.
        data[
            "protein_gpu"
        ] = data[
            "protein"
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
            f"GPU memory after load: "
            f"allocated={torch.cuda.memory_allocated(0)/(1024**3):.3f} GiB, "
            f"reserved={torch.cuda.memory_reserved(0)/(1024**3):.3f} GiB",
            flush=True,
        )

        # -----------------------------------------------------------------
        # Save the exact split actually used by this run.
        # -----------------------------------------------------------------
        split_manifest_rows = []
        for i, seq_id in enumerate(data["ids"]):
            subtype = data["subtypes"][i]
            split_manifest_rows.append(
                {
                    "sequence_id": seq_id,
                    "subtype": subtype,
                    "type": data["subtype_to_type"][subtype],
                    "external_split": data["external_splits"][i],
                    "model_split": data["model_splits"][i],
                }
            )

        write_csv(
            args.output_dir / "split_manifest.csv",
            split_manifest_rows,
        )

        for split_name in ("train", "validation", "test"):
            ids_for_split = [
                data["ids"][i]
                for i, value in enumerate(data["model_splits"])
                if value == split_name
            ]
            with (args.output_dir / f"{split_name}_ids.txt").open(
                "w", encoding="utf-8"
            ) as handle:
                for seq_id in ids_for_split:
                    handle.write(seq_id + "\n")

        # Also preserve the exact external split supplied by the user.
        for split_name in ("train", "test"):
            ids_for_split = [
                data["ids"][i]
                for i, value in enumerate(data["external_splits"])
                if value == split_name
            ]
            with (args.output_dir / f"external_{split_name}_ids.txt").open(
                "w", encoding="utf-8"
            ) as handle:
                for seq_id in ids_for_split:
                    handle.write(seq_id + "\n")

        # -----------------------------------------------------------------
        # Stage 1: Fusion Teacher
        # IMPORTANT: retrain on the NEW split. Do NOT load old teacher weights.
        # -----------------------------------------------------------------

        hb.update(
            phase="train_new_split_teacher",
            note="old teacher checkpoint intentionally ignored",
        )

        seed_everything(config.seed)

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
        )

        print(
            f"Retrained FusionTeacher on NEW split | "
            f"best_epoch={teacher_best_epoch} | "
            f"best_score={teacher_best_score:.6f}",
            flush=True,
        )

        # -----------------------------------------------------------------
        # Stage 2: Protein-only Student
        # -----------------------------------------------------------------

        # Reset RNG immediately before Student retraining for reproducibility.
        seed_everything(config.seed)

        snapshot_dir = (
            args.output_dir
            / "epoch_features"
        )

        (
            student,
            student_history,
            student_best_epoch,
            student_best_score,
            protein_mean,
            protein_std,
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

        # -----------------------------------------------------------------
        # Validation-only binary threshold
        # -----------------------------------------------------------------

        # Reuse the already selected validation threshold from the GRID-best
        # search for final reporting. Epoch feature snapshots themselves do
        # not depend on this threshold.
        binary_threshold = reference_binary_threshold
        threshold_rows = []
        print(
            f"Using fixed GRID-best ARG threshold: {binary_threshold:.2f}",
            flush=True,
        )

        # -----------------------------------------------------------------
        # TRAIN metrics: reporting only.
        # -----------------------------------------------------------------

        hb.update(
            phase="final_train_reporting",
            note="does not affect model selection",
        )

        train_metrics, train_prediction_rows = evaluate_split(
            student,
            "train",
            data,
            config,
            use_amp,
            binary_threshold,
        )

        # -----------------------------------------------------------------
        # TEST: only after selection is fixed.
        # -----------------------------------------------------------------

        hb.update(
            phase="FINAL_TEST",
            note="teacher/student/threshold already fixed",
        )

        test_metrics, test_prediction_rows = evaluate_split(
            student,
            "test",
            data,
            config,
            use_amp,
            binary_threshold,
        )

        print(
            "\n===== FINAL TEST RESULTS =====",
            flush=True,
        )

        for row in summary_rows(
            test_metrics
        ):
            print(
                f"{row['task']:14s} | "
                f"n={row['n']:,} | "
                f"ACC={row['accuracy']:.6f} | "
                f"macro_P={row['macro_precision']:.6f} | "
                f"weighted_P={row['weighted_precision']:.6f} | "
                f"micro_P={row['micro_precision']:.6f} | "
                f"macro_R={row['macro_recall']:.6f} | "
                f"weighted_R={row['weighted_recall']:.6f} | "
                f"micro_R={row['micro_recall']:.6f} | "
                f"macro_F1={row['macro_f1']:.6f} | "
                f"weighted_F1={row['weighted_f1']:.6f} | "
                f"micro_F1={row['micro_f1']:.6f} | "
                f"MCC={row['mcc']:.6f}",
                flush=True,
            )

        # -----------------------------------------------------------------
        # Output files
        # -----------------------------------------------------------------

        loss_rows = []

        for row in teacher_history:
            loss_rows.append(
                dict(
                    row
                )
            )

        for row in student_history:
            loss_rows.append(
                dict(
                    row
                )
            )

        write_csv(
            args.output_dir
            / "training_loss.csv",
            loss_rows,
        )

        write_csv(
            args.output_dir
            / "train_metrics.csv",
            summary_rows(
                train_metrics
            ),
        )

        write_csv(
            args.output_dir
            / "train_sequence_predictions.csv",
            train_prediction_rows,
        )

        write_csv(
            args.output_dir
            / "test_metrics.csv",
            summary_rows(
                test_metrics
            ),
        )

        write_csv(
            args.output_dir
            / "test_sequence_predictions.csv",
            test_prediction_rows,
        )

        (
            test_subtype_rows,
            test_type_rows,
            test_binary_rows,
        ) = build_test_level_tables(
            test_prediction_rows,
            data,
        )

        write_csv(
            args.output_dir
            / "test_subtype_level_metrics.csv",
            test_subtype_rows,
        )

        write_csv(
            args.output_dir
            / "test_type_level_metrics.csv",
            test_type_rows,
        )

        write_csv(
            args.output_dir
            / "test_ARG_nonARG_level_metrics.csv",
            test_binary_rows,
        )

        write_csv(
            args.output_dir
            / "validation_binary_threshold_sweep.csv",
            threshold_rows,
        )

        # -----------------------------------------------------------------
        # Checkpoint
        # -----------------------------------------------------------------

        checkpoint = {
            "model_name":
                (
                    "ARGScope full-length Protein-only hierarchical "
                    "fusion-teacher refined-prototype Student"
                ),

            "script_version":
                SCRIPT_VERSION,

            "protein_embedding_definition":
                (
                    "ESM-2 esm2_t33_650M_UR50D; "
                    "official first-1022-aa truncation; "
                    "mean of real residue representations excluding BOS/EOS; "
                    "1280 dimensions"
                ),

            "student_state_dict":
                {
                    key:
                        value.detach()
                        .cpu()
                    for key, value
                    in student.state_dict().items()
                },

            "teacher_state_dict":
                {
                    key:
                        value.detach()
                        .cpu()
                    for key, value
                    in teacher.state_dict().items()
                },

            "protein_input_dimension":
                1280,

            "embedding_dimension":
                512,

            "arg_subtype_names":
                data[
                    "arg_names"
                ],

            "full_class_names":
                [
                    NONARG
                ]
                + data[
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

            "protein_train_mean":
                protein_mean.detach()
                .cpu(),

            "protein_train_std":
                protein_std.detach()
                .cpu(),

            "refined_ARG_subtype_prototypes":
                data[
                    "refined"
                ].detach()
                .cpu(),

            "selected_binary_ARG_probability_threshold":
                float(
                    binary_threshold
                ),

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

            "train_metrics_reporting_only":
                train_metrics,

            "test_metrics":
                test_metrics,

            "configuration":
                asdict(
                    config
                ),

            "subtype_to_type":
                data[
                    "subtype_to_type"
                ],

            "type_names":
                data[
                    "type_names"
                ],

            "validation_binary_threshold_sweep":
                threshold_rows,

            "grid_best_parameter_row":
                grid_row,

            "reference_GRID_binary_threshold":
                float(reference_binary_threshold),

            "epoch_representation_snapshots":
                snapshot_records,

            "training_rule":
                (
                    "NEW external train/test IDs are authoritative; "
                    "validation is carved only from NEW external train while "
                    "retaining at least one model-train sample per ARG subtype; "
                    "FusionTeacher and ProteinStudent are retrained on the NEW split; "
                    "NEW external TEST is evaluated only after model selection; "
                    "inference uses ESM-2 protein embeddings only"
                ),

            "source_protein_pt":
                str(
                    args.protein_pt
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
        }

        checkpoint_path = (
            args.output_dir
            / "protein_100pct_epoch_features_run_best_model.pt"
        )

        hb.update(
            phase="saving",
            note=str(
                checkpoint_path
            ),
        )

        safe_save(
            checkpoint,
            checkpoint_path,
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
                        "current_run_best_epoch":
                            int(student_best_epoch),
                        "current_run_best_validation_score":
                            float(student_best_score),
                        "student_epochs_completed":
                            int(config.student_epochs),
                        "requested_snapshot_epochs":
                            sorted(set(int(x) for x in args.snapshot_epochs)),
                        "best_epoch_snapshot":
                            f"run_best_epoch_{student_best_epoch:03d}_test_ARG_type_features.pt",
                        "selection_note":
                            (
                                "Best epoch is determined from validation "
                                "performance in THIS training run. No previous "
                                "best epoch is assumed."
                            ),
                    }
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        metrics_json = (
            args.output_dir
            / "test_metrics.json"
        )

        temporary = metrics_json.with_suffix(
            ".json.tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                json_safe(
                    {
                        "test":
                            test_metrics,
                        "selected_binary_ARG_probability_threshold":
                            binary_threshold,
                    }
                ),
                handle,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temporary,
            metrics_json,
        )

        print(
            "\n"
            + "=" * 125,
            flush=True,
        )
        print(
            "PROTEIN STUDENT TRAINING COMPLETED",
            flush=True,
        )
        print(
            "=" * 125,
            flush=True,
        )
        print(
            f"Teacher best epoch  : {teacher_best_epoch}",
            flush=True,
        )
        print(
            f"Student best epoch  : {student_best_epoch}",
            flush=True,
        )
        print(
            f"ARG threshold       : {binary_threshold:.2f}",
            flush=True,
        )
        print(
            f"Model               : {checkpoint_path}",
            flush=True,
        )
        print(
            f"Train metrics       : "
            f"{args.output_dir / 'train_metrics.csv'}",
            flush=True,
        )
        print(
            f"Final TEST metrics  : "
            f"{args.output_dir / 'test_metrics.csv'}",
            flush=True,
        )
        print(
            f"Subtype metrics     : "
            f"{args.output_dir / 'test_subtype_level_metrics.csv'}",
            flush=True,
        )
        print(
            f"Type metrics        : "
            f"{args.output_dir / 'test_type_level_metrics.csv'}",
            flush=True,
        )

        print(
            f"Epoch features      : {args.output_dir / 'epoch_features'}",
            flush=True,
        )
        print(
            f"This run best epoch : {student_best_epoch}",
            flush=True,
        )
        print(
            "Plotting             : disabled; only raw epoch feature files saved",
            flush=True,
        )
        print(
            f"Peak GPU memory     : "
            f"{torch.cuda.max_memory_allocated(0)/(1024**3):.3f} GiB",
            flush=True,
        )
        print(
            f"Elapsed minutes     : "
            f"{(time.time()-started)/60.0:.3f}",
            flush=True,
        )

    finally:
        hb.stop()


if __name__ == "__main__":
    main()
