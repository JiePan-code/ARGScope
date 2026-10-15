#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BGCScope full-length DNA-only hierarchical Student (local Windows version).

Inputs
------
1. DNABERT2_BGC/BGC_nonBGC_dnabert2_mean.pt
      Full-length DNABERT-2 DNA embeddings [N, 768].
2. Long_fusion/long_fusion.pt
      Protein+DNA gated-fusion embeddings [N, 512], plus types, subtypes,
      external train/test split and leakage-free model train/validation/test.
3. Refined_subtype_prototypes_long/
      refined_BGC_subtype_prototypes_residual_GAT.pt
      Residual-GAT refined BGC subtype prototypes [C, 512] and the train-only
      nonBGC fusion centroid.

Training
--------
Stage 1 retrains a lightweight FusionTeacher on the current model-train split.
Stage 2 trains a DNAStudent that receives only DNABERT-2 embeddings and learns
binary BGC/nonBGC classification, BGC subtype classification, teacher knowledge
distillation, feature alignment, prototype alignment and classifier-head
anchoring.

Leakage policy
--------------
* The split metadata already stored in long_fusion.pt is authoritative.
* DNA normalization and class weighting use model-train only.
* Validation selects teacher/student epochs and the binary BGC threshold.
* Test is evaluated only after teacher, student and threshold are fixed.
* Refined prototypes must cover every BGC subtype exactly.

Local execution
---------------
CUDA is selected automatically when available. CPU fallback is supported but
will be substantially slower. Inputs stay in CPU memory and only minibatches
are transferred to the selected device, making the script suitable for local
GPUs with less memory than an A100.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_VERSION = "2026-09-15-bgcscope-local-dna-student-v1"
NONBGC = "nonBGC"

ROOT = Path(__file__).resolve().parent

DEFAULT_DNA_PT = ROOT / "DNABERT2_BGC" / "BGC_nonBGC_dnabert2_mean.pt"
DEFAULT_FUSION_PT = ROOT / "Long_fusion" / "long_fusion.pt"
DEFAULT_REFINED_PT = (
    ROOT
    / "Refined_subtype_prototypes_long"
    / "refined_BGC_subtype_prototypes_residual_GAT.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "DNA_student_BGC"


@dataclass
class Config:
    seed: int = 20260915
    batch_size: int = 128
    eval_batch_size: int = 512

    # Preserved from the existing DNA-student configuration. These can be
    # shortened from the command line for a quick local smoke test.
    teacher_epochs: int = 300
    student_epochs: int = 300
    teacher_patience: int = 15
    student_patience: int = 25

    teacher_lr: float = 3.0e-4
    student_lr: float = 2.0e-4
    weight_decay: float = 5.0e-5

    hidden_dim: int = 1024
    embedding_dim: int = 512
    dropout: float = 0.10

    logit_scale_binary: float = 16.0
    logit_scale_subtype: float = 24.0
    label_smoothing: float = 0.002
    class_weight_power: float = 0.0
    teacher_class_weight_power: float = 0.0
    temperature: float = 3.0

    teacher_binary_ce: float = 1.0
    teacher_subtype_ce: float = 1.0
    teacher_refined_anchor: float = 0.05

    student_binary_ce: float = 1.0
    student_subtype_ce: float = 1.0
    student_binary_kd: float = 0.20
    student_subtype_kd: float = 0.05
    student_feature_align: float = 0.05
    student_prototype_align: float = 0.05
    student_head_anchor: float = 0.05

    gradient_clip: float = 1.0
    min_delta: float = 1.0e-5
    minimum_bgc_recall: float = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a local full-length DNA-only BGC/nonBGC and BGC-subtype "
            "student using DNABERT-2, gated fusion and refined BGC prototypes."
        )
    )
    parser.add_argument("--dna-pt", type=Path, default=DEFAULT_DNA_PT)
    parser.add_argument("--fusion-pt", type=Path, default=DEFAULT_FUSION_PT)
    parser.add_argument("--refined-pt", type=Path, default=DEFAULT_REFINED_PT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--teacher-epochs", type=int, default=Config.teacher_epochs)
    parser.add_argument("--student-epochs", type=int, default=Config.student_epochs)
    parser.add_argument("--teacher-patience", type=int, default=Config.teacher_patience)
    parser.add_argument("--student-patience", type=int, default=Config.student_patience)
    parser.add_argument("--teacher-lr", type=float, default=Config.teacher_lr)
    parser.add_argument("--student-lr", type=float, default=Config.student_lr)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--snapshot-epochs",
        nargs="+",
        type=int,
        default=[0, 20, 50, 100, 150],
        help="Reached Student epochs whose test-BGC 512D features are saved.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def update_config(args: argparse.Namespace) -> Config:
    config = Config()
    config.seed = args.seed
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.teacher_epochs = args.teacher_epochs
    config.student_epochs = args.student_epochs
    config.teacher_patience = args.teacher_patience
    config.student_patience = args.student_patience
    config.teacher_lr = args.teacher_lr
    config.student_lr = args.student_lr
    return config


def canonical_id(value: Any) -> str:
    text = str(value).strip()
    return text.split()[0] if text else ""


def clean_label(value: Any) -> str:
    return " ".join(str(value).strip().split())


def compact_label(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def is_nonbgc(value: Any) -> bool:
    return compact_label(value) in {
        "nonbgc",
        "nonbiosyntheticgenecluster",
        "negative",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch cannot use CUDA.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(obj, temporary)
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"Invalid temporary checkpoint: {temporary}")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def extract_embedding_pt(path: Path, name: str) -> Tuple[List[str], torch.Tensor, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    obj = torch_load_cpu(path)
    if not isinstance(obj, dict) or "ids" not in obj or "embeddings" not in obj:
        raise KeyError(f"{name} PT requires keys 'ids' and 'embeddings': {path}")
    ids = [canonical_id(value) for value in obj["ids"]]
    embeddings = obj["embeddings"]
    if any(not seq_id for seq_id in ids):
        raise ValueError(f"Empty ID in {name}.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {name}.")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise ValueError(f"{name} embeddings must be a 2D Tensor.")
    if embeddings.shape[0] != len(ids):
        raise ValueError(f"{name} ID/embedding count mismatch.")
    embeddings = embeddings.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError(f"NaN/Inf detected in {name} embeddings.")
    return ids, embeddings, obj


def load_and_align_data(
    dna_path: Path,
    fusion_path: Path,
    refined_path: Path,
) -> Dict[str, Any]:
    dna_ids, dna, dna_obj = extract_embedding_pt(
        dna_path, "DNABERT-2 dna"
    )
    fusion_ids, fusion, fusion_obj = extract_embedding_pt(
        fusion_path, "protein-DNA fusion"
    )

    if dna.shape[1] != 768:
        raise ValueError(f"Expected dna dimension 768; found {dna.shape[1]}.")
    if fusion.shape[1] != 512:
        raise ValueError(f"Expected fusion dimension 512; found {fusion.shape[1]}.")
    if set(dna_ids) != set(fusion_ids):
        raise ValueError(
            "DNA and fusion ID sets differ: "
            f"dna_only={len(set(dna_ids)-set(fusion_ids))}, "
            f"fusion_only={len(set(fusion_ids)-set(dna_ids))}."
        )

    dna_index = {seq_id: index for index, seq_id in enumerate(dna_ids)}
    order = torch.tensor([dna_index[seq_id] for seq_id in fusion_ids], dtype=torch.long)
    dna = dna.index_select(0, order).contiguous()

    required_fusion = {"types", "subtypes", "model_splits"}
    missing_fusion = required_fusion - set(fusion_obj)
    if missing_fusion:
        raise KeyError(f"Fusion PT lacks keys: {sorted(missing_fusion)}")

    ids = fusion_ids
    types = [clean_label(value) for value in fusion_obj["types"]]
    subtypes = [clean_label(value) for value in fusion_obj["subtypes"]]
    model_splits = [clean_label(value).lower() for value in fusion_obj["model_splits"]]
    external_splits = [
        clean_label(value).lower()
        for value in fusion_obj.get("splits", model_splits)
    ]
    n = len(ids)
    for name, values in {
        "types": types,
        "subtypes": subtypes,
        "model_splits": model_splits,
        "external_splits": external_splits,
    }.items():
        if len(values) != n:
            raise ValueError(f"Fusion {name} length mismatch: {len(values)} vs {n}.")
    if set(model_splits) - {"train", "validation", "test"}:
        raise ValueError(f"Invalid model_splits: {sorted(set(model_splits))}")
    if any(split not in {"train", "test"} for split in external_splits):
        raise ValueError(f"Invalid external splits: {sorted(set(external_splits))}")
    for external, model_split in zip(external_splits, model_splits):
        if external == "test" and model_split != "test":
            raise RuntimeError("External-test leakage into train/validation detected.")
        if external == "train" and model_split == "test":
            raise RuntimeError("External train was incorrectly assigned to model test.")

    inconsistent_negative = []
    for seq_id, type_name, subtype in zip(ids, types, subtypes):
        if is_nonbgc(type_name) != is_nonbgc(subtype):
            inconsistent_negative.append((seq_id, type_name, subtype))
    if inconsistent_negative:
        raise ValueError(
            "Inconsistent nonBGC labels between Type and Subtype; "
            f"examples={inconsistent_negative[:10]}"
        )

    if not refined_path.is_file():
        raise FileNotFoundError(f"Missing refined prototype PT: {refined_path}")
    refined_obj = torch_load_cpu(refined_path)
    if not isinstance(refined_obj, dict):
        raise TypeError("Refined prototype PT must contain a dictionary.")
    required_refined = {
        "prototype_subtypes",
        "prototype_types",
        "refined_prototypes",
        "nonbgc_reference_centroid",
    }
    missing_refined = required_refined - set(refined_obj)
    if missing_refined:
        raise KeyError(f"Refined prototype PT lacks keys: {sorted(missing_refined)}")

    bgc_names = [clean_label(value) for value in refined_obj["prototype_subtypes"]]
    prototype_types = [clean_label(value) for value in refined_obj["prototype_types"]]
    refined = refined_obj["refined_prototypes"]
    nonbgc_centroid = refined_obj["nonbgc_reference_centroid"]
    if len(bgc_names) != len(set(bgc_names)):
        raise ValueError("Duplicate BGC subtype names in refined prototype PT.")
    if any(is_nonbgc(name) for name in bgc_names):
        raise ValueError("nonBGC must not be included among BGC subtype prototypes.")
    if len(prototype_types) != len(bgc_names):
        raise ValueError("prototype_subtypes/prototype_types length mismatch.")
    if not isinstance(refined, torch.Tensor) or refined.ndim != 2:
        raise ValueError("refined_prototypes must be a 2D Tensor.")
    if tuple(refined.shape) != (len(bgc_names), 512):
        raise ValueError(
            f"Refined prototype shape mismatch: names={len(bgc_names)}, "
            f"shape={tuple(refined.shape)}."
        )
    refined = F.normalize(refined.detach().cpu().float(), p=2, dim=1)
    nonbgc_centroid = F.normalize(
        nonbgc_centroid.detach().cpu().float().reshape(-1), p=2, dim=0
    )
    if nonbgc_centroid.numel() != 512:
        raise ValueError(
            f"Expected 512D nonBGC centroid; found {nonbgc_centroid.numel()}."
        )

    observed_bgc = {
        subtype
        for type_name, subtype in zip(types, subtypes)
        if not is_nonbgc(type_name) and not is_nonbgc(subtype)
    }
    if observed_bgc != set(bgc_names):
        raise ValueError(
            "BGC subtype/prototype sets differ: "
            f"without_prototype={sorted(observed_bgc-set(bgc_names))[:20]}, "
            f"extra_prototype={sorted(set(bgc_names)-observed_bgc)[:20]}."
        )

    subtype_to_type = {
        subtype: type_name for subtype, type_name in zip(bgc_names, prototype_types)
    }
    subtype_to_type[NONBGC] = NONBGC
    for type_name, subtype in zip(types, subtypes):
        if is_nonbgc(subtype):
            continue
        if subtype_to_type[subtype] != type_name:
            raise ValueError(
                f"Subtype-to-type mismatch for {subtype}: "
                f"{subtype_to_type[subtype]} vs {type_name}."
            )

    bgc_to_index = {name: index for index, name in enumerate(bgc_names)}
    binary_targets = torch.tensor(
        [0 if is_nonbgc(subtype) else 1 for subtype in subtypes], dtype=torch.long
    )
    bgc_targets = torch.tensor(
        [-1 if is_nonbgc(subtype) else bgc_to_index[subtype] for subtype in subtypes],
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
            f"{ {key:int(value.numel()) for key,value in split_indices.items()} }"
        )
    train_bgc_names = {
        subtypes[index]
        for index in split_indices["train"].tolist()
        if not is_nonbgc(subtypes[index])
    }
    if train_bgc_names != set(bgc_names):
        raise ValueError(
            "Every BGC subtype must occur in model-train; missing="
            f"{sorted(set(bgc_names)-train_bgc_names)[:20]}."
        )

    print(f"DNABERT-2 shape             : {tuple(dna.shape)}", flush=True)
    print(f"Fusion shape            : {tuple(fusion.shape)}", flush=True)
    print(f"Refined prototype shape : {tuple(refined.shape)}", flush=True)
    print(f"BGC subtype count        : {len(bgc_names):,}", flush=True)
    print(f"BGC Type count           : {len(set(prototype_types)):,}", flush=True)
    print(f"BGC total                : {int((binary_targets == 1).sum()):,}", flush=True)
    print(f"nonBGC total            : {int((binary_targets == 0).sum()):,}", flush=True)
    for split, indices in split_indices.items():
        local_binary = binary_targets.index_select(0, indices)
        print(
            f"{split:10s}: total={indices.numel():,}, "
            f"BGC={int((local_binary==1).sum()):,}, "
            f"nonBGC={int((local_binary==0).sum()):,}",
            flush=True,
        )

    return {
        "ids": ids,
        "types": types,
        "subtypes": subtypes,
        "external_splits": external_splits,
        "model_splits": model_splits,
        "split_indices": split_indices,
        "dna": dna,
        "fusion": fusion,
        "binary_targets": binary_targets,
        "bgc_targets": bgc_targets,
        "bgc_names": bgc_names,
        "bgc_to_index": bgc_to_index,
        "prototype_types": prototype_types,
        "subtype_to_type": subtype_to_type,
        "type_names": sorted(set(prototype_types), key=str.lower),
        "refined": refined,
        "nonbgc_centroid": nonbgc_centroid,
        "dna_obj": dna_obj,
        "fusion_obj": fusion_obj,
        "refined_obj": refined_obj,
    }


def mild_class_weights(
    targets: torch.Tensor,
    number_classes: int,
    power: float,
) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=number_classes).float()
    if bool((counts == 0).any()):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Missing model-train class(es): {missing[:20]}")
    weights = (counts.mean() / counts).pow(power)
    weights = weights / weights.mean()
    return weights.clamp(0.70, 2.50)


class CosineHead(nn.Module):
    def __init__(self, number_classes: int, dimension: int, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(number_classes, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(
            F.normalize(features, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )

    def normalized_weight(self) -> torch.Tensor:
        return F.normalize(self.weight.float(), p=2, dim=1)


class FusionTeacher(nn.Module):
    def __init__(self, refined: torch.Tensor, config: Config):
        super().__init__()
        self.binary_head = CosineHead(2, config.embedding_dim, config.logit_scale_binary)
        self.bgc_head = CosineHead(
            refined.shape[0], config.embedding_dim, config.logit_scale_subtype
        )
        with torch.no_grad():
            self.bgc_head.weight.copy_(refined)

    def encode(self, fusion: torch.Tensor) -> torch.Tensor:
        return F.normalize(fusion, p=2, dim=1)

    def forward(self, fusion: torch.Tensor):
        representation = self.encode(fusion)
        return (
            self.binary_head(representation),
            self.bgc_head(representation),
            representation,
        )


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(inputs + self.net(inputs))


class DNAStudent(nn.Module):
    def __init__(
        self,
        dna_mean: torch.Tensor,
        dna_std: torch.Tensor,
        number_bgc_subtypes: int,
        config: Config,
    ):
        super().__init__()
        self.register_buffer("dna_mean", dna_mean.clone())
        self.register_buffer("dna_std", dna_std.clone())
        self.input_layers = nn.Sequential(
            nn.Linear(768, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(config.hidden_dim, config.dropout),
            ResidualBlock(config.hidden_dim, config.dropout),
        )
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )
        self.binary_head = CosineHead(2, config.embedding_dim, config.logit_scale_binary)
        self.bgc_head = CosineHead(
            number_bgc_subtypes, config.embedding_dim, config.logit_scale_subtype
        )

    def encode(self, dna: torch.Tensor) -> torch.Tensor:
        normalized = (dna - self.dna_mean) / self.dna_std
        hidden = self.input_layers(normalized)
        hidden = self.blocks(hidden)
        return F.normalize(self.output_layers(hidden), p=2, dim=1)

    def forward(self, dna: torch.Tensor):
        representation = self.encode(dna)
        return (
            self.binary_head(representation),
            self.bgc_head(representation),
            representation,
        )


def batch_indices(
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> Iterable[torch.Tensor]:
    current = indices
    if shuffle:
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        permutation = torch.randperm(indices.numel(), generator=generator)
        current = indices.index_select(0, permutation)
    for start in range(0, current.numel(), batch_size):
        yield current[start:start + batch_size]


def cosine_pull(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (
        1.0
        - (
            F.normalize(first.float(), p=2, dim=1)
            * F.normalize(second.float().detach(), p=2, dim=1)
        ).sum(dim=1)
    ).mean()


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student_logits.float() / temperature, dim=1),
            F.softmax(teacher_logits.float().detach() / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature
        * temperature
    )


def initialize_binary_centers(
    model: nn.Module,
    fusion: torch.Tensor,
    binary_targets: torch.Tensor,
    train_indices: torch.Tensor,
) -> None:
    normalized = F.normalize(fusion.index_select(0, train_indices), p=2, dim=1)
    local_targets = binary_targets.index_select(0, train_indices)
    centers = []
    for class_index in (0, 1):
        selected = normalized[local_targets == class_index]
        if selected.shape[0] == 0:
            raise ValueError(f"No model-train samples for binary class {class_index}.")
        centers.append(F.normalize(selected.mean(dim=0), p=2, dim=0))
    with torch.no_grad():
        model.binary_head.weight.copy_(torch.stack(centers))


def confusion_metrics(
    true: torch.Tensor,
    predicted: torch.Tensor,
    number_classes: int | None = None,
) -> Dict[str, Any]:
    true = true.detach().cpu().long().reshape(-1)
    predicted = predicted.detach().cpu().long().reshape(-1)
    if true.numel() == 0:
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
    inferred = int(torch.maximum(true.max(), predicted.max()).item()) + 1
    number_classes = max(inferred, int(number_classes or 0))
    encoded = true * number_classes + predicted
    matrix = torch.bincount(
        encoded, minlength=number_classes * number_classes
    ).reshape(number_classes, number_classes).double()
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)
    precision = true_positive / predicted_count.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    active = (support + predicted_count) > 0
    total = matrix.sum()
    accuracy = true_positive.sum() / total
    macro_precision = precision[active].mean()
    macro_recall = recall[active].mean()
    macro_f1 = f1[active].mean()
    weighted_precision = (precision * support).sum() / support.sum()
    weighted_recall = (recall * support).sum() / support.sum()
    weighted_f1 = (f1 * support).sum() / support.sum()
    correct = matrix.trace()
    numerator = correct * total - torch.dot(support, predicted_count)
    denominator = torch.sqrt(
        (total * total - torch.dot(predicted_count, predicted_count)).clamp_min(0.0)
        * (total * total - torch.dot(support, support)).clamp_min(0.0)
    )
    mcc = numerator / denominator if float(denominator) > 0.0 else total.new_tensor(0.0)
    return {
        "n": int(true.numel()),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "weighted_precision": float(weighted_precision),
        "micro_precision": float(accuracy),
        "macro_recall": float(macro_recall),
        "weighted_recall": float(weighted_recall),
        "micro_recall": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(accuracy),
        "mcc": float(mcc),
    }


@torch.inference_mode()
def collect_outputs(
    model: nn.Module,
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, torch.Tensor]:
    model.eval()
    binary_logits = []
    bgc_logits = []
    features = []
    output_indices = []
    for local_indices in batch_indices(indices, batch_size, shuffle=False):
        inputs_batch = inputs.index_select(0, local_indices).to(device, non_blocking=True)
        with amp_context(device, use_amp):
            binary, subtype, representation = model(inputs_batch)
        binary_logits.append(binary.float().cpu())
        bgc_logits.append(subtype.float().cpu())
        features.append(representation.float().cpu())
        output_indices.append(local_indices.cpu())
    return {
        "indices": torch.cat(output_indices),
        "binary_logits": torch.cat(binary_logits),
        "bgc_logits": torch.cat(bgc_logits),
        "features": torch.cat(features),
    }


def predictions_and_metrics(
    outputs: Dict[str, torch.Tensor],
    data: Dict[str, Any],
    binary_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    indices = outputs["indices"]
    true_binary = data["binary_targets"].index_select(0, indices)
    true_bgc = data["bgc_targets"].index_select(0, indices)
    binary_probability = torch.softmax(outputs["binary_logits"], dim=1)[:, 1]
    bgc_probability = torch.softmax(outputs["bgc_logits"], dim=1)
    predicted_binary = (binary_probability >= binary_threshold).long()
    predicted_bgc = bgc_probability.argmax(dim=1)
    true_full = torch.where(true_binary == 0, 0, true_bgc + 1)
    predicted_full = torch.where(predicted_binary == 0, 0, predicted_bgc + 1)
    bgc_mask = true_binary == 1
    metrics = {
        "binary": confusion_metrics(true_binary, predicted_binary, 2),
        "BGC_subtype_oracle_binary": confusion_metrics(
            true_bgc[bgc_mask], predicted_bgc[bgc_mask], len(data["bgc_names"])
        ),
        "end_to_end_all_classes": confusion_metrics(
            true_full, predicted_full, len(data["bgc_names"]) + 1
        ),
    }
    arrays = {
        "indices": indices,
        "true_binary": true_binary,
        "predicted_binary": predicted_binary,
        "true_bgc": true_bgc,
        "predicted_bgc": predicted_bgc,
        "true_full": true_full,
        "predicted_full": predicted_full,
        "BGC_probability": binary_probability,
        "BGC_subtype_max_probability": bgc_probability.max(dim=1).values,
    }
    return metrics, arrays


def validation_score(metrics: Dict[str, Any]) -> float:
    full = metrics["end_to_end_all_classes"]
    subtype = metrics["BGC_subtype_oracle_binary"]
    binary = metrics["binary"]
    return (
        0.45 * subtype["weighted_precision"]
        + 0.25 * full["weighted_precision"]
        + 0.15 * full["accuracy"]
        + 0.10 * full["mcc"]
        + 0.05 * binary["weighted_precision"]
    )


def calibrate_threshold_from_outputs(
    outputs: Dict[str, torch.Tensor],
    data: Dict[str, Any],
    minimum_bgc_recall: float,
) -> Dict[str, Any]:
    rows = []
    for integer_threshold in range(30, 91):
        threshold = integer_threshold / 100.0
        metrics, arrays = predictions_and_metrics(outputs, data, threshold)
        true_binary = arrays["true_binary"]
        predicted_binary = arrays["predicted_binary"]
        true_positive = int(((true_binary == 1) & (predicted_binary == 1)).sum())
        false_positive = int(((true_binary == 0) & (predicted_binary == 1)).sum())
        false_negative = int(((true_binary == 1) & (predicted_binary == 0)).sum())
        bgc_precision = true_positive / max(1, true_positive + false_positive)
        bgc_recall = true_positive / max(1, true_positive + false_negative)
        rows.append(
            {
                "threshold": threshold,
                "BGC_precision": bgc_precision,
                "BGC_recall": bgc_recall,
                "binary_accuracy": metrics["binary"]["accuracy"],
                "validation_score": validation_score(metrics),
                "metrics": metrics,
            }
        )
    eligible = [row for row in rows if row["BGC_recall"] >= minimum_bgc_recall]
    candidates = eligible if eligible else rows
    best = max(
        candidates,
        key=lambda row: (
            row["validation_score"],
            row["BGC_precision"],
            row["binary_accuracy"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "threshold": best["threshold"],
        "BGC_precision": best["BGC_precision"],
        "BGC_recall": best["BGC_recall"],
        "score": best["validation_score"],
        "metrics": best["metrics"],
        "rows": [
            {key: value for key, value in row.items() if key != "metrics"}
            for row in rows
        ],
        "minimum_recall_satisfied": bool(eligible),
    }


def train_teacher(
    data: Dict[str, Any],
    config: Config,
    device: torch.device,
    use_amp: bool,
) -> Tuple[nn.Module, List[Dict[str, Any]], int, float]:
    train_indices = data["split_indices"]["train"]
    validation_indices = data["split_indices"]["validation"]
    train_bgc_targets = data["bgc_targets"].index_select(0, train_indices)
    subtype_weights = mild_class_weights(
        train_bgc_targets[train_bgc_targets >= 0],
        len(data["bgc_names"]),
        config.teacher_class_weight_power,
    ).to(device)
    model = FusionTeacher(data["refined"], config).to(device)
    initialize_binary_centers(
        model,
        data["fusion"],
        data["binary_targets"],
        train_indices,
    )
    refined_device = data["refined"].to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.teacher_lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1.0e-6
    )
    scaler = make_grad_scaler(use_amp)
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    print("\n===== STAGE 1: TRAIN BGC FUSION TEACHER =====", flush=True)
    for epoch in range(1, config.teacher_epochs + 1):
        model.train()
        totals = defaultdict(float)
        seen = 0
        for indices in batch_indices(
            train_indices, config.batch_size, shuffle=True, seed=config.seed + epoch
        ):
            fusion_batch = data["fusion"].index_select(0, indices).to(
                device, non_blocking=True
            )
            binary_y = data["binary_targets"].index_select(0, indices).to(device)
            bgc_y = data["bgc_targets"].index_select(0, indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                binary_logits, bgc_logits, _ = model(fusion_batch)
                binary_loss = F.cross_entropy(
                    binary_logits.float(), binary_y, label_smoothing=config.label_smoothing
                )
                bgc_mask = bgc_y >= 0
                if bool(bgc_mask.any()):
                    subtype_loss = F.cross_entropy(
                        bgc_logits[bgc_mask].float(),
                        bgc_y[bgc_mask],
                        weight=subtype_weights,
                        label_smoothing=config.label_smoothing,
                    )
                else:
                    subtype_loss = binary_loss.new_zeros(())
                refined_anchor = (
                    1.0
                    - (model.bgc_head.normalized_weight() * refined_device).sum(dim=1)
                ).mean()
                loss = (
                    config.teacher_binary_ce * binary_loss
                    + config.teacher_subtype_ce * subtype_loss
                    + config.teacher_refined_anchor * refined_anchor
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_n = int(binary_y.numel())
            seen += batch_n
            totals["loss"] += float(loss.detach()) * batch_n
            totals["binary"] += float(binary_loss.detach()) * batch_n
            totals["subtype"] += float(subtype_loss.detach()) * batch_n
            totals["anchor"] += float(refined_anchor.detach()) * batch_n

        outputs = collect_outputs(
            model,
            data["fusion"],
            validation_indices,
            config.eval_batch_size,
            device,
            use_amp,
        )
        calibration = calibrate_threshold_from_outputs(
            outputs, data, config.minimum_bgc_recall
        )
        score = float(calibration["score"])
        scheduler.step(score)
        row = {
            "stage": "teacher",
            "epoch": epoch,
            "loss": totals["loss"] / seen,
            "binary_loss": totals["binary"] / seen,
            "subtype_loss": totals["subtype"] / seen,
            "refined_anchor_loss": totals["anchor"] / seen,
            "validation_threshold": calibration["threshold"],
            "validation_BGC_precision": calibration["BGC_precision"],
            "validation_BGC_recall": calibration["BGC_recall"],
            "validation_score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Teacher {epoch:03d}/{config.teacher_epochs} | "
            f"loss={row['loss']:.5f} | binary={row['binary_loss']:.5f} | "
            f"subtype={row['subtype_loss']:.5f} | val_score={score:.5f} | "
            f"BGC_P={calibration['BGC_precision']:.5f} | "
            f"BGC_R={calibration['BGC_recall']:.5f} | lr={row['lr']:.2e}",
            flush=True,
        )
        if score > best_score + config.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.teacher_patience:
                print(
                    f"Teacher early stopping at epoch {epoch}; best={best_epoch}.",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("FusionTeacher did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, best_score


@torch.inference_mode()
def save_test_bgc_snapshot(
    model: DNAStudent,
    data: Dict[str, Any],
    config: Config,
    device: torch.device,
    use_amp: bool,
    snapshot_dir: Path,
    epoch: int,
    tag: str,
) -> Dict[str, Any]:
    test_indices = data["split_indices"]["test"]
    test_bgc_indices = test_indices[
        data["binary_targets"].index_select(0, test_indices) == 1
    ]
    outputs = collect_outputs(
        model,
        data["dna"],
        test_bgc_indices,
        config.eval_batch_size,
        device,
        use_amp,
    )
    indices = outputs["indices"].tolist()
    path = snapshot_dir / f"{tag}_test_BGC_type_features.pt"
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "tag": tag,
            "BGC_only": True,
            "ids": [data["ids"][index] for index in indices],
            "types": [data["types"][index] for index in indices],
            "subtypes": [data["subtypes"][index] for index in indices],
            "features": outputs["features"],
            "feature_dimension": int(outputs["features"].shape[1]),
            "source_split": "external_test_only",
        },
        path,
    )
    print(
        f"Saved test-BGC snapshot epoch={epoch}: "
        f"N={outputs['features'].shape[0]:,}, path={path}",
        flush=True,
    )
    return {
        "epoch": int(epoch),
        "tag": tag,
        "number_of_test_BGC_sequences": int(outputs["features"].shape[0]),
        "feature_dimension": int(outputs["features"].shape[1]),
        "path": str(path),
    }


def train_student(
    teacher: FusionTeacher,
    data: Dict[str, Any],
    config: Config,
    device: torch.device,
    use_amp: bool,
    snapshot_epochs: Sequence[int],
    snapshot_dir: Path,
) -> Tuple[DNAStudent, List[Dict[str, Any]], int, float, List[Dict[str, Any]]]:
    train_indices = data["split_indices"]["train"]
    validation_indices = data["split_indices"]["validation"]
    dna_train = data["dna"].index_select(0, train_indices)
    dna_mean = dna_train.mean(dim=0)
    dna_std = dna_train.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    train_bgc = data["bgc_targets"].index_select(0, train_indices)
    subtype_weights = mild_class_weights(
        train_bgc[train_bgc >= 0],
        len(data["bgc_names"]),
        config.class_weight_power,
    ).to(device)
    model = DNAStudent(
        dna_mean.to(device),
        dna_std.to(device),
        len(data["bgc_names"]),
        config,
    ).to(device)
    with torch.no_grad():
        model.binary_head.weight.copy_(teacher.binary_head.weight)
        model.bgc_head.weight.copy_(teacher.bgc_head.weight)
    teacher_binary_anchor = teacher.binary_head.normalized_weight().detach()
    teacher_bgc_anchor = teacher.bgc_head.normalized_weight().detach()
    refined_device = data["refined"].to(device)
    nonbgc_centroid_device = data["nonbgc_centroid"].to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.student_lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7, min_lr=1.0e-6
    )
    scaler = make_grad_scaler(use_amp)
    teacher.eval()
    best_score = -math.inf
    best_precision = -math.inf
    best_accuracy = -math.inf
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    snapshots = []
    requested_snapshots = sorted(
        {int(epoch) for epoch in snapshot_epochs if 0 <= int(epoch) <= config.student_epochs}
    )

    if 0 in requested_snapshots:
        snapshots.append(
            save_test_bgc_snapshot(
                model,
                data,
                config,
                device,
                use_amp,
                snapshot_dir,
                epoch=0,
                tag="epoch_000",
            )
        )

    print("\n===== STAGE 2: TRAIN DNA-ONLY BGC STUDENT =====", flush=True)
    for epoch in range(1, config.student_epochs + 1):
        epoch_started = time.time()
        model.train()
        totals = defaultdict(float)
        seen = 0
        for indices in batch_indices(
            train_indices, config.batch_size, shuffle=True, seed=config.seed + 100000 + epoch
        ):
            dna_batch = data["dna"].index_select(0, indices).to(
                device, non_blocking=True
            )
            fusion_batch = data["fusion"].index_select(0, indices).to(
                device, non_blocking=True
            )
            binary_y = data["binary_targets"].index_select(0, indices).to(device)
            bgc_y = data["bgc_targets"].index_select(0, indices).to(device)
            with torch.no_grad():
                with amp_context(device, use_amp):
                    teacher_binary, teacher_bgc, teacher_representation = teacher(
                        fusion_batch
                    )
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                student_binary, student_bgc, student_representation = model(dna_batch)
                binary_loss = F.cross_entropy(
                    student_binary.float(), binary_y, label_smoothing=config.label_smoothing
                )
                bgc_mask = bgc_y >= 0
                if bool(bgc_mask.any()):
                    subtype_loss = F.cross_entropy(
                        student_bgc[bgc_mask].float(),
                        bgc_y[bgc_mask],
                        weight=subtype_weights,
                        label_smoothing=config.label_smoothing,
                    )
                    subtype_kd = kd_loss(
                        student_bgc[bgc_mask], teacher_bgc[bgc_mask], config.temperature
                    )
                else:
                    subtype_loss = binary_loss.new_zeros(())
                    subtype_kd = binary_loss.new_zeros(())
                binary_kd = kd_loss(
                    student_binary, teacher_binary, config.temperature
                )
                feature_loss = cosine_pull(
                    student_representation, teacher_representation
                )
                target_prototypes = torch.empty_like(student_representation)
                if bool(bgc_mask.any()):
                    target_prototypes[bgc_mask] = refined_device.index_select(
                        0, bgc_y[bgc_mask]
                    )
                target_prototypes[~bgc_mask] = nonbgc_centroid_device
                prototype_loss = cosine_pull(student_representation, target_prototypes)
                head_loss = (
                    1.0
                    - (
                        model.binary_head.normalized_weight() * teacher_binary_anchor
                    ).sum(dim=1)
                ).mean() + (
                    1.0
                    - (model.bgc_head.normalized_weight() * teacher_bgc_anchor).sum(dim=1)
                ).mean()
                loss = (
                    config.student_binary_ce * binary_loss
                    + config.student_subtype_ce * subtype_loss
                    + config.student_binary_kd * binary_kd
                    + config.student_subtype_kd * subtype_kd
                    + config.student_feature_align * feature_loss
                    + config.student_prototype_align * prototype_loss
                    + config.student_head_anchor * head_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_n = int(binary_y.numel())
            seen += batch_n
            for key, value in {
                "loss": loss,
                "binary": binary_loss,
                "subtype": subtype_loss,
                "binary_kd": binary_kd,
                "subtype_kd": subtype_kd,
                "feature": feature_loss,
                "prototype": prototype_loss,
                "head": head_loss,
            }.items():
                totals[key] += float(value.detach()) * batch_n

        validation_outputs = collect_outputs(
            model,
            data["dna"],
            validation_indices,
            config.eval_batch_size,
            device,
            use_amp,
        )
        calibration = calibrate_threshold_from_outputs(
            validation_outputs, data, config.minimum_bgc_recall
        )
        validation = calibration["metrics"]
        score = float(calibration["score"])
        scheduler.step(score)
        current_precision = float(
            validation["end_to_end_all_classes"]["weighted_precision"]
        )
        current_accuracy = float(validation["end_to_end_all_classes"]["accuracy"])
        row = {
            "stage": "student",
            "epoch": epoch,
            "loss": totals["loss"] / seen,
            "binary_loss": totals["binary"] / seen,
            "subtype_loss": totals["subtype"] / seen,
            "binary_kd_loss": totals["binary_kd"] / seen,
            "subtype_kd_loss": totals["subtype_kd"] / seen,
            "feature_loss": totals["feature"] / seen,
            "prototype_loss": totals["prototype"] / seen,
            "head_loss": totals["head"] / seen,
            "validation_threshold": calibration["threshold"],
            "validation_BGC_precision": calibration["BGC_precision"],
            "validation_BGC_recall": calibration["BGC_recall"],
            "validation_score": score,
            "validation_end_accuracy": current_accuracy,
            "validation_end_weighted_precision": current_precision,
            "validation_end_weighted_f1": validation["end_to_end_all_classes"][
                "weighted_f1"
            ],
            "validation_end_mcc": validation["end_to_end_all_classes"]["mcc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(
            f"Student {epoch:03d}/{config.student_epochs} | "
            f"loss={row['loss']:.5f} | binary={row['binary_loss']:.5f} | "
            f"subtype={row['subtype_loss']:.5f} | KD={row['subtype_kd_loss']:.5f} | "
            f"val_score={score:.5f} | val_thr={calibration['threshold']:.2f} | "
            f"BGC_P={calibration['BGC_precision']:.5f} | "
            f"BGC_R={calibration['BGC_recall']:.5f} | "
            f"val_ACC={current_accuracy:.5f} | lr={row['lr']:.2e}",
            flush=True,
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
            best_score = score
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

        if epoch in requested_snapshots:
            snapshots.append(
                save_test_bgc_snapshot(
                    model,
                    data,
                    config,
                    device,
                    use_amp,
                    snapshot_dir,
                    epoch=epoch,
                    tag=f"epoch_{epoch:03d}",
                )
            )
        if no_improve >= config.student_patience:
            print(
                f"Student early stopping at epoch {epoch}; best={best_epoch}.",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("DNAStudent did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    snapshots.append(
        save_test_bgc_snapshot(
            model,
            data,
            config,
            device,
            use_amp,
            snapshot_dir,
            epoch=best_epoch,
            tag=f"run_best_epoch_{best_epoch:03d}",
        )
    )
    return model, history, best_epoch, best_score, snapshots


def evaluate_split(
    model: DNAStudent,
    split: str,
    data: Dict[str, Any],
    config: Config,
    device: torch.device,
    use_amp: bool,
    threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    outputs = collect_outputs(
        model,
        data["dna"],
        data["split_indices"][split],
        config.eval_batch_size,
        device,
        use_amp,
    )
    metrics, arrays = predictions_and_metrics(outputs, data, threshold)
    rows = []
    for local_position, global_index in enumerate(arrays["indices"].tolist()):
        true_binary = int(arrays["true_binary"][local_position])
        predicted_binary = int(arrays["predicted_binary"][local_position])
        true_bgc_index = int(arrays["true_bgc"][local_position])
        predicted_bgc_index = int(arrays["predicted_bgc"][local_position])
        true_subtype = NONBGC if true_binary == 0 else data["bgc_names"][true_bgc_index]
        predicted_subtype = (
            NONBGC if predicted_binary == 0 else data["bgc_names"][predicted_bgc_index]
        )
        true_type = NONBGC if true_binary == 0 else data["subtype_to_type"][true_subtype]
        predicted_type = (
            NONBGC
            if predicted_binary == 0
            else data["subtype_to_type"][predicted_subtype]
        )
        rows.append(
            {
                "id": data["ids"][global_index],
                "split": split,
                "true_binary": "BGC" if true_binary == 1 else NONBGC,
                "predicted_binary": "BGC" if predicted_binary == 1 else NONBGC,
                "true_type": true_type,
                "predicted_type": predicted_type,
                "true_subtype": true_subtype,
                "predicted_subtype": predicted_subtype,
                "BGC_probability": float(arrays["BGC_probability"][local_position]),
                "BGC_subtype_max_probability": float(
                    arrays["BGC_subtype_max_probability"][local_position]
                ),
                "binary_correct": int(true_binary == predicted_binary),
                "type_correct": int(true_type == predicted_type),
                "subtype_correct": int(true_subtype == predicted_subtype),
            }
        )
    return metrics, rows


def summary_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_names = {
        "binary": "BGC_nonBGC",
        "BGC_subtype_oracle_binary": "BGC_Subtype",
        "end_to_end_all_classes": "End_to_end",
    }
    return [
        {"task": task_names[key], **metrics[key]}
        for key in (
            "binary",
            "BGC_subtype_oracle_binary",
            "end_to_end_all_classes",
        )
    ]


def save_split_manifest(data: Dict[str, Any], output_dir: Path) -> None:
    rows = []
    for index, seq_id in enumerate(data["ids"]):
        rows.append(
            {
                "id": seq_id,
                "type": data["types"][index],
                "subtype": data["subtypes"][index],
                "external_split": data["external_splits"][index],
                "model_split": data["model_splits"][index],
            }
        )
    write_csv(output_dir / "split_manifest.csv", rows)


def validate_arguments(args: argparse.Namespace, config: Config) -> None:
    for name in ("dna_pt", "fusion_pt", "refined_pt"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    for name in (
        "batch_size",
        "eval_batch_size",
        "teacher_epochs",
        "student_epochs",
        "teacher_patience",
        "student_patience",
    ):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    checkpoint = args.output_dir / "dna_student_BGC_best_model.pt"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output checkpoint already exists: {checkpoint}\n"
            "Use --overwrite only if you intend to replace this run."
        )


def main() -> None:
    args = parse_args()
    args.dna_pt = args.dna_pt.expanduser().resolve()
    args.fusion_pt = args.fusion_pt.expanduser().resolve()
    args.refined_pt = args.refined_pt.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    config = update_config(args)
    validate_arguments(args, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(config.seed)
    started = time.time()

    print("=" * 80, flush=True)
    print("BGCScope FULL-LENGTH DNA-ONLY STUDENT", flush=True)
    print("=" * 80, flush=True)
    print(f"DNA PT : {args.dna_pt}", flush=True)
    print(f"Fusion PT  : {args.fusion_pt}", flush=True)
    print(f"Refined PT : {args.refined_pt}", flush=True)
    print(f"Output     : {args.output_dir}", flush=True)
    print(f"Device     : {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"AMP        : {use_amp}", flush=True)

    data = load_and_align_data(args.dna_pt, args.fusion_pt, args.refined_pt)
    save_split_manifest(data, args.output_dir)

    seed_everything(config.seed)
    teacher, teacher_history, teacher_best_epoch, teacher_best_score = train_teacher(
        data, config, device, use_amp
    )

    seed_everything(config.seed)
    snapshot_dir = args.output_dir / "epoch_features"
    (
        student,
        student_history,
        student_best_epoch,
        student_best_score,
        snapshot_records,
    ) = train_student(
        teacher,
        data,
        config,
        device,
        use_amp,
        args.snapshot_epochs,
        snapshot_dir,
    )

    validation_outputs = collect_outputs(
        student,
        data["dna"],
        data["split_indices"]["validation"],
        config.eval_batch_size,
        device,
        use_amp,
    )
    final_calibration = calibrate_threshold_from_outputs(
        validation_outputs, data, config.minimum_bgc_recall
    )
    binary_threshold = float(final_calibration["threshold"])
    print(
        f"\nValidation-selected BGC threshold: {binary_threshold:.2f} | "
        f"BGC precision={final_calibration['BGC_precision']:.5f} | "
        f"BGC recall={final_calibration['BGC_recall']:.5f}",
        flush=True,
    )

    train_metrics, train_rows = evaluate_split(
        student, "train", data, config, device, use_amp, binary_threshold
    )
    validation_metrics, validation_rows = evaluate_split(
        student, "validation", data, config, device, use_amp, binary_threshold
    )
    test_metrics, test_rows = evaluate_split(
        student, "test", data, config, device, use_amp, binary_threshold
    )

    print("\n===== FINAL TEST RESULTS =====", flush=True)
    for row in summary_rows(test_metrics):
        print(
            f"{row['task']:12s} | n={row['n']:,} | ACC={row['accuracy']:.6f} | "
            f"macro_F1={row['macro_f1']:.6f} | "
            f"weighted_F1={row['weighted_f1']:.6f} | MCC={row['mcc']:.6f}",
            flush=True,
        )

    write_csv(args.output_dir / "training_loss.csv", teacher_history + student_history)
    write_csv(args.output_dir / "train_metrics.csv", summary_rows(train_metrics))
    write_csv(args.output_dir / "validation_metrics.csv", summary_rows(validation_metrics))
    write_csv(args.output_dir / "test_metrics.csv", summary_rows(test_metrics))
    write_csv(args.output_dir / "train_sequence_predictions.csv", train_rows)
    write_csv(args.output_dir / "validation_sequence_predictions.csv", validation_rows)
    write_csv(args.output_dir / "test_sequence_predictions.csv", test_rows)
    write_csv(
        args.output_dir / "validation_binary_threshold_sweep.csv",
        final_calibration["rows"],
    )
    write_csv(args.output_dir / "snapshot_manifest.csv", snapshot_records)

    checkpoint = {
        "model_name": (
            "BGCScope full-length dna-only hierarchical fusion-teacher "
            "refined-prototype Student"
        ),
        "script_version": SCRIPT_VERSION,
        "student_state_dict": {
            key: value.detach().cpu() for key, value in student.state_dict().items()
        },
        "teacher_state_dict": {
            key: value.detach().cpu() for key, value in teacher.state_dict().items()
        },
        "dna_input_dimension": 768,
        "embedding_dimension": config.embedding_dim,
        "bgc_subtype_names": data["bgc_names"],
        "prototype_types": data["prototype_types"],
        "full_class_names": [NONBGC] + data["bgc_names"],
        "number_of_BGC_subtypes": len(data["bgc_names"]),
        "number_of_full_classes": len(data["bgc_names"]) + 1,
        "nonBGC_class_index": 0,
        "dna_train_mean": student.dna_mean.detach().cpu(),
        "dna_train_std": student.dna_std.detach().cpu(),
        "refined_BGC_subtype_prototypes": data["refined"],
        "nonbgc_reference_centroid": data["nonbgc_centroid"],
        "selected_binary_BGC_probability_threshold": binary_threshold,
        "teacher_best_epoch": teacher_best_epoch,
        "teacher_best_validation_score": teacher_best_score,
        "student_best_epoch": student_best_epoch,
        "student_best_validation_score": student_best_score,
        "train_metrics_reporting_only": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "configuration": asdict(config),
        "subtype_to_type": data["subtype_to_type"],
        "type_names": data["type_names"],
        "validation_binary_threshold_sweep": final_calibration["rows"],
        "epoch_representation_snapshots": snapshot_records,
        "source_dna_pt": str(args.dna_pt),
        "source_fusion_pt": str(args.fusion_pt),
        "source_refined_pt": str(args.refined_pt),
        "training_rule": (
            "The split metadata in long_fusion.pt is authoritative; normalization, "
            "class weighting and optimization use model-train only; validation "
            "selects teacher/student epochs and binary threshold; external test is "
            "evaluated only after all selection; inference uses DNABERT-2 dna "
            "embeddings only."
        ),
        "elapsed_seconds": time.time() - started,
    }
    checkpoint_path = args.output_dir / "dna_student_BGC_best_model.pt"
    atomic_torch_save(checkpoint, checkpoint_path)

    summary = {
        "status": "COMPLETED",
        "script_version": SCRIPT_VERSION,
        "teacher_best_epoch": teacher_best_epoch,
        "teacher_best_validation_score": teacher_best_score,
        "student_best_epoch": student_best_epoch,
        "student_best_validation_score": student_best_score,
        "selected_binary_BGC_probability_threshold": binary_threshold,
        "number_of_BGC_subtypes": len(data["bgc_names"]),
        "split_counts": dict(Counter(data["model_splits"])),
        "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80, flush=True)
    print("BGC DNA STUDENT TRAINING COMPLETED", flush=True)
    print("=" * 80, flush=True)
    print(f"Teacher best epoch : {teacher_best_epoch}", flush=True)
    print(f"Student best epoch : {student_best_epoch}", flush=True)
    print(f"BGC threshold       : {binary_threshold:.2f}", flush=True)
    print(f"Model              : {checkpoint_path}", flush=True)
    print(f"Test metrics       : {args.output_dir / 'test_metrics.csv'}", flush=True)
    print(f"Test predictions   : {args.output_dir / 'test_sequence_predictions.csv'}", flush=True)
    print(f"Epoch features     : {snapshot_dir}", flush=True)
    if device.type == "cuda":
        print(
            f"Peak GPU memory    : "
            f"{torch.cuda.max_memory_allocated(0)/(1024**3):.3f} GiB",
            flush=True,
        )
    print(f"Elapsed minutes    : {(time.time()-started)/60.0:.3f}", flush=True)


if __name__ == "__main__":
    main()
