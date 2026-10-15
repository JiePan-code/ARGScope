#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protein + DNA subtype-supervised gated fusion.

输入：
1. ESM2_BGC/BGC_nonBGC_esm2_mean.pt
2. DNABERT2_BGC/BGC_nonBGC_dnabert2_mean.pt
3. Dataset_split/BGC_nonBGC_id_type_subtype_split.csv

数据使用原则：
- external train：来自第一步划分
- validation：仅从 external train 内部划分
- test：完全不参与标准化、训练、early stopping 和模型选择
- 标准化参数仅使用 model-train 计算

输出：
Long_fusion/long_fusion.pt
Long_fusion/model_split_metadata.csv
Long_fusion/run_summary.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# 默认路径
# ============================================================

ROOT = Path(__file__).resolve().parent

DEFAULT_PROTEIN_PT = (
    ROOT
    / "ESM2_BGC"
    / "BGC_nonBGC_esm2_mean.pt"
)

DEFAULT_DNA_PT = (
    ROOT
    / "DNABERT2_BGC"
    / "BGC_nonBGC_dnabert2_mean.pt"
)

DEFAULT_SPLIT_CSV = (
    ROOT
    / "Dataset_split"
    / "BGC_nonBGC_id_type_subtype_split.csv"
)

DEFAULT_OUTPUT_PT = (
    ROOT
    / "Long_fusion"
    / "long_fusion.pt"
)

SCRIPT_VERSION = "2026-09-15-local-bgc-gated-fusion-v1"


# ============================================================
# 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train protein + DNA gated fusion."
    )

    parser.add_argument(
        "--protein-pt",
        type=Path,
        default=DEFAULT_PROTEIN_PT,
    )

    parser.add_argument(
        "--dna-pt",
        type=Path,
        default=DEFAULT_DNA_PT,
    )

    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_SPLIT_CSV,
    )

    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
    )

    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
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
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="从 external train 中划出的 validation 比例。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA mixed precision。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 通用函数
# ============================================================

def canonical_id(value):

    text = str(value).strip()

    return (
        text.split()[0]
        if text
        else ""
    )


def clean_label(value):

    return " ".join(
        str(value).strip().split()
    )


def stable_seed(seed, label):

    content = f"{seed}\0{label}".encode("utf-8")

    digest = hashlib.sha256(
        content
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def calculate_id_sha256(ids):

    digest = hashlib.sha256()

    for seq_id in ids:

        digest.update(
            seq_id.encode("utf-8")
        )

        digest.update(b"\n")

    return digest.hexdigest()


def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_cpu(path):

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


def atomic_torch_save(obj, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    torch.save(
        obj,
        temporary_path,
    )

    if (
        not temporary_path.is_file()
        or temporary_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"临时输出文件无效：{temporary_path}"
        )

    os.replace(
        temporary_path,
        path,
    )


def choose_device(requested_device):

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA，但当前 PyTorch 无法使用 CUDA。"
            )

        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# 读取 embedding
# ============================================================

def load_embedding_pt(
    path,
    expected_dimension,
):

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 embedding 文件：{path}"
        )

    obj = torch_load_cpu(path)

    if not isinstance(obj, dict):
        raise TypeError(
            f"PT 文件必须保存字典：{path}"
        )

    required_keys = {
        "ids",
        "embeddings",
    }

    missing_keys = (
        required_keys
        - set(obj)
    )

    if missing_keys:
        raise KeyError(
            f"{path} 缺少字段：{sorted(missing_keys)}"
        )

    ids = [
        canonical_id(value)
        for value in obj["ids"]
    ]

    embeddings = obj["embeddings"]

    if any(not seq_id for seq_id in ids):
        raise ValueError(
            f"{path} 中存在空 ID。"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"{path} 中存在重复 ID。"
        )

    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
    ):
        raise ValueError(
            f"{path} 的 embeddings 必须是二维 Tensor。"
        )

    expected_shape = (
        len(ids),
        expected_dimension,
    )

    if tuple(embeddings.shape) != expected_shape:
        raise ValueError(
            f"{path} 的 embedding 形状错误："
            f"{tuple(embeddings.shape)}；"
            f"预期为 {expected_shape}。"
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
            f"{path} 中存在 NaN 或 Inf。"
        )

    return ids, embeddings


# ============================================================
# 读取第一步的 train/test 划分
# ============================================================

def load_split_csv(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到划分文件：{path}"
        )

    records = {}
    group_splits = defaultdict(set)

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                f"CSV 没有表头：{path}"
            )

        column_map = {
            column.strip().lower(): column
            for column in reader.fieldnames
        }

        required_columns = {
            "id",
            "type",
            "subtype",
            "group_id",
            "split",
        }

        missing_columns = (
            required_columns
            - set(column_map)
        )

        if missing_columns:
            raise ValueError(
                f"划分表缺少列："
                f"{sorted(missing_columns)}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):

            seq_id = canonical_id(
                row[column_map["id"]]
            )

            type_label = clean_label(
                row[column_map["type"]]
            )

            subtype = clean_label(
                row[column_map["subtype"]]
            )

            group_id = clean_label(
                row[column_map["group_id"]]
            )

            split = clean_label(
                row[column_map["split"]]
            ).lower()

            if not seq_id:
                raise ValueError(
                    f"第 {line_number} 行 ID 为空。"
                )

            if not type_label:
                raise ValueError(
                    f"第 {line_number} 行 type 为空。"
                )

            if not subtype:
                raise ValueError(
                    f"第 {line_number} 行 subtype 为空。"
                )

            if not group_id:
                raise ValueError(
                    f"第 {line_number} 行 group_id 为空。"
                )

            if split not in {
                "train",
                "test",
            }:
                raise ValueError(
                    f"第 {line_number} 行 split 无效："
                    f"{split}"
                )

            if seq_id in records:
                raise ValueError(
                    f"划分表存在重复 ID：{seq_id}"
                )

            records[seq_id] = {
                "type": type_label,
                "subtype": subtype,
                "group_id": group_id,
                "split": split,
            }

            group_splits[group_id].add(split)

    if not records:
        raise ValueError(
            f"划分表中没有记录：{path}"
        )

    leaked_groups = {
        group_id: sorted(splits)
        for group_id, splits in group_splits.items()
        if len(splits) > 1
    }

    if leaked_groups:
        examples = list(leaked_groups.items())[:10]
        raise ValueError(
            "同一个 group_id 出现在多个 external split 中。"
            f"示例：{examples}"
        )

    return records


# ============================================================
# 对齐 protein、DNA 和标签
# ============================================================

def align_inputs(
    protein_ids,
    protein_embeddings,
    dna_ids,
    dna_embeddings,
    split_records,
):

    protein_id_set = set(
        protein_ids
    )

    dna_id_set = set(
        dna_ids
    )

    split_id_set = set(
        split_records
    )

    if protein_id_set != dna_id_set:

        raise ValueError(
            "Protein 与 DNA embedding 的 ID 不一致。\n"
            f"Protein-only："
            f"{len(protein_id_set - dna_id_set)}\n"
            f"DNA-only："
            f"{len(dna_id_set - protein_id_set)}"
        )

    if protein_id_set != split_id_set:

        raise ValueError(
            "Embedding 与划分表的 ID 不一致。\n"
            f"Embedding-only："
            f"{len(protein_id_set - split_id_set)}\n"
            f"Split-only："
            f"{len(split_id_set - protein_id_set)}"
        )

    # 将 DNA embedding 调整到 protein embedding 的 ID 顺序
    dna_index = {
        seq_id: index
        for index, seq_id in enumerate(dna_ids)
    }

    dna_order = torch.tensor(
        [
            dna_index[seq_id]
            for seq_id in protein_ids
        ],
        dtype=torch.long,
    )

    aligned_dna = dna_embeddings.index_select(
        0,
        dna_order,
    ).contiguous()

    types = [
        split_records[seq_id]["type"]
        for seq_id in protein_ids
    ]

    subtypes = [
        split_records[seq_id]["subtype"]
        for seq_id in protein_ids
    ]

    external_splits = [
        split_records[seq_id]["split"]
        for seq_id in protein_ids
    ]

    group_ids = [
        split_records[seq_id]["group_id"]
        for seq_id in protein_ids
    ]

    return (
        protein_ids,
        protein_embeddings,
        aligned_dna,
        types,
        subtypes,
        group_ids,
        external_splits,
    )


# ============================================================
# 从 external train 内部划分 validation
# ============================================================

def derive_internal_validation(
    ids,
    subtypes,
    group_ids,
    external_splits,
    validation_fraction,
    seed,
):

    train_groups_by_subtype = defaultdict(
        lambda: defaultdict(list)
    )

    for index, (
        seq_id,
        subtype,
        group_id,
        split,
    ) in enumerate(
        zip(
            ids,
            subtypes,
            group_ids,
            external_splits,
        )
    ):

        if split == "train":

            train_groups_by_subtype[
                subtype
            ][group_id].append(index)

    model_splits = list(
        external_splits
    )

    validation_indices = set()

    for subtype in sorted(
        train_groups_by_subtype
    ):

        subtype_groups = list(
            train_groups_by_subtype[
                subtype
            ].items()
        )

        # 先按 group_id 排序，避免 CSV 顺序影响结果
        subtype_groups.sort(
            key=lambda item: item[0]
        )

        subtype_group_count = len(
            subtype_groups
        )

        # train 中只有 1–2 个 group 时全部保留用于训练
        if subtype_group_count <= 2:
            continue

        number_of_validation_groups = max(
            1,
            int(
                math.floor(
                    subtype_group_count
                    * validation_fraction
                    + 0.5
                )
            ),
        )

        # 每个 subtype 至少保留 2 个完整 model-train group
        number_of_validation_groups = min(
            number_of_validation_groups,
            subtype_group_count - 2,
        )

        rng = random.Random(
            stable_seed(
                seed,
                subtype,
            )
        )

        rng.shuffle(
            subtype_groups
        )

        for _, group_indices in subtype_groups[
            :number_of_validation_groups
        ]:
            validation_indices.update(group_indices)

    for index in validation_indices:

        if external_splits[index] != "train":
            raise RuntimeError(
                "Validation 中出现了 external test 数据。"
            )

        model_splits[index] = "validation"

    # 检查所有 subtype 是否都保留在 model-train
    all_subtypes = set(
        subtypes
    )

    model_train_subtypes = {
        subtype
        for subtype, split in zip(
            subtypes,
            model_splits,
        )
        if split == "train"
    }

    missing_train_subtypes = (
        all_subtypes
        - model_train_subtypes
    )

    if missing_train_subtypes:

        raise RuntimeError(
            "以下 subtype 没有 model-train 样本："
            f"{sorted(missing_train_subtypes)[:10]}"
        )

    if "validation" not in model_splits:
        raise RuntimeError(
            "没有生成 validation 数据。"
        )

    return model_splits


# ============================================================
# 检查数据划分
# ============================================================

def verify_splits(
    ids,
    types,
    subtypes,
    group_ids,
    external_splits,
    model_splits,
):

    if not (
        len(ids)
        == len(types)
        == len(subtypes)
        == len(group_ids)
        == len(external_splits)
        == len(model_splits)
    ):
        raise RuntimeError(
            "数据字段长度不一致。"
        )

    valid_external = {
        "train",
        "test",
    }

    valid_model = {
        "train",
        "validation",
        "test",
    }

    if (
        set(external_splits)
        - valid_external
    ):
        raise RuntimeError(
            "external split 含有无效值。"
        )

    if (
        set(model_splits)
        - valid_model
    ):
        raise RuntimeError(
            "model split 含有无效值。"
        )

    for external, model in zip(
        external_splits,
        model_splits,
    ):

        if external == "test" and model != "test":
            raise RuntimeError(
                "External test 被用于 train/validation。"
            )

        if external == "train" and model == "test":
            raise RuntimeError(
                "External train 被错误改成 test。"
            )

    all_types = set(types)
    all_subtypes = set(subtypes)

    external_train_types = {
        type_label
        for type_label, split in zip(
            types,
            external_splits,
        )
        if split == "train"
    }

    external_test_types = {
        type_label
        for type_label, split in zip(
            types,
            external_splits,
        )
        if split == "test"
    }

    external_train_subtypes = {
        subtype
        for subtype, split in zip(
            subtypes,
            external_splits,
        )
        if split == "train"
    }

    external_test_subtypes = {
        subtype
        for subtype, split in zip(
            subtypes,
            external_splits,
        )
        if split == "test"
    }

    if external_train_types != all_types:
        raise RuntimeError(
            "External train 没有包含所有 type。"
        )

    if external_test_types != all_types:
        raise RuntimeError(
            "External test 没有包含所有 type。"
        )

    if external_train_subtypes != all_subtypes:
        raise RuntimeError(
            "External train 没有包含所有 subtype。"
        )

    if external_test_subtypes != all_subtypes:
        raise RuntimeError(
            "External test 没有包含所有 subtype。"
        )

    group_model_splits = defaultdict(set)
    group_external_splits = defaultdict(set)

    for group_id, external_split, model_split in zip(
        group_ids,
        external_splits,
        model_splits,
    ):
        group_external_splits[group_id].add(external_split)
        group_model_splits[group_id].add(model_split)

    external_leaks = {
        group_id: sorted(splits)
        for group_id, splits in group_external_splits.items()
        if len(splits) > 1
    }
    model_leaks = {
        group_id: sorted(splits)
        for group_id, splits in group_model_splits.items()
        if len(splits) > 1
    }

    if external_leaks or model_leaks:
        raise RuntimeError(
            "检测到 BGC group 跨 split 泄漏。"
            f"external 示例：{list(external_leaks.items())[:5]}；"
            f"model 示例：{list(model_leaks.items())[:5]}"
        )


# ============================================================
# 标准化
# ============================================================

def fit_standardizer(
    embeddings,
    train_indices,
):

    train_embeddings = (
        embeddings.index_select(
            0,
            train_indices,
        )
    )

    mean = train_embeddings.mean(
        dim=0
    )

    standard_deviation = (
        train_embeddings
        .std(
            dim=0,
            unbiased=False,
        )
        .clamp_min(1e-6)
    )

    if not bool(
        torch.isfinite(mean).all()
    ):
        raise ValueError(
            "标准化均值包含 NaN/Inf。"
        )

    if not bool(
        torch.isfinite(
            standard_deviation
        ).all()
    ):
        raise ValueError(
            "标准化标准差包含 NaN/Inf。"
        )

    return (
        mean,
        standard_deviation,
    )


# ============================================================
# Gated fusion 模型
# ============================================================

class SupervisedGatedFusion(nn.Module):

    def __init__(
        self,
        protein_dimension,
        dna_dimension,
        fusion_dimension,
        number_of_subtypes,
        dropout,
    ):

        super().__init__()

        self.protein_projection = nn.Sequential(
            nn.Linear(
                protein_dimension,
                fusion_dimension,
            ),
            nn.LayerNorm(
                fusion_dimension
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.dna_projection = nn.Sequential(
            nn.Linear(
                dna_dimension,
                fusion_dimension,
            ),
            nn.LayerNorm(
                fusion_dimension
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(
                fusion_dimension * 2,
                fusion_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                fusion_dimension,
                fusion_dimension,
            ),
            nn.Sigmoid(),
        )

        self.fusion_normalization = nn.LayerNorm(
            fusion_dimension
        )

        self.subtype_classifier = nn.Linear(
            fusion_dimension,
            number_of_subtypes,
        )

    def forward(
        self,
        protein,
        dna,
    ):

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

        protein_gate = (
            self.gate_network(
                torch.cat(
                    [
                        protein_projected,
                        dna_projected,
                    ],
                    dim=1,
                )
            )
        )

        fused = (
            self.fusion_normalization(
                protein_gate
                * protein_projected
                + (
                    1.0
                    - protein_gate
                )
                * dna_projected
            )
        )

        logits = (
            self.subtype_classifier(
                fused
            )
        )

        return (
            fused,
            logits,
            protein_gate,
        )


# ============================================================
# Batch 与评价函数
# ============================================================

def iter_batches(
    indices,
    batch_size,
):

    for start in range(
        0,
        indices.numel(),
        batch_size,
    ):

        yield indices[
            start:start + batch_size
        ]


def calculate_class_weights(
    train_targets,
    number_of_classes,
):

    class_counts = torch.bincount(
        train_targets,
        minlength=number_of_classes,
    ).float()

    if bool(
        (class_counts == 0).any()
    ):
        raise ValueError(
            "至少一个 subtype 没有 model-train 样本。"
        )

    # 使用 inverse-square-root，避免极稀有类别权重过大
    weights = class_counts.rsqrt()

    weights = (
        weights
        / weights.mean()
    )

    return weights


def classification_metrics(
    targets,
    predictions,
    number_of_classes,
):

    targets = (
        targets
        .detach()
        .cpu()
        .long()
    )

    predictions = (
        predictions
        .detach()
        .cpu()
        .long()
    )

    accuracy = float(
        (
            targets
            == predictions
        )
        .float()
        .mean()
        .item()
    )

    encoded = (
        targets
        * number_of_classes
        + predictions
    )

    confusion = torch.bincount(
        encoded,
        minlength=(
            number_of_classes
            * number_of_classes
        ),
    ).reshape(
        number_of_classes,
        number_of_classes,
    )

    true_positive = (
        confusion
        .diag()
        .float()
    )

    support = (
        confusion
        .sum(dim=1)
        .float()
    )

    predicted_count = (
        confusion
        .sum(dim=0)
        .float()
    )

    precision = (
        true_positive
        / predicted_count.clamp_min(1.0)
    )

    recall = (
        true_positive
        / support.clamp_min(1.0)
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        ).clamp_min(1e-12)
    )

    present_classes = (
        support > 0
    )

    macro_f1 = float(
        f1[present_classes]
        .mean()
        .item()
    )

    balanced_accuracy = float(
        recall[present_classes]
        .mean()
        .item()
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "number_of_samples": int(
            targets.numel()
        ),
        "number_of_present_classes": int(
            present_classes.sum().item()
        ),
    }


@torch.no_grad()
def evaluate(
    model,
    protein_embeddings,
    dna_embeddings,
    targets,
    indices,
    protein_mean,
    protein_std,
    dna_mean,
    dna_std,
    criterion,
    device,
    batch_size,
    number_of_classes,
    amp_enabled,
):

    model.eval()

    total_loss = 0.0
    total_records = 0

    all_targets = []
    all_predictions = []

    protein_gate_sum = 0.0
    gate_element_count = 0

    for batch_indices in iter_batches(
        indices,
        batch_size,
    ):

        protein_batch = (
            protein_embeddings
            .index_select(
                0,
                batch_indices,
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        dna_batch = (
            dna_embeddings
            .index_select(
                0,
                batch_indices,
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        target_batch = (
            targets
            .index_select(
                0,
                batch_indices,
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        protein_batch = (
            protein_batch
            - protein_mean
        ) / protein_std

        dna_batch = (
            dna_batch
            - dna_mean
        ) / dna_std

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):

            _, logits, protein_gate = model(
                protein_batch,
                dna_batch,
            )

            loss = criterion(
                logits,
                target_batch,
            )

        batch_count = int(
            target_batch.numel()
        )

        total_loss += (
            float(loss.item())
            * batch_count
        )

        total_records += batch_count

        all_targets.append(
            target_batch.cpu()
        )

        all_predictions.append(
            logits.argmax(
                dim=1
            ).cpu()
        )

        protein_gate_sum += float(
            protein_gate
            .float()
            .sum()
            .item()
        )

        gate_element_count += (
            protein_gate.numel()
        )

    result = classification_metrics(
        torch.cat(all_targets),
        torch.cat(all_predictions),
        number_of_classes,
    )

    result["loss"] = (
        total_loss
        / total_records
    )

    result["mean_protein_gate"] = (
        protein_gate_sum
        / gate_element_count
    )

    result["mean_dna_gate"] = (
        1.0
        - result["mean_protein_gate"]
    )

    return result


# ============================================================
# 生成所有序列的 fused embedding
# ============================================================

@torch.no_grad()
def generate_fused_embeddings(
    model,
    protein_embeddings,
    dna_embeddings,
    protein_mean,
    protein_std,
    dna_mean,
    dna_std,
    device,
    batch_size,
    amp_enabled,
):

    model.eval()

    all_indices = torch.arange(
        protein_embeddings.shape[0],
        dtype=torch.long,
    )

    fused_batches = []
    protein_gate_batches = []

    for batch_indices in iter_batches(
        all_indices,
        batch_size,
    ):

        protein_batch = (
            protein_embeddings
            .index_select(
                0,
                batch_indices,
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        dna_batch = (
            dna_embeddings
            .index_select(
                0,
                batch_indices,
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        protein_batch = (
            protein_batch
            - protein_mean
        ) / protein_std

        dna_batch = (
            dna_batch
            - dna_mean
        ) / dna_std

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):

            fused, _, protein_gate = model(
                protein_batch,
                dna_batch,
            )

        fused_batches.append(
            fused.float().cpu()
        )

        # 每条序列保存一个平均 protein gate
        protein_gate_batches.append(
            protein_gate
            .float()
            .mean(dim=1)
            .cpu()
        )

    return (
        torch.cat(
            fused_batches,
            dim=0,
        ),
        torch.cat(
            protein_gate_batches,
            dim=0,
        ),
    )


# ============================================================
# 保存 model split 表
# ============================================================

def save_model_split_csv(
    output_path,
    ids,
    types,
    subtypes,
    group_ids,
    external_splits,
    model_splits,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow([
            "id",
            "type",
            "subtype",
            "group_id",
            "external_split",
            "model_split",
        ])

        for values in zip(
            ids,
            types,
            subtypes,
            group_ids,
            external_splits,
            model_splits,
        ):
            writer.writerow(values)


# ============================================================
# 主程序
# ============================================================

def main():

    args = parse_args()

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

    args.split_csv = (
        args.split_csv
        .expanduser()
        .resolve()
    )

    args.output_pt = (
        args.output_pt
        .expanduser()
        .resolve()
    )

    if (
        args.output_pt.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"输出文件已经存在：{args.output_pt}\n"
            "如需覆盖，请添加 --overwrite。"
        )

    if not (
        0.0
        < args.validation_fraction
        < 1.0
    ):
        raise ValueError(
            "--validation-fraction 必须位于 0 和 1 之间。"
        )

    set_seed(
        args.seed
    )

    start_time = time.time()

    print("=" * 70)
    print("Protein + DNA gated fusion")
    print("=" * 70)

    print(f"Protein PT：{args.protein_pt}")
    print(f"DNA PT：    {args.dna_pt}")
    print(f"Split CSV： {args.split_csv}")
    print(f"Output：    {args.output_pt}")

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    protein_ids, protein_embeddings = (
        load_embedding_pt(
            args.protein_pt,
            expected_dimension=1280,
        )
    )

    dna_ids, dna_embeddings = (
        load_embedding_pt(
            args.dna_pt,
            expected_dimension=768,
        )
    )

    split_records = load_split_csv(
        args.split_csv
    )

    (
        ids,
        protein_embeddings,
        dna_embeddings,
        types,
        subtypes,
        group_ids,
        external_splits,
    ) = align_inputs(
        protein_ids,
        protein_embeddings,
        dna_ids,
        dna_embeddings,
        split_records,
    )

    # --------------------------------------------------------
    # 从 train 中划分 validation
    # --------------------------------------------------------

    model_splits = (
        derive_internal_validation(
            ids,
            subtypes,
            group_ids,
            external_splits,
            validation_fraction=(
                args.validation_fraction
            ),
            seed=args.seed,
        )
    )

    verify_splits(
        ids,
        types,
        subtypes,
        group_ids,
        external_splits,
        model_splits,
    )

    split_counts = Counter(
        model_splits
    )

    print("\n数据统计：")
    print(f"序列数：       {len(ids):,}")
    print(f"Type 数：      {len(set(types)):,}")
    print(f"Subtype 数：   {len(set(subtypes)):,}")

    print(
        "Model split：  "
        f"train={split_counts['train']:,}, "
        f"validation={split_counts['validation']:,}, "
        f"test={split_counts['test']:,}"
    )

    print("数据对齐与防泄漏检查：PASS")

    # --------------------------------------------------------
    # 建立 subtype 标签
    # --------------------------------------------------------

    subtype_names = sorted(
        set(subtypes)
    )

    subtype_to_index = {
        subtype: index
        for index, subtype in enumerate(
            subtype_names
        )
    }

    targets = torch.tensor(
        [
            subtype_to_index[subtype]
            for subtype in subtypes
        ],
        dtype=torch.long,
    )

    split_indices = {
        split: torch.tensor(
            [
                index
                for index, value
                in enumerate(model_splits)
                if value == split
            ],
            dtype=torch.long,
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    # --------------------------------------------------------
    # 仅使用 model-train 计算标准化参数
    # --------------------------------------------------------

    protein_mean, protein_std = (
        fit_standardizer(
            protein_embeddings,
            split_indices["train"],
        )
    )

    dna_mean, dna_std = (
        fit_standardizer(
            dna_embeddings,
            split_indices["train"],
        )
    )

    # --------------------------------------------------------
    # 初始化设备和模型
    # --------------------------------------------------------

    device = choose_device(
        args.device
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    if device.type == "cuda":

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(
            f"\nGPU：{torch.cuda.get_device_name(0)}"
        )

    else:
        print(
            "\n当前使用 CPU，训练速度可能较慢。"
        )

    print(f"AMP：{amp_enabled}")

    model = SupervisedGatedFusion(
        protein_dimension=1280,
        dna_dimension=768,
        fusion_dimension=args.fusion_dim,
        number_of_subtypes=len(
            subtype_names
        ),
        dropout=args.dropout,
    ).to(device)

    protein_mean_device = (
        protein_mean.to(device)
    )

    protein_std_device = (
        protein_std.to(device)
    )

    dna_mean_device = (
        dna_mean.to(device)
    )

    dna_std_device = (
        dna_std.to(device)
    )

    class_weights = (
        calculate_class_weights(
            targets.index_select(
                0,
                split_indices["train"],
            ),
            number_of_classes=len(
                subtype_names
            ),
        )
        .to(device)
    )

    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=(
            args.label_smoothing
        ),
    )

    evaluation_criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
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

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )
    )

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------

    best_model_state = None
    best_epoch = 0
    best_validation_macro_f1 = -math.inf

    epochs_without_improvement = 0
    training_history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            args.seed + epoch
        )

        train_order = (
            split_indices["train"][
                torch.randperm(
                    split_indices[
                        "train"
                    ].numel(),
                    generator=generator,
                )
            ]
        )

        total_train_loss = 0.0
        number_of_train_records = 0

        for batch_indices in iter_batches(
            train_order,
            args.batch_size,
        ):

            protein_batch = (
                protein_embeddings
                .index_select(
                    0,
                    batch_indices,
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            dna_batch = (
                dna_embeddings
                .index_select(
                    0,
                    batch_indices,
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            target_batch = (
                targets
                .index_select(
                    0,
                    batch_indices,
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            protein_batch = (
                protein_batch
                - protein_mean_device
            ) / protein_std_device

            dna_batch = (
                dna_batch
                - dna_mean_device
            ) / dna_std_device

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=amp_enabled
            ):

                _, logits, _ = model(
                    protein_batch,
                    dna_batch,
                )

                loss = train_criterion(
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
                max_norm=5.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_count = int(
                target_batch.numel()
            )

            total_train_loss += (
                float(loss.item())
                * batch_count
            )

            number_of_train_records += (
                batch_count
            )

        train_loss = (
            total_train_loss
            / number_of_train_records
        )

        # Validation 只来自 external train
        validation_metrics = evaluate(
            model=model,
            protein_embeddings=protein_embeddings,
            dna_embeddings=dna_embeddings,
            targets=targets,
            indices=split_indices[
                "validation"
            ],
            protein_mean=(
                protein_mean_device
            ),
            protein_std=(
                protein_std_device
            ),
            dna_mean=dna_mean_device,
            dna_std=dna_std_device,
            criterion=(
                evaluation_criterion
            ),
            device=device,
            batch_size=(
                args.eval_batch_size
            ),
            number_of_classes=len(
                subtype_names
            ),
            amp_enabled=amp_enabled,
        )

        scheduler.step(
            validation_metrics[
                "macro_f1"
            ]
        )

        current_learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": (
                validation_metrics
            ),
            "learning_rate": (
                current_learning_rate
            ),
        })

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss="
            f"{validation_metrics['loss']:.6f} | "
            f"val_acc="
            f"{validation_metrics['accuracy']:.4f} | "
            f"val_macro_f1="
            f"{validation_metrics['macro_f1']:.4f} | "
            f"lr={current_learning_rate:.2e}"
        )

        if (
            validation_metrics[
                "macro_f1"
            ]
            > best_validation_macro_f1
            + 1e-8
        ):

            best_validation_macro_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            best_model_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= args.patience
            ):

                print(
                    f"Early stopping：epoch {epoch}"
                )

                break

    if best_model_state is None:
        raise RuntimeError(
            "训练没有产生有效模型。"
        )

    model.load_state_dict(
        best_model_state
    )

    # --------------------------------------------------------
    # 最终评价
    # test 只在模型选择结束后评价
    # --------------------------------------------------------

    final_metrics = {}

    print("\n最终评价：")

    for split in [
        "train",
        "validation",
        "test",
    ]:

        split_metrics = evaluate(
            model=model,
            protein_embeddings=protein_embeddings,
            dna_embeddings=dna_embeddings,
            targets=targets,
            indices=split_indices[split],
            protein_mean=(
                protein_mean_device
            ),
            protein_std=(
                protein_std_device
            ),
            dna_mean=dna_mean_device,
            dna_std=dna_std_device,
            criterion=(
                evaluation_criterion
            ),
            device=device,
            batch_size=(
                args.eval_batch_size
            ),
            number_of_classes=len(
                subtype_names
            ),
            amp_enabled=amp_enabled,
        )

        final_metrics[split] = (
            split_metrics
        )

        print(
            f"{split:10s} | "
            f"loss={split_metrics['loss']:.6f} | "
            f"accuracy={split_metrics['accuracy']:.4f} | "
            f"macro_f1={split_metrics['macro_f1']:.4f}"
        )

    # --------------------------------------------------------
    # 生成全部序列的 512D fused embedding
    # --------------------------------------------------------

    print(
        "\n正在生成所有序列的 fused embeddings..."
    )

    (
        fused_embeddings,
        protein_gate_mean,
    ) = generate_fused_embeddings(
        model=model,
        protein_embeddings=(
            protein_embeddings
        ),
        dna_embeddings=(
            dna_embeddings
        ),
        protein_mean=(
            protein_mean_device
        ),
        protein_std=(
            protein_std_device
        ),
        dna_mean=dna_mean_device,
        dna_std=dna_std_device,
        device=device,
        batch_size=(
            args.eval_batch_size
        ),
        amp_enabled=amp_enabled,
    )

    if tuple(
        fused_embeddings.shape
    ) != (
        len(ids),
        args.fusion_dim,
    ):
        raise RuntimeError(
            "最终 fused embedding 形状错误。"
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    elapsed_seconds = (
        time.time()
        - start_time
    )

    output = {
        "script_version": (
            SCRIPT_VERSION
        ),

        "ids": ids,

        "embeddings": (
            fused_embeddings
        ),

        "types": types,

        "subtypes": subtypes,

        "group_ids": group_ids,

        # 第一步生成的原始 train/test
        "splits": external_splits,

        # train 内部加入 validation 后的划分
        "model_splits": model_splits,

        "protein_gate_mean": (
            protein_gate_mean
        ),

        "dna_gate_mean": (
            1.0
            - protein_gate_mean
        ),

        "subtype_names": (
            subtype_names
        ),

        "subtype_to_index": (
            subtype_to_index
        ),

        "normalization": {
            "fit_split": (
                "model_train_only"
            ),
            "protein_mean": (
                protein_mean
            ),
            "protein_std": (
                protein_std
            ),
            "dna_mean": dna_mean,
            "dna_std": dna_std,
        },

        "model_state_dict": (
            best_model_state
        ),

        "model_config": {
            "protein_dimension": 1280,
            "dna_dimension": 768,
            "fusion_dimension": (
                args.fusion_dim
            ),
            "number_of_subtypes": (
                len(subtype_names)
            ),
            "dropout": args.dropout,
        },

        "training_config": {
            "seed": args.seed,
            "batch_size": (
                args.batch_size
            ),
            "eval_batch_size": (
                args.eval_batch_size
            ),
            "epochs_requested": (
                args.epochs
            ),
            "epochs_completed": (
                len(training_history)
            ),
            "patience": (
                args.patience
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "label_smoothing": (
                args.label_smoothing
            ),
            "validation_fraction": (
                args.validation_fraction
            ),
            "device": str(device),
            "amp_enabled": (
                amp_enabled
            ),
        },

        "split_counts": dict(
            Counter(model_splits)
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_validation_macro_f1": (
            best_validation_macro_f1
        ),

        "training_history": (
            training_history
        ),

        "final_metrics": (
            final_metrics
        ),

        "input_files": {
            "protein_pt": str(
                args.protein_pt
            ),
            "dna_pt": str(
                args.dna_pt
            ),
            "split_csv": str(
                args.split_csv
            ),
        },

        "id_sha256": (
            calculate_id_sha256(ids)
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    atomic_torch_save(
        output,
        args.output_pt,
    )

    split_output_path = (
        args.output_pt.parent
        / "model_split_metadata.csv"
    )

    save_model_split_csv(
        split_output_path,
        ids,
        types,
        subtypes,
        group_ids,
        external_splits,
        model_splits,
    )

    summary = {
        "status": "COMPLETED",
        "script_version": (
            SCRIPT_VERSION
        ),
        "number_of_sequences": (
            len(ids)
        ),
        "number_of_types": (
            len(set(types))
        ),
        "number_of_subtypes": (
            len(subtype_names)
        ),
        "number_of_groups": (
            len(set(group_ids))
        ),
        "fused_embedding_shape": (
            list(
                fused_embeddings.shape
            )
        ),
        "split_counts": dict(
            Counter(model_splits)
        ),
        "best_epoch": (
            best_epoch
        ),
        "best_validation_macro_f1": (
            best_validation_macro_f1
        ),
        "final_metrics": (
            final_metrics
        ),
        "output_pt": str(
            args.output_pt
        ),
        "id_sha256": (
            calculate_id_sha256(ids)
        ),
        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    summary_path = (
        args.output_pt.parent
        / "run_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("Gated fusion 运行完成")
    print("=" * 70)

    print(
        f"Fused embedding：{args.output_pt}"
    )

    print(
        f"Model split：    {split_output_path}"
    )

    print(
        f"Summary：        {summary_path}"
    )


if __name__ == "__main__":
    main()
