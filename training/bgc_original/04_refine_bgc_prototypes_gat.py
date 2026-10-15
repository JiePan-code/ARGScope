#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 04：使用 Mutual-KNN + Residual GAT 优化 BGC subtype prototypes。

输入：
1. Raw_subtype_prototypes_long/
   raw_BGC_subtype_prototypes_and_baseline.pt
2. Long_fusion/long_fusion.pt

流程：
1. 使用 raw subtype prototypes 的 cosine similarity 建立 Mutual-KNN 图。
2. 分别测试 K = 5, 10, 15, 20。
3. 使用 model-train BGC 序列训练 residual GAT。
4. 使用 validation macro-F1 选择最佳 K 和最佳 epoch。
5. test 只在全部模型选择完成后评价一次。
6. 保存 refined subtype prototypes。

损失：
subtype classification loss
+ anchor_weight × prototype anchor loss
+ type_weight × type auxiliary loss

输出：
Refined_subtype_prototypes_long/
    refined_BGC_subtype_prototypes_residual_GAT.pt
    gat_k_search_summary.csv
    refined_prototype_summary.json
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
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 默认路径
# ============================================================

ROOT = Path(__file__).resolve().parent

DEFAULT_RAW_PT = (
    ROOT
    / "Raw_subtype_prototypes_long"
    / "raw_BGC_subtype_prototypes_and_baseline.pt"
)

DEFAULT_FUSION_PT = (
    ROOT
    / "Long_fusion"
    / "long_fusion.pt"
)

DEFAULT_OUTPUT_PT = (
    ROOT
    / "Refined_subtype_prototypes_long"
    / "refined_BGC_subtype_prototypes_residual_GAT.pt"
)

SCRIPT_VERSION = "2026-09-15-local-bgc-residual-gat-v1"


# ============================================================
# 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Refine BGC subtype prototypes using "
            "Mutual-KNN and residual GAT."
        )
    )

    parser.add_argument(
        "--raw-pt",
        type=Path,
        default=DEFAULT_RAW_PT,
    )

    parser.add_argument(
        "--fusion-pt",
        type=Path,
        default=DEFAULT_FUSION_PT,
    )

    parser.add_argument(
        "--output-pt",
        type=Path,
        default=DEFAULT_OUTPUT_PT,
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
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
        default=0.25,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--initial-residual-scale",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--maximum-residual-scale",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
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


def compact_label(value):

    return "".join(
        character
        for character in str(value).lower()
        if character.isalnum()
    )


def is_nonbgc(value):

    return compact_label(value) in {
        "nonbgc",
        "nonbiosyntheticgenecluster",
        "negative",
    }


def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了 CUDA，但当前 PyTorch 无法使用 CUDA。"
            )

        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


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


def iter_batches(indices, batch_size):

    for start in range(
        0,
        indices.numel(),
        batch_size,
    ):

        yield indices[
            start:start + batch_size
        ]


# ============================================================
# 读取 raw prototypes
# ============================================================

def load_raw_prototypes(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 raw prototype 文件：{path}"
        )

    obj = torch_load_cpu(path)

    if not isinstance(obj, dict):
        raise TypeError(
            "Raw prototype PT 必须保存字典。"
        )

    required_keys = {
        "prototype_subtypes",
        "prototype_types",
        "raw_prototypes",
        "nonbgc_reference_centroid",
    }

    missing_keys = (
        required_keys
        - set(obj)
    )

    if missing_keys:
        raise KeyError(
            "Raw prototype PT 缺少字段："
            f"{sorted(missing_keys)}"
        )

    prototype_subtypes = [
        clean_label(value)
        for value in obj[
            "prototype_subtypes"
        ]
    ]

    prototype_types = [
        clean_label(value)
        for value in obj[
            "prototype_types"
        ]
    ]

    raw_prototypes = obj[
        "raw_prototypes"
    ]

    nonbgc_centroid = obj[
        "nonbgc_reference_centroid"
    ]

    number_of_prototypes = len(
        prototype_subtypes
    )

    if len(prototype_types) != number_of_prototypes:
        raise ValueError(
            "prototype_subtypes 与 prototype_types 长度不一致。"
        )

    if len(set(prototype_subtypes)) != number_of_prototypes:
        raise ValueError(
            "发现重复 prototype subtype。"
        )

    if (
        not isinstance(raw_prototypes, torch.Tensor)
        or raw_prototypes.ndim != 2
    ):
        raise ValueError(
            "raw_prototypes 必须是二维 Tensor。"
        )

    if raw_prototypes.shape[0] != number_of_prototypes:
        raise ValueError(
            "Raw prototype 数量与 subtype 数量不一致。"
        )

    raw_prototypes = (
        raw_prototypes
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    raw_prototypes = F.normalize(
        raw_prototypes,
        p=2,
        dim=1,
        eps=1e-12,
    )

    nonbgc_centroid = (
        nonbgc_centroid
        .detach()
        .cpu()
        .float()
        .reshape(-1)
    )

    nonbgc_centroid = F.normalize(
        nonbgc_centroid,
        p=2,
        dim=0,
        eps=1e-12,
    )

    if raw_prototypes.shape[1] != nonbgc_centroid.numel():
        raise ValueError(
            "Prototype 与 nonBGC centroid 维度不一致。"
        )

    if not bool(
        torch.isfinite(
            raw_prototypes
        ).all()
    ):
        raise ValueError(
            "Raw prototypes 中存在 NaN/Inf。"
        )

    return {
        "source": obj,
        "prototype_subtypes": prototype_subtypes,
        "prototype_types": prototype_types,
        "raw_prototypes": raw_prototypes,
        "nonbgc_centroid": nonbgc_centroid,
    }


# ============================================================
# 读取 fused sequence embeddings
# ============================================================

def load_fusion_data(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 fusion 文件：{path}"
        )

    obj = torch_load_cpu(path)

    if not isinstance(obj, dict):
        raise TypeError(
            "long_fusion.pt 必须保存字典。"
        )

    required_keys = {
        "ids",
        "embeddings",
        "types",
        "subtypes",
        "model_splits",
    }

    missing_keys = (
        required_keys
        - set(obj)
    )

    if missing_keys:
        raise KeyError(
            "long_fusion.pt 缺少字段："
            f"{sorted(missing_keys)}"
        )

    ids = [
        canonical_id(value)
        for value in obj["ids"]
    ]

    types = [
        clean_label(value)
        for value in obj["types"]
    ]

    subtypes = [
        clean_label(value)
        for value in obj["subtypes"]
    ]

    model_splits = [
        clean_label(value).lower()
        for value in obj[
            "model_splits"
        ]
    ]

    external_splits = [
        clean_label(value).lower()
        for value in obj.get(
            "splits",
            model_splits,
        )
    ]

    embeddings = obj["embeddings"]

    number_of_records = len(ids)

    for name, values in {
        "types": types,
        "subtypes": subtypes,
        "model_splits": model_splits,
        "external_splits": external_splits,
    }.items():

        if len(values) != number_of_records:

            raise ValueError(
                f"字段长度错误：ids={number_of_records}，"
                f"{name}={len(values)}"
            )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Fusion 数据中存在重复 ID。"
        )

    if set(model_splits) - {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "model_splits 中存在无效值。"
        )

    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
    ):
        raise ValueError(
            "Fusion embeddings 必须是二维 Tensor。"
        )

    if embeddings.shape[0] != number_of_records:
        raise ValueError(
            "Fusion embedding 数量与 ID 数量不一致。"
        )

    embeddings = (
        embeddings
        .detach()
        .cpu()
        .float()
        .contiguous()
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
        eps=1e-12,
    )

    if not bool(
        torch.isfinite(
            embeddings
        ).all()
    ):
        raise ValueError(
            "Fusion embeddings 中存在 NaN/Inf。"
        )

    return {
        "source": obj,
        "ids": ids,
        "types": types,
        "subtypes": subtypes,
        "model_splits": model_splits,
        "external_splits": external_splits,
        "embeddings": embeddings,
    }


# ============================================================
# 检查 prototype 与 sequence 标签
# ============================================================

def validate_labels(
    prototype_subtypes,
    prototype_types,
    sequence_types,
    sequence_subtypes,
):

    prototype_subtype_to_type = {
        subtype: type_label
        for subtype, type_label in zip(
            prototype_subtypes,
            prototype_types,
        )
    }

    sequence_bgc_subtypes = {
        subtype
        for type_label, subtype in zip(
            sequence_types,
            sequence_subtypes,
        )
        if (
            not is_nonbgc(type_label)
            and not is_nonbgc(subtype)
        )
    }

    prototype_subtype_set = set(
        prototype_subtypes
    )

    if sequence_bgc_subtypes != prototype_subtype_set:

        missing_prototypes = (
            sequence_bgc_subtypes
            - prototype_subtype_set
        )

        extra_prototypes = (
            prototype_subtype_set
            - sequence_bgc_subtypes
        )

        raise ValueError(
            "Sequence subtype 与 prototype subtype 不一致。\n"
            f"缺少 prototype：{sorted(missing_prototypes)[:10]}\n"
            f"多余 prototype：{sorted(extra_prototypes)[:10]}"
        )

    for type_label, subtype in zip(
        sequence_types,
        sequence_subtypes,
    ):

        if (
            is_nonbgc(type_label)
            or is_nonbgc(subtype)
        ):
            continue

        expected_type = (
            prototype_subtype_to_type[
                subtype
            ]
        )

        if type_label != expected_type:
            raise ValueError(
                f"Subtype-to-type 映射不一致："
                f"{subtype} -> "
                f"{expected_type} / {type_label}"
            )


# ============================================================
# Mutual-KNN graph
# ============================================================

def build_mutual_knn_graph(
    raw_prototypes,
    k,
):

    number_of_nodes = (
        raw_prototypes.shape[0]
    )

    if not 1 <= k < number_of_nodes:
        raise ValueError(
            f"K 必须位于 1 到 {number_of_nodes - 1}；"
            f"当前 K={k}"
        )

    # raw prototypes 已经 L2 normalized
    similarity = (
        raw_prototypes
        @ raw_prototypes.T
    )

    similarity = similarity.clone()

    similarity.fill_diagonal_(
        -torch.inf
    )

    nearest_neighbors = (
        similarity.topk(
            k=k,
            dim=1,
            largest=True,
        ).indices
    )

    directed_adjacency = torch.zeros(
        (
            number_of_nodes,
            number_of_nodes,
        ),
        dtype=torch.bool,
    )

    row_indices = (
        torch.arange(
            number_of_nodes
        )
        .unsqueeze(1)
        .expand_as(
            nearest_neighbors
        )
    )

    directed_adjacency[
        row_indices,
        nearest_neighbors,
    ] = True

    # 只有双方互为 KNN 才保留边
    mutual_adjacency = (
        directed_adjacency
        & directed_adjacency.T
    )

    # GAT 必须保留 self-loop
    adjacency = (
        mutual_adjacency
        | torch.eye(
            number_of_nodes,
            dtype=torch.bool,
        )
    )

    degree_without_self = (
        adjacency.sum(dim=1)
        - 1
    )

    number_of_undirected_edges = int(
        (
            adjacency.sum().item()
            - number_of_nodes
        )
        // 2
    )

    graph_statistics = {
        "k": k,
        "number_of_nodes": (
            number_of_nodes
        ),
        "number_of_undirected_edges": (
            number_of_undirected_edges
        ),
        "minimum_degree": int(
            degree_without_self.min().item()
        ),
        "maximum_degree": int(
            degree_without_self.max().item()
        ),
        "mean_degree": float(
            degree_without_self
            .float()
            .mean()
            .item()
        ),
        "isolated_nodes_without_self_loop": int(
            (
                degree_without_self == 0
            ).sum().item()
        ),
    }

    return (
        adjacency,
        graph_statistics,
    )


# ============================================================
# Dense graph attention layer
# ============================================================

class DenseGraphAttentionLayer(nn.Module):

    def __init__(
        self,
        input_dimension,
        output_dimension,
        number_of_heads,
        concatenate_heads,
        dropout,
    ):

        super().__init__()

        self.input_dimension = (
            input_dimension
        )

        self.output_dimension = (
            output_dimension
        )

        self.number_of_heads = (
            number_of_heads
        )

        self.concatenate_heads = (
            concatenate_heads
        )

        self.dropout = dropout

        self.weight = nn.Parameter(
            torch.empty(
                number_of_heads,
                input_dimension,
                output_dimension,
            )
        )

        self.attention_source = nn.Parameter(
            torch.empty(
                number_of_heads,
                output_dimension,
            )
        )

        self.attention_target = nn.Parameter(
            torch.empty(
                number_of_heads,
                output_dimension,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.weight
        )

        nn.init.xavier_uniform_(
            self.attention_source
            .unsqueeze(-1)
        )

        nn.init.xavier_uniform_(
            self.attention_target
            .unsqueeze(-1)
        )

    def forward(
        self,
        node_features,
        adjacency,
    ):

        # [heads, nodes, output_dimension]
        transformed = torch.einsum(
            "nd,hdo->hno",
            node_features,
            self.weight,
        )

        source_scores = (
            transformed
            * self.attention_source[
                :, None, :
            ]
        ).sum(
            dim=-1
        )

        target_scores = (
            transformed
            * self.attention_target[
                :, None, :
            ]
        ).sum(
            dim=-1
        )

        # [heads, source_node, target_node]
        attention_logits = (
            source_scores.unsqueeze(2)
            + target_scores.unsqueeze(1)
        )

        attention_logits = F.leaky_relu(
            attention_logits,
            negative_slope=0.2,
        )

        mask = adjacency.unsqueeze(0)

        attention_logits = (
            attention_logits.masked_fill(
                ~mask,
                torch.finfo(
                    attention_logits.dtype
                ).min,
            )
        )

        attention_weights = F.softmax(
            attention_logits,
            dim=-1,
        )

        attention_weights = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )

        output = torch.einsum(
            "hij,hjo->hio",
            attention_weights,
            transformed,
        )

        if self.concatenate_heads:

            output = (
                output
                .permute(1, 0, 2)
                .contiguous()
                .reshape(
                    node_features.shape[0],
                    self.number_of_heads
                    * self.output_dimension,
                )
            )

        else:

            output = output.mean(
                dim=0
            )

        return output


# ============================================================
# Residual prototype GAT
# ============================================================

class ResidualPrototypeGAT(nn.Module):

    def __init__(
        self,
        prototype_dimension,
        hidden_dimension,
        number_of_heads,
        dropout,
        initial_residual_scale,
        maximum_residual_scale,
    ):

        super().__init__()

        if hidden_dimension % number_of_heads != 0:
            raise ValueError(
                "hidden_dimension 必须能被 number_of_heads 整除。"
            )

        if not (
            0.0
            < initial_residual_scale
            < maximum_residual_scale
        ):
            raise ValueError(
                "initial_residual_scale 必须大于 0，"
                "并且小于 maximum_residual_scale。"
            )

        hidden_per_head = (
            hidden_dimension
            // number_of_heads
        )

        self.gat1 = DenseGraphAttentionLayer(
            input_dimension=(
                prototype_dimension
            ),
            output_dimension=(
                hidden_per_head
            ),
            number_of_heads=(
                number_of_heads
            ),
            concatenate_heads=True,
            dropout=dropout,
        )

        self.gat2 = DenseGraphAttentionLayer(
            input_dimension=(
                hidden_dimension
            ),
            output_dimension=(
                prototype_dimension
            ),
            number_of_heads=(
                number_of_heads
            ),
            concatenate_heads=False,
            dropout=dropout,
        )

        self.hidden_normalization = (
            nn.LayerNorm(
                hidden_dimension
            )
        )

        self.delta_normalization = (
            nn.LayerNorm(
                prototype_dimension
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.maximum_residual_scale = float(
            maximum_residual_scale
        )

        initial_ratio = (
            initial_residual_scale
            / maximum_residual_scale
        )

        initial_logit = math.log(
            initial_ratio
            / (
                1.0
                - initial_ratio
            )
        )

        self.residual_scale_logit = nn.Parameter(
            torch.tensor(
                initial_logit,
                dtype=torch.float32,
            )
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

        hidden = (
            self.hidden_normalization(
                hidden
            )
        )

        hidden = F.gelu(
            hidden
        )

        hidden = self.dropout(
            hidden
        )

        prototype_delta = self.gat2(
            hidden,
            adjacency,
        )

        prototype_delta = (
            self.delta_normalization(
                prototype_delta
            )
        )

        residual_scale = (
            self.maximum_residual_scale
            * torch.sigmoid(
                self.residual_scale_logit
            )
        )

        refined_prototypes = F.normalize(
            raw_prototypes
            + residual_scale
            * prototype_delta,
            p=2,
            dim=1,
            eps=1e-12,
        )

        return (
            refined_prototypes,
            residual_scale,
        )


# ============================================================
# 构建 sequence targets
# ============================================================

def select_bgc_split(
    embeddings,
    types,
    subtypes,
    model_splits,
    target_split,
    subtype_to_index,
):

    selected_indices = []

    target_subtypes = []

    for index, (
        type_label,
        subtype,
        split,
    ) in enumerate(
        zip(
            types,
            subtypes,
            model_splits,
        )
    ):

        if split != target_split:
            continue

        if (
            is_nonbgc(type_label)
            or is_nonbgc(subtype)
        ):
            continue

        if subtype not in subtype_to_index:
            raise RuntimeError(
                f"找不到 subtype prototype：{subtype}"
            )

        selected_indices.append(
            index
        )

        target_subtypes.append(
            subtype_to_index[
                subtype
            ]
        )

    if not selected_indices:
        raise RuntimeError(
            f"{target_split} 中没有 BGC 序列。"
        )

    index_tensor = torch.tensor(
        selected_indices,
        dtype=torch.long,
    )

    target_tensor = torch.tensor(
        target_subtypes,
        dtype=torch.long,
    )

    selected_embeddings = (
        embeddings.index_select(
            0,
            index_tensor,
        )
    )

    return (
        selected_embeddings,
        target_tensor,
    )


# ============================================================
# 类别权重
# ============================================================

def calculate_class_weights(
    targets,
    number_of_classes,
):

    counts = torch.bincount(
        targets,
        minlength=number_of_classes,
    ).float()

    if bool(
        (counts == 0).any()
    ):
        raise ValueError(
            "至少一个 subtype 没有 train 样本。"
        )

    weights = counts.rsqrt()

    return (
        weights
        / weights.mean()
    )


# ============================================================
# 从 subtype logits 聚合 type logits
# ============================================================

def subtype_logits_to_type_logits(
    subtype_logits,
    prototype_type_indices,
    number_of_types,
):

    type_logits = []

    for type_index in range(
        number_of_types
    ):

        subtype_indices = torch.nonzero(
            prototype_type_indices
            == type_index,
            as_tuple=False,
        ).squeeze(1)

        if subtype_indices.numel() == 0:
            raise RuntimeError(
                f"Type index {type_index} 没有 subtype。"
            )

        current_type_logits = (
            torch.logsumexp(
                subtype_logits.index_select(
                    1,
                    subtype_indices,
                ),
                dim=1,
            )
        )

        type_logits.append(
            current_type_logits
        )

    return torch.stack(
        type_logits,
        dim=1,
    )


# ============================================================
# 分类指标
# ============================================================

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

    confusion_matrix = torch.bincount(
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
        confusion_matrix
        .diag()
        .float()
    )

    support = (
        confusion_matrix
        .sum(dim=1)
        .float()
    )

    prediction_count = (
        confusion_matrix
        .sum(dim=0)
        .float()
    )

    precision = (
        true_positive
        / prediction_count.clamp_min(1.0)
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

    weighted_f1 = float(
        (
            f1
            * support
        ).sum().item()
        / support.sum().item()
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "number_of_samples": int(
            targets.numel()
        ),
        "number_of_present_subtypes": int(
            present_classes.sum().item()
        ),
        "confusion_matrix": confusion_matrix,
    }


# ============================================================
# Prototype evaluation
# ============================================================

@torch.no_grad()
def evaluate_prototypes(
    sequence_embeddings,
    sequence_targets,
    prototypes,
    device,
    batch_size,
):

    prototypes_device = (
        prototypes.to(device)
    )

    number_of_classes = (
        prototypes.shape[0]
    )

    all_targets = []
    all_predictions = []
    all_top5 = []
    all_true_similarities = []
    all_margins = []

    indices = torch.arange(
        sequence_embeddings.shape[0],
        dtype=torch.long,
    )

    for batch_indices in iter_batches(
        indices,
        batch_size,
    ):

        embedding_batch = (
            sequence_embeddings
            .index_select(
                0,
                batch_indices,
            )
            .to(device)
        )

        target_batch = (
            sequence_targets
            .index_select(
                0,
                batch_indices,
            )
            .to(device)
        )

        similarity = (
            embedding_batch
            @ prototypes_device.T
        )

        predictions = similarity.argmax(
            dim=1
        )

        true_similarity = (
            similarity
            .gather(
                1,
                target_batch.unsqueeze(1),
            )
            .squeeze(1)
        )

        wrong_similarity = (
            similarity.clone()
        )

        wrong_similarity[
            torch.arange(
                target_batch.numel(),
                device=device,
            ),
            target_batch,
        ] = -torch.inf

        best_wrong_similarity = (
            wrong_similarity.max(
                dim=1
            ).values
        )

        margin = (
            true_similarity
            - best_wrong_similarity
        )

        top_k = min(
            5,
            number_of_classes,
        )

        top_indices = similarity.topk(
            k=top_k,
            dim=1,
        ).indices

        top5_correct = (
            top_indices
            == target_batch.unsqueeze(1)
        ).any(dim=1)

        all_targets.append(
            target_batch.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_top5.append(
            top5_correct.cpu()
        )

        all_true_similarities.append(
            true_similarity.cpu()
        )

        all_margins.append(
            margin.cpu()
        )

    targets = torch.cat(
        all_targets
    )

    predictions = torch.cat(
        all_predictions
    )

    result = classification_metrics(
        targets,
        predictions,
        number_of_classes,
    )

    result["top5_accuracy"] = float(
        torch.cat(
            all_top5
        )
        .float()
        .mean()
        .item()
    )

    result["mean_true_similarity"] = float(
        torch.cat(
            all_true_similarities
        )
        .mean()
        .item()
    )

    result["mean_true_margin"] = float(
        torch.cat(
            all_margins
        )
        .mean()
        .item()
    )

    return result


# ============================================================
# 单个 K 的 GAT 训练
# ============================================================

def train_one_k(
    k,
    raw_prototypes,
    adjacency,
    graph_statistics,
    train_embeddings,
    train_targets,
    validation_embeddings,
    validation_targets,
    prototype_type_indices,
    number_of_types,
    args,
    device,
):

    run_seed = (
        args.seed
        + 1009 * k
    )

    set_seed(
        run_seed
    )

    raw_device = (
        raw_prototypes.to(device)
    )

    adjacency_device = (
        adjacency.to(device)
    )

    train_embeddings_device = (
        train_embeddings.to(device)
    )

    train_targets_device = (
        train_targets.to(device)
    )

    prototype_type_indices_device = (
        prototype_type_indices.to(device)
    )

    train_type_targets = (
        prototype_type_indices_device
        .index_select(
            0,
            train_targets_device,
        )
    )

    subtype_weights = (
        calculate_class_weights(
            train_targets,
            raw_prototypes.shape[0],
        )
        .to(device)
    )

    type_weights = (
        calculate_class_weights(
            train_type_targets.cpu(),
            number_of_types,
        )
        .to(device)
    )

    subtype_criterion = (
        nn.CrossEntropyLoss(
            weight=subtype_weights
        )
    )

    type_criterion = (
        nn.CrossEntropyLoss(
            weight=type_weights
        )
    )

    model = ResidualPrototypeGAT(
        prototype_dimension=(
            raw_prototypes.shape[1]
        ),
        hidden_dimension=(
            args.hidden_dim
        ),
        number_of_heads=(
            args.heads
        ),
        dropout=args.dropout,
        initial_residual_scale=(
            args.initial_residual_scale
        ),
        maximum_residual_scale=(
            args.maximum_residual_scale
        ),
    ).to(device)

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
            patience=6,
            min_lr=1e-6,
        )
    )

    best_state = None
    best_epoch = 0
    best_validation_macro_f1 = -math.inf
    best_validation_accuracy = -math.inf

    epochs_without_improvement = 0
    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            refined_prototypes,
            residual_scale,
        ) = model(
            raw_device,
            adjacency_device,
        )

        subtype_logits = (
            train_embeddings_device
            @ refined_prototypes.T
        ) / args.temperature

        subtype_loss = (
            subtype_criterion(
                subtype_logits,
                train_targets_device,
            )
        )

        type_logits = (
            subtype_logits_to_type_logits(
                subtype_logits,
                prototype_type_indices_device,
                number_of_types,
            )
        )

        type_loss = type_criterion(
            type_logits,
            train_type_targets,
        )

        anchor_loss = (
            1.0
            - (
                refined_prototypes
                * raw_device
            )
            .sum(dim=1)
            .mean()
        )

        total_loss = (
            subtype_loss
            + args.anchor_weight
            * anchor_loss
            + args.type_weight
            * type_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        # --------------------------------------------
        # Validation
        # --------------------------------------------

        model.eval()

        with torch.no_grad():

            (
                validation_prototypes,
                validation_scale,
            ) = model(
                raw_device,
                adjacency_device,
            )

            validation_metrics = (
                evaluate_prototypes(
                    sequence_embeddings=(
                        validation_embeddings
                    ),
                    sequence_targets=(
                        validation_targets
                    ),
                    prototypes=(
                        validation_prototypes.cpu()
                    ),
                    device=device,
                    batch_size=(
                        args.eval_batch_size
                    ),
                )
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

        history.append({
            "epoch": epoch,
            "total_loss": float(
                total_loss.item()
            ),
            "subtype_loss": float(
                subtype_loss.item()
            ),
            "type_loss": float(
                type_loss.item()
            ),
            "anchor_loss": float(
                anchor_loss.item()
            ),
            "residual_scale": float(
                validation_scale.item()
            ),
            "validation_accuracy": (
                validation_metrics[
                    "accuracy"
                ]
            ),
            "validation_macro_f1": (
                validation_metrics[
                    "macro_f1"
                ]
            ),
            "learning_rate": (
                current_learning_rate
            ),
        })

        if (
            epoch == 1
            or epoch % args.log_every == 0
        ):

            print(
                f"K={k:02d} | "
                f"epoch={epoch:03d} | "
                f"loss={total_loss.item():.6f} | "
                f"subtype={subtype_loss.item():.6f} | "
                f"type={type_loss.item():.6f} | "
                f"anchor={anchor_loss.item():.6f} | "
                f"alpha={validation_scale.item():.4f} | "
                f"val_acc="
                f"{validation_metrics['accuracy']:.4f} | "
                f"val_macro_f1="
                f"{validation_metrics['macro_f1']:.4f}"
            )

        current_f1 = (
            validation_metrics[
                "macro_f1"
            ]
        )

        current_accuracy = (
            validation_metrics[
                "accuracy"
            ]
        )

        improved = (
            current_f1
            > best_validation_macro_f1
            + 1e-8
        )

        tied_f1_better_accuracy = (
            abs(
                current_f1
                - best_validation_macro_f1
            )
            <= 1e-8
            and current_accuracy
            > best_validation_accuracy
            + 1e-8
        )

        if (
            improved
            or tied_f1_better_accuracy
        ):

            best_validation_macro_f1 = (
                current_f1
            )

            best_validation_accuracy = (
                current_accuracy
            )

            best_epoch = epoch

            best_state = {
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
                    f"K={k} early stopping："
                    f"epoch {epoch}"
                )

                break

    if best_state is None:
        raise RuntimeError(
            f"K={k} 没有产生有效模型。"
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        (
            best_refined_prototypes,
            best_residual_scale,
        ) = model(
            raw_device,
            adjacency_device,
        )

    best_refined_prototypes = (
        best_refined_prototypes
        .detach()
        .cpu()
    )

    final_validation_metrics = (
        evaluate_prototypes(
            sequence_embeddings=(
                validation_embeddings
            ),
            sequence_targets=(
                validation_targets
            ),
            prototypes=(
                best_refined_prototypes
            ),
            device=device,
            batch_size=(
                args.eval_batch_size
            ),
        )
    )

    return {
        "k": k,
        "graph_statistics": (
            graph_statistics
        ),
        "adjacency": (
            adjacency
        ),
        "model_state_dict": (
            best_state
        ),
        "refined_prototypes": (
            best_refined_prototypes
        ),
        "residual_scale": float(
            best_residual_scale.item()
        ),
        "best_epoch": (
            best_epoch
        ),
        "validation_metrics": (
            final_validation_metrics
        ),
        "history": history,
    }


# ============================================================
# JSON-safe metrics
# ============================================================

def metrics_for_json(metrics):

    output = {}

    for key, value in metrics.items():

        if key == "confusion_matrix":
            continue

        if isinstance(value, float):

            output[key] = round(
                value,
                8,
            )

        else:

            output[key] = value

    return output


def metric_deltas(
    raw_metrics,
    refined_metrics,
):

    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "top5_accuracy",
        "mean_true_similarity",
        "mean_true_margin",
    ]

    return {
        metric: (
            refined_metrics[metric]
            - raw_metrics[metric]
        )
        for metric in metric_names
    }


# ============================================================
# 主程序
# ============================================================

def main():

    args = parse_args()

    args.raw_pt = (
        args.raw_pt
        .expanduser()
        .resolve()
    )

    args.fusion_pt = (
        args.fusion_pt
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
            f"输出已经存在：{args.output_pt}\n"
            "如需覆盖，请添加 --overwrite。"
        )

    if args.temperature <= 0:
        raise ValueError(
            "--temperature 必须大于 0。"
        )

    if args.eval_batch_size <= 0:
        raise ValueError(
            "--eval-batch-size 必须大于 0。"
        )

    k_values = sorted(
        set(args.k_values)
    )

    if not k_values:
        raise ValueError(
            "至少需要一个 K。"
        )

    set_seed(
        args.seed
    )

    started = time.time()

    print("=" * 72)
    print("Step 04：Mutual-KNN + Residual GAT")
    print("=" * 72)

    print(f"Raw PT：    {args.raw_pt}")
    print(f"Fusion PT： {args.fusion_pt}")
    print(f"Output：    {args.output_pt}")
    print(f"K values：  {k_values}")

    raw_data = load_raw_prototypes(
        args.raw_pt
    )

    fusion_data = load_fusion_data(
        args.fusion_pt
    )

    prototype_subtypes = raw_data[
        "prototype_subtypes"
    ]

    prototype_types = raw_data[
        "prototype_types"
    ]

    raw_prototypes = raw_data[
        "raw_prototypes"
    ]

    nonbgc_centroid = raw_data[
        "nonbgc_centroid"
    ]

    ids = fusion_data["ids"]
    types = fusion_data["types"]
    subtypes = fusion_data[
        "subtypes"
    ]

    model_splits = fusion_data[
        "model_splits"
    ]

    embeddings = fusion_data[
        "embeddings"
    ]

    validate_labels(
        prototype_subtypes,
        prototype_types,
        types,
        subtypes,
    )

    if (
        embeddings.shape[1]
        != raw_prototypes.shape[1]
    ):
        raise ValueError(
            "Sequence embedding 与 prototype 维度不一致："
            f"{embeddings.shape[1]} / "
            f"{raw_prototypes.shape[1]}"
        )

    number_of_prototypes = (
        raw_prototypes.shape[0]
    )

    for k in k_values:

        if not 1 <= k < number_of_prototypes:
            raise ValueError(
                f"K={k} 无效；prototype 数量为 "
                f"{number_of_prototypes}。"
            )

    subtype_to_index = {
        subtype: index
        for index, subtype in enumerate(
            prototype_subtypes
        )
    }

    type_names = sorted(
        set(prototype_types)
    )

    type_to_index = {
        type_label: index
        for index, type_label in enumerate(
            type_names
        )
    }

    prototype_type_indices = torch.tensor(
        [
            type_to_index[type_label]
            for type_label in prototype_types
        ],
        dtype=torch.long,
    )

    (
        train_embeddings,
        train_targets,
    ) = select_bgc_split(
        embeddings,
        types,
        subtypes,
        model_splits,
        target_split="train",
        subtype_to_index=(
            subtype_to_index
        ),
    )

    (
        validation_embeddings,
        validation_targets,
    ) = select_bgc_split(
        embeddings,
        types,
        subtypes,
        model_splits,
        target_split="validation",
        subtype_to_index=(
            subtype_to_index
        ),
    )

    (
        test_embeddings,
        test_targets,
    ) = select_bgc_split(
        embeddings,
        types,
        subtypes,
        model_splits,
        target_split="test",
        subtype_to_index=(
            subtype_to_index
        ),
    )

    print("\n数据统计：")
    print(
        f"BGC prototypes：   {number_of_prototypes:,}"
    )
    print(
        f"Prototype shape： {tuple(raw_prototypes.shape)}"
    )
    print(
        f"BGC types：        {len(type_names):,}"
    )
    print(
        f"Train BGC：        {train_embeddings.shape[0]:,}"
    )
    print(
        f"Validation BGC：   {validation_embeddings.shape[0]:,}"
    )
    print(
        f"Test BGC：         {test_embeddings.shape[0]:,}"
    )

    device = choose_device(
        args.device
    )

    if device.type == "cuda":

        print(
            f"Device：          "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print("Device：          CPU")

    # --------------------------------------------------------
    # Validation-only K selection
    # --------------------------------------------------------

    k_results = []

    for k in k_values:

        print("\n" + "-" * 72)
        print(f"Training K={k}")
        print("-" * 72)

        (
            adjacency,
            graph_statistics,
        ) = build_mutual_knn_graph(
            raw_prototypes,
            k,
        )

        print(
            f"Mutual edges："
            f"{graph_statistics['number_of_undirected_edges']:,}"
        )

        print(
            f"Mean degree："
            f"{graph_statistics['mean_degree']:.2f}"
        )

        print(
            f"Isolated nodes："
            f"{graph_statistics['isolated_nodes_without_self_loop']:,}"
        )

        result = train_one_k(
            k=k,
            raw_prototypes=(
                raw_prototypes
            ),
            adjacency=adjacency,
            graph_statistics=(
                graph_statistics
            ),
            train_embeddings=(
                train_embeddings
            ),
            train_targets=(
                train_targets
            ),
            validation_embeddings=(
                validation_embeddings
            ),
            validation_targets=(
                validation_targets
            ),
            prototype_type_indices=(
                prototype_type_indices
            ),
            number_of_types=len(
                type_names
            ),
            args=args,
            device=device,
        )

        k_results.append(
            result
        )

        print(
            f"K={k} best epoch="
            f"{result['best_epoch']} | "
            f"validation accuracy="
            f"{result['validation_metrics']['accuracy']:.4f} | "
            f"validation macro-F1="
            f"{result['validation_metrics']['macro_f1']:.4f}"
        )

    # 优先 macro-F1，其次 accuracy，最后偏向较小 K
    selected_result = max(
        k_results,
        key=lambda result: (
            result[
                "validation_metrics"
            ]["macro_f1"],
            result[
                "validation_metrics"
            ]["accuracy"],
            -result["k"],
        ),
    )

    selected_k = selected_result[
        "k"
    ]

    refined_prototypes = selected_result[
        "refined_prototypes"
    ]

    selected_adjacency = selected_result[
        "adjacency"
    ]

    print("\n" + "=" * 72)
    print(
        f"Validation 选择的最佳 K：{selected_k}"
    )
    print(
        f"最佳 epoch：{selected_result['best_epoch']}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # K 选择结束后才评价 test
    # --------------------------------------------------------

    raw_validation_metrics = (
        evaluate_prototypes(
            validation_embeddings,
            validation_targets,
            raw_prototypes,
            device,
            args.eval_batch_size,
        )
    )

    refined_validation_metrics = (
        evaluate_prototypes(
            validation_embeddings,
            validation_targets,
            refined_prototypes,
            device,
            args.eval_batch_size,
        )
    )

    raw_test_metrics = (
        evaluate_prototypes(
            test_embeddings,
            test_targets,
            raw_prototypes,
            device,
            args.eval_batch_size,
        )
    )

    refined_test_metrics = (
        evaluate_prototypes(
            test_embeddings,
            test_targets,
            refined_prototypes,
            device,
            args.eval_batch_size,
        )
    )

    validation_deltas = metric_deltas(
        raw_validation_metrics,
        refined_validation_metrics,
    )

    test_deltas = metric_deltas(
        raw_test_metrics,
        refined_test_metrics,
    )

    print("\nValidation：")
    print(
        f"Raw macro-F1：     "
        f"{raw_validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Refined macro-F1： "
        f"{refined_validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Delta：            "
        f"{validation_deltas['macro_f1']:+.4f}"
    )

    print("\nTest：")
    print(
        f"Raw accuracy：     "
        f"{raw_test_metrics['accuracy']:.4f}"
    )
    print(
        f"Refined accuracy： "
        f"{refined_test_metrics['accuracy']:.4f}"
    )
    print(
        f"Accuracy delta：   "
        f"{test_deltas['accuracy']:+.4f}"
    )
    print(
        f"Raw macro-F1：     "
        f"{raw_test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Refined macro-F1： "
        f"{refined_test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Macro-F1 delta：   "
        f"{test_deltas['macro_f1']:+.4f}"
    )

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    edge_index = (
        selected_adjacency
        .nonzero(
            as_tuple=False
        )
        .T
        .contiguous()
    )

    elapsed_seconds = (
        time.time()
        - started
    )

    k_search_summary = []

    for result in k_results:

        k_search_summary.append({
            "k": result["k"],
            "best_epoch": (
                result["best_epoch"]
            ),
            "residual_scale": (
                result["residual_scale"]
            ),
            "graph_statistics": (
                result[
                    "graph_statistics"
                ]
            ),
            "validation_metrics": (
                metrics_for_json(
                    result[
                        "validation_metrics"
                    ]
                )
            ),
        })

    output = {
        "script_version": (
            SCRIPT_VERSION
        ),

        "prototype_subtypes": (
            prototype_subtypes
        ),

        "prototype_types": (
            prototype_types
        ),

        "type_names": type_names,

        "prototype_type_indices": (
            prototype_type_indices
        ),

        "raw_prototypes": (
            raw_prototypes
        ),

        "refined_prototypes": (
            refined_prototypes
        ),

        "nonbgc_reference_centroid": (
            nonbgc_centroid
        ),

        "selected_k": selected_k,

        "selected_best_epoch": (
            selected_result[
                "best_epoch"
            ]
        ),

        "selected_residual_scale": (
            selected_result[
                "residual_scale"
            ]
        ),

        "selected_adjacency": (
            selected_adjacency
        ),

        "selected_edge_index": (
            edge_index
        ),

        "selected_graph_statistics": (
            selected_result[
                "graph_statistics"
            ]
        ),

        "model_state_dict": (
            selected_result[
                "model_state_dict"
            ]
        ),

        "model_config": {
            "prototype_dimension": int(
                raw_prototypes.shape[1]
            ),
            "hidden_dimension": (
                args.hidden_dim
            ),
            "number_of_heads": (
                args.heads
            ),
            "dropout": args.dropout,
            "initial_residual_scale": (
                args.initial_residual_scale
            ),
            "maximum_residual_scale": (
                args.maximum_residual_scale
            ),
        },

        "training_config": {
            "k_values": k_values,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "temperature": (
                args.temperature
            ),
            "anchor_weight": (
                args.anchor_weight
            ),
            "type_weight": (
                args.type_weight
            ),
            "seed": args.seed,
            "selection_metric": (
                "validation_macro_f1"
            ),
            "test_used_for_selection": False,
        },

        "k_search_summary": (
            k_search_summary
        ),

        "selected_training_history": (
            selected_result[
                "history"
            ]
        ),

        "raw_validation_metrics": (
            raw_validation_metrics
        ),

        "refined_validation_metrics": (
            refined_validation_metrics
        ),

        "validation_metric_deltas": (
            validation_deltas
        ),

        "raw_test_metrics": (
            raw_test_metrics
        ),

        "refined_test_metrics": (
            refined_test_metrics
        ),

        "test_metric_deltas": (
            test_deltas
        ),

        "source_raw_pt": str(
            args.raw_pt
        ),

        "source_fusion_pt": str(
            args.fusion_pt
        ),

        "source_id_sha256": (
            fusion_data[
                "source"
            ].get(
                "id_sha256"
            )
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    atomic_torch_save(
        output,
        args.output_pt,
    )

    # --------------------------------------------------------
    # 保存 K 搜索 CSV
    # --------------------------------------------------------

    k_csv_path = (
        args.output_pt.parent
        / "gat_k_search_summary.csv"
    )

    with k_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow([
            "k",
            "best_epoch",
            "residual_scale",
            "undirected_edges",
            "mean_degree",
            "isolated_nodes",
            "validation_accuracy",
            "validation_macro_f1",
            "selected",
        ])

        for result in k_results:

            graph = result[
                "graph_statistics"
            ]

            metrics = result[
                "validation_metrics"
            ]

            writer.writerow([
                result["k"],
                result["best_epoch"],
                result["residual_scale"],
                graph[
                    "number_of_undirected_edges"
                ],
                graph["mean_degree"],
                graph[
                    "isolated_nodes_without_self_loop"
                ],
                metrics["accuracy"],
                metrics["macro_f1"],
                (
                    "yes"
                    if result["k"] == selected_k
                    else "no"
                ),
            ])

    # --------------------------------------------------------
    # 保存 JSON summary
    # --------------------------------------------------------

    summary = {
        "status": "COMPLETED",
        "script_version": (
            SCRIPT_VERSION
        ),
        "selected_k": selected_k,
        "selected_best_epoch": (
            selected_result[
                "best_epoch"
            ]
        ),
        "selected_residual_scale": (
            selected_result[
                "residual_scale"
            ]
        ),
        "number_of_bgc_prototypes": (
            number_of_prototypes
        ),
        "number_of_types": (
            len(type_names)
        ),
        "raw_validation_metrics": (
            metrics_for_json(
                raw_validation_metrics
            )
        ),
        "refined_validation_metrics": (
            metrics_for_json(
                refined_validation_metrics
            )
        ),
        "validation_metric_deltas": (
            validation_deltas
        ),
        "raw_test_metrics": (
            metrics_for_json(
                raw_test_metrics
            )
        ),
        "refined_test_metrics": (
            metrics_for_json(
                refined_test_metrics
            )
        ),
        "test_metric_deltas": (
            test_deltas
        ),
        "output_pt": str(
            args.output_pt
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            3,
        ),
    }

    summary_path = (
        args.output_pt.parent
        / "refined_prototype_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Step 04 完成")
    print("=" * 72)

    print(
        f"Refined PT：{args.output_pt}"
    )

    print(
        f"K summary：{k_csv_path}"
    )

    print(
        f"Summary：   {summary_path}"
    )


if __name__ == "__main__":
    main()
