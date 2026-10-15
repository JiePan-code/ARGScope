#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 03：构建 train-only BGC subtype raw prototypes。

输入：
Long_fusion/long_fusion.pt

处理：
1. 对每条 512D fused embedding 做 L2 normalization。
2. 只使用 model_splits == "train" 的 BGC 序列。
3. 按 subtype 计算平均 embedding。
4. 对 subtype 均值再次做 L2 normalization。
5. 单独构建 train-only nonBGC reference centroid。
6. 在 validation/test BGC 数据上进行 nearest-prototype baseline。

输出：
Raw_subtype_prototypes_long/
    raw_BGC_subtype_prototypes_and_baseline.pt
    prototype_train_counts.csv
    raw_prototype_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# 默认路径
# ============================================================

ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_PT = (
    ROOT
    / "Long_fusion"
    / "long_fusion.pt"
)

DEFAULT_OUTPUT_PT = (
    ROOT
    / "Raw_subtype_prototypes_long"
    / "raw_BGC_subtype_prototypes_and_baseline.pt"
)

SCRIPT_VERSION = "2026-09-15-local-bgc-raw-prototypes-v1"


# ============================================================
# 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build train-only BGC subtype prototypes "
            "from fused protein-DNA embeddings."
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
        default=4096,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 工具函数
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


# ============================================================
# 读取并检查 long_fusion.pt
# ============================================================

def load_and_validate_input(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"找不到输入文件：{path}"
        )

    obj = torch_load_cpu(path)

    if not isinstance(obj, dict):
        raise TypeError(
            "long_fusion.pt 必须保存一个字典。"
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
        for value in obj["model_splits"]
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
                f"字段长度不一致："
                f"ids={number_of_records}，"
                f"{name}={len(values)}"
            )

    if any(not seq_id for seq_id in ids):
        raise ValueError(
            "存在空 ID。"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "存在重复 ID。"
        )

    if any(not value for value in types):
        raise ValueError(
            "存在空 type。"
        )

    if any(not value for value in subtypes):
        raise ValueError(
            "存在空 subtype。"
        )

    valid_model_splits = {
        "train",
        "validation",
        "test",
    }

    invalid_model_splits = (
        set(model_splits)
        - valid_model_splits
    )

    if invalid_model_splits:
        raise ValueError(
            "model_splits 含有无效值："
            f"{sorted(invalid_model_splits)}"
        )

    if (
        not isinstance(
            embeddings,
            torch.Tensor,
        )
        or embeddings.ndim != 2
    ):
        raise ValueError(
            "embeddings 必须是二维 Tensor。"
        )

    if embeddings.shape[0] != number_of_records:
        raise ValueError(
            "embedding 数量与 ID 数量不一致。"
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
            "Fused embeddings 中存在 NaN 或 Inf。"
        )

    # 检查 nonBGC 的 type 和 subtype 标记是否一致
    inconsistent_nonbgc = []

    for seq_id, type_label, subtype in zip(
        ids,
        types,
        subtypes,
    ):

        type_is_nonbgc = is_nonbgc(
            type_label
        )

        subtype_is_nonbgc = is_nonbgc(
            subtype
        )

        if type_is_nonbgc != subtype_is_nonbgc:

            inconsistent_nonbgc.append({
                "id": seq_id,
                "type": type_label,
                "subtype": subtype,
            })

    if inconsistent_nonbgc:
        raise ValueError(
            "发现 type/subtype 的 nonBGC 标记不一致；"
            f"示例：{inconsistent_nonbgc[:10]}"
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
# 检查 subtype 到 type 的映射
# ============================================================

def build_subtype_to_type(
    types,
    subtypes,
):

    subtype_to_types = defaultdict(set)

    for type_label, subtype in zip(
        types,
        subtypes,
    ):

        if is_nonbgc(subtype):
            continue

        subtype_to_types[
            subtype
        ].add(type_label)

    ambiguous_subtypes = {
        subtype: sorted(type_labels)
        for subtype, type_labels
        in subtype_to_types.items()
        if len(type_labels) != 1
    }

    if ambiguous_subtypes:

        examples = list(
            ambiguous_subtypes.items()
        )[:10]

        raise ValueError(
            "发现同一个 BGC subtype 对应多个 type；"
            f"示例：{examples}"
        )

    return {
        subtype: next(
            iter(type_labels)
        )
        for subtype, type_labels
        in subtype_to_types.items()
    }


# ============================================================
# 构建 train-only raw prototypes
# ============================================================

def build_raw_prototypes(
    normalized_embeddings,
    types,
    subtypes,
    model_splits,
):

    subtype_to_type = build_subtype_to_type(
        types,
        subtypes,
    )

    all_bgc_subtypes = sorted(
        subtype_to_type
    )

    train_indices_by_subtype = defaultdict(
        list
    )

    nonbgc_train_indices = []

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

        if split != "train":
            continue

        if (
            is_nonbgc(type_label)
            and is_nonbgc(subtype)
        ):

            nonbgc_train_indices.append(
                index
            )

        else:

            train_indices_by_subtype[
                subtype
            ].append(index)

    missing_train_subtypes = [
        subtype
        for subtype in all_bgc_subtypes
        if not train_indices_by_subtype[
            subtype
        ]
    ]

    if missing_train_subtypes:

        raise RuntimeError(
            "以下 BGC subtype 没有 model-train 样本："
            f"{missing_train_subtypes[:10]}"
        )

    if not nonbgc_train_indices:
        raise RuntimeError(
            "model-train 中没有 nonBGC 样本。"
        )

    raw_prototypes = []
    prototype_train_counts = []

    for subtype in all_bgc_subtypes:

        subtype_indices = torch.tensor(
            train_indices_by_subtype[
                subtype
            ],
            dtype=torch.long,
        )

        subtype_embeddings = (
            normalized_embeddings
            .index_select(
                0,
                subtype_indices,
            )
        )

        prototype = (
            subtype_embeddings
            .mean(
                dim=0,
                keepdim=True,
            )
        )

        prototype = F.normalize(
            prototype,
            p=2,
            dim=1,
            eps=1e-12,
        ).squeeze(0)

        raw_prototypes.append(
            prototype
        )

        prototype_train_counts.append(
            len(
                train_indices_by_subtype[
                    subtype
                ]
            )
        )

    raw_prototypes = torch.stack(
        raw_prototypes,
        dim=0,
    )

    # 单独构建 nonBGC reference centroid
    nonbgc_indices_tensor = torch.tensor(
        nonbgc_train_indices,
        dtype=torch.long,
    )

    nonbgc_reference_centroid = (
        normalized_embeddings
        .index_select(
            0,
            nonbgc_indices_tensor,
        )
        .mean(
            dim=0,
            keepdim=True,
        )
    )

    nonbgc_reference_centroid = (
        F.normalize(
            nonbgc_reference_centroid,
            p=2,
            dim=1,
            eps=1e-12,
        )
        .squeeze(0)
    )

    prototype_types = [
        subtype_to_type[subtype]
        for subtype in all_bgc_subtypes
    ]

    return {
        "prototype_subtypes": (
            all_bgc_subtypes
        ),
        "prototype_types": (
            prototype_types
        ),
        "raw_prototypes": (
            raw_prototypes
        ),
        "prototype_train_counts": (
            prototype_train_counts
        ),
        "nonbgc_reference_centroid": (
            nonbgc_reference_centroid
        ),
        "nonbgc_train_count": (
            len(nonbgc_train_indices)
        ),
        "subtype_to_type": (
            subtype_to_type
        ),
    }


# ============================================================
# 最近 prototype 分类指标
# ============================================================

def calculate_metrics(
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

    predicted_count = (
        confusion_matrix
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
        "confusion_matrix": (
            confusion_matrix
        ),
    }


# ============================================================
# Validation/test nearest-prototype baseline
# ============================================================

@torch.no_grad()
def evaluate_split(
    split_name,
    ids,
    normalized_embeddings,
    types,
    subtypes,
    model_splits,
    prototype_subtypes,
    raw_prototypes,
    device,
    batch_size,
):

    subtype_to_index = {
        subtype: index
        for index, subtype in enumerate(
            prototype_subtypes
        )
    }

    selected_indices = []

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

        # 这里只评价 BGC subtype 分类
        if split != split_name:
            continue

        if (
            is_nonbgc(type_label)
            or is_nonbgc(subtype)
        ):
            continue

        if subtype not in subtype_to_index:
            raise RuntimeError(
                f"真实 subtype 没有对应 prototype："
                f"{subtype}"
            )

        selected_indices.append(
            index
        )

    if not selected_indices:
        raise RuntimeError(
            f"{split_name} 中没有 BGC 样本。"
        )

    selected_indices_tensor = torch.tensor(
        selected_indices,
        dtype=torch.long,
    )

    prototypes_device = (
        raw_prototypes
        .to(device)
    )

    all_targets = []
    all_predictions = []
    all_top5_correct = []
    all_true_similarities = []
    all_best_similarities = []
    all_margins = []

    prediction_ids = []
    true_subtype_names = []
    predicted_subtype_names = []

    for batch_indices in iter_batches(
        selected_indices_tensor,
        batch_size,
    ):

        embedding_batch = (
            normalized_embeddings
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

        batch_subtypes = [
            subtypes[index]
            for index in batch_indices.tolist()
        ]

        targets = torch.tensor(
            [
                subtype_to_index[subtype]
                for subtype in batch_subtypes
            ],
            dtype=torch.long,
            device=device,
        )

        true_similarity = (
            similarity
            .gather(
                1,
                targets.unsqueeze(1),
            )
            .squeeze(1)
        )

        best_similarity = (
            similarity.max(
                dim=1
            ).values
        )

        incorrect_similarity = (
            similarity.clone()
        )

        incorrect_similarity[
            torch.arange(
                targets.numel(),
                device=device,
            ),
            targets,
        ] = -torch.inf

        best_incorrect_similarity = (
            incorrect_similarity.max(
                dim=1
            ).values
        )

        margin = (
            true_similarity
            - best_incorrect_similarity
        )

        top_k = min(
            5,
            len(prototype_subtypes),
        )

        top_indices = (
            similarity.topk(
                k=top_k,
                dim=1,
            ).indices
        )

        top5_correct = (
            top_indices
            == targets.unsqueeze(1)
        ).any(dim=1)

        all_targets.append(
            targets.cpu()
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_top5_correct.append(
            top5_correct.cpu()
        )

        all_true_similarities.append(
            true_similarity.cpu()
        )

        all_best_similarities.append(
            best_similarity.cpu()
        )

        all_margins.append(
            margin.cpu()
        )

        batch_index_list = (
            batch_indices.tolist()
        )

        prediction_ids.extend(
            ids[index]
            for index in batch_index_list
        )

        true_subtype_names.extend(
            batch_subtypes
        )

        predicted_subtype_names.extend(
            prototype_subtypes[index]
            for index in predictions.cpu().tolist()
        )

    targets = torch.cat(
        all_targets
    )

    predictions = torch.cat(
        all_predictions
    )

    metrics = calculate_metrics(
        targets,
        predictions,
        number_of_classes=len(
            prototype_subtypes
        ),
    )

    metrics["top5_accuracy"] = float(
        torch.cat(
            all_top5_correct
        )
        .float()
        .mean()
        .item()
    )

    metrics["mean_true_similarity"] = float(
        torch.cat(
            all_true_similarities
        )
        .mean()
        .item()
    )

    metrics["mean_best_similarity"] = float(
        torch.cat(
            all_best_similarities
        )
        .mean()
        .item()
    )

    metrics["mean_true_margin"] = float(
        torch.cat(
            all_margins
        )
        .mean()
        .item()
    )

    predictions_output = {
        "ids": prediction_ids,
        "true_subtypes": (
            true_subtype_names
        ),
        "predicted_subtypes": (
            predicted_subtype_names
        ),
    }

    return (
        metrics,
        predictions_output,
    )


# ============================================================
# 保存 prototype 统计表
# ============================================================

def save_prototype_counts(
    path,
    prototype_subtypes,
    prototype_types,
    prototype_train_counts,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow([
            "prototype_index",
            "type",
            "subtype",
            "model_train_count",
        ])

        for index, (
            type_label,
            subtype,
            train_count,
        ) in enumerate(
            zip(
                prototype_types,
                prototype_subtypes,
                prototype_train_counts,
            )
        ):

            writer.writerow([
                index,
                type_label,
                subtype,
                train_count,
            ])


# ============================================================
# JSON 指标
# ============================================================

def metrics_for_json(metrics):

    result = {}

    for key, value in metrics.items():

        if key == "confusion_matrix":
            continue

        if isinstance(value, float):
            result[key] = round(
                value,
                8,
            )
        else:
            result[key] = value

    return result


# ============================================================
# 主程序
# ============================================================

def main():

    args = parse_args()

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

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size 必须大于 0。"
        )

    if (
        args.output_pt.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"输出文件已经存在：{args.output_pt}\n"
            "如果需要覆盖，请添加 --overwrite。"
        )

    started = time.time()

    print("=" * 70)
    print("Step 03：Train-only raw BGC subtype prototypes")
    print("=" * 70)

    print(f"输入：{args.input_pt}")
    print(f"输出：{args.output_pt}")

    # --------------------------------------------------------
    # 读取 fused embeddings
    # --------------------------------------------------------

    data = load_and_validate_input(
        args.input_pt
    )

    ids = data["ids"]
    types = data["types"]
    subtypes = data["subtypes"]

    model_splits = data[
        "model_splits"
    ]

    external_splits = data[
        "external_splits"
    ]

    embeddings = data[
        "embeddings"
    ]

    print("\n输入数据：")
    print(f"序列数：     {len(ids):,}")
    print(
        f"Embedding：  {tuple(embeddings.shape)}"
    )
    print(
        f"Type 数：    {len(set(types)):,}"
    )
    print(
        f"Subtype 数： {len(set(subtypes)):,}"
    )
    print(
        f"Model split："
        f"{dict(Counter(model_splits))}"
    )

    # --------------------------------------------------------
    # 每条序列先进行 L2 normalization
    # --------------------------------------------------------

    normalized_embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
        eps=1e-12,
    )

    embedding_norms = (
        normalized_embeddings
        .norm(
            p=2,
            dim=1,
        )
    )

    if not bool(
        torch.allclose(
            embedding_norms,
            torch.ones_like(
                embedding_norms
            ),
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise RuntimeError(
            "序列 embedding L2 normalization 失败。"
        )

    # --------------------------------------------------------
    # 构建 train-only prototypes
    # --------------------------------------------------------

    prototype_data = (
        build_raw_prototypes(
            normalized_embeddings=(
                normalized_embeddings
            ),
            types=types,
            subtypes=subtypes,
            model_splits=model_splits,
        )
    )

    prototype_subtypes = prototype_data[
        "prototype_subtypes"
    ]

    prototype_types = prototype_data[
        "prototype_types"
    ]

    raw_prototypes = prototype_data[
        "raw_prototypes"
    ]

    prototype_train_counts = prototype_data[
        "prototype_train_counts"
    ]

    nonbgc_reference_centroid = (
        prototype_data[
            "nonbgc_reference_centroid"
        ]
    )

    prototype_norms = (
        raw_prototypes.norm(
            p=2,
            dim=1,
        )
    )

    if not bool(
        torch.allclose(
            prototype_norms,
            torch.ones_like(
                prototype_norms
            ),
            atol=1e-5,
            rtol=1e-5,
        )
    ):
        raise RuntimeError(
            "Raw prototype L2 normalization 失败。"
        )

    print("\nPrototype 数据：")
    print(
        f"BGC subtype prototypes："
        f"{len(prototype_subtypes):,}"
    )
    print(
        f"Prototype shape："
        f"{tuple(raw_prototypes.shape)}"
    )
    print(
        f"nonBGC train count："
        f"{prototype_data['nonbgc_train_count']:,}"
    )
    print(
        f"最小 prototype train count："
        f"{min(prototype_train_counts):,}"
    )
    print(
        f"最大 prototype train count："
        f"{max(prototype_train_counts):,}"
    )

    # --------------------------------------------------------
    # Baseline evaluation
    # --------------------------------------------------------

    device = choose_device(
        args.device
    )

    if device.type == "cuda":

        print(
            f"\n评价设备："
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        print(
            "\n评价设备：CPU"
        )

    baseline_metrics = {}
    baseline_predictions = {}

    for split_name in [
        "validation",
        "test",
    ]:

        (
            split_metrics,
            split_predictions,
        ) = evaluate_split(
            split_name=split_name,
            ids=ids,
            normalized_embeddings=(
                normalized_embeddings
            ),
            types=types,
            subtypes=subtypes,
            model_splits=model_splits,
            prototype_subtypes=(
                prototype_subtypes
            ),
            raw_prototypes=(
                raw_prototypes
            ),
            device=device,
            batch_size=args.batch_size,
        )

        baseline_metrics[
            split_name
        ] = split_metrics

        baseline_predictions[
            split_name
        ] = split_predictions

        print(
            f"\n{split_name} nearest-prototype baseline："
        )

        print(
            f"  BGC samples："
            f"{split_metrics['number_of_samples']:,}"
        )

        print(
            f"  Accuracy："
            f"{split_metrics['accuracy']:.4f}"
        )

        print(
            f"  Macro-F1："
            f"{split_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Top-5 accuracy："
            f"{split_metrics['top5_accuracy']:.4f}"
        )

        print(
            f"  Mean true margin："
            f"{split_metrics['mean_true_margin']:.4f}"
        )

    # --------------------------------------------------------
    # 保存输出
    # --------------------------------------------------------

    elapsed_seconds = (
        time.time()
        - started
    )

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

        "subtype_to_type": (
            prototype_data[
                "subtype_to_type"
            ]
        ),

        "raw_prototypes": (
            raw_prototypes
        ),

        "prototype_train_counts": (
            torch.tensor(
                prototype_train_counts,
                dtype=torch.long,
            )
        ),

        "nonbgc_reference_centroid": (
            nonbgc_reference_centroid
        ),

        "nonbgc_train_count": (
            prototype_data[
                "nonbgc_train_count"
            ]
        ),

        "prototype_construction": {
            "sequence_normalization": (
                "L2"
            ),
            "aggregation": (
                "model-train subtype mean"
            ),
            "prototype_normalization": (
                "L2"
            ),
            "used_split": (
                "model_splits == train"
            ),
            "validation_used": False,
            "test_used": False,
        },

        "baseline_metrics": (
            baseline_metrics
        ),

        "baseline_predictions": (
            baseline_predictions
        ),

        "source_long_fusion_pt": str(
            args.input_pt
        ),

        "source_id_sha256": (
            data["source"].get(
                "id_sha256"
            )
        ),

        "embedding_dimension": int(
            raw_prototypes.shape[1]
        ),

        "number_of_bgc_prototypes": (
            len(prototype_subtypes)
        ),

        "external_split_counts": dict(
            Counter(external_splits)
        ),

        "model_split_counts": dict(
            Counter(model_splits)
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
    # 保存 prototype 数量表
    # --------------------------------------------------------

    counts_csv = (
        args.output_pt.parent
        / "prototype_train_counts.csv"
    )

    save_prototype_counts(
        path=counts_csv,
        prototype_subtypes=(
            prototype_subtypes
        ),
        prototype_types=(
            prototype_types
        ),
        prototype_train_counts=(
            prototype_train_counts
        ),
    )

    # --------------------------------------------------------
    # 保存 JSON summary
    # --------------------------------------------------------

    summary = {
        "status": "COMPLETED",
        "script_version": (
            SCRIPT_VERSION
        ),
        "source_long_fusion_pt": str(
            args.input_pt
        ),
        "output_pt": str(
            args.output_pt
        ),
        "number_of_sequences": (
            len(ids)
        ),
        "number_of_bgc_prototypes": (
            len(prototype_subtypes)
        ),
        "prototype_shape": list(
            raw_prototypes.shape
        ),
        "minimum_train_count": min(
            prototype_train_counts
        ),
        "maximum_train_count": max(
            prototype_train_counts
        ),
        "nonbgc_train_count": (
            prototype_data[
                "nonbgc_train_count"
            ]
        ),
        "validation_baseline": (
            metrics_for_json(
                baseline_metrics[
                    "validation"
                ]
            )
        ),
        "test_baseline": (
            metrics_for_json(
                baseline_metrics[
                    "test"
                ]
            )
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            3,
        ),
    }

    summary_path = (
        args.output_pt.parent
        / "raw_prototype_summary.json"
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
    print("Step 03 完成")
    print("=" * 70)

    print(
        f"Prototype PT：{args.output_pt}"
    )

    print(
        f"Count CSV：  {counts_csv}"
    )

    print(
        f"Summary：    {summary_path}"
    )


if __name__ == "__main__":
    main()
