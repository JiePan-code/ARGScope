#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 subtype 分层、以完整 BGC group 为单位划分 train/test。

规则：
1. 每个 subtype 的独立 group 按 8:2 划分。
2. 同一个 group 的全部序列只能进入同一个 split。
3. 每个 subtype 在 train 和 test 中至少各有 1 个 group。
4. 因此所有 type 也必须同时出现在 train 和 test。
5. train/test ID 和 group 严格互斥。
6. 使用固定随机种子，可重复获得相同结果。
"""

import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 路径设置
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_CSV = ROOT / "BGC_corrected_files" / "corrected_BGC_nonBGC_id_type_subtype.csv"

OUTPUT_DIR = ROOT / "Dataset_split"

TRAIN_CSV = OUTPUT_DIR / "BGC_nonBGC_id_type_subtype_train.csv"
TEST_CSV = OUTPUT_DIR / "BGC_nonBGC_id_type_subtype_test.csv"
SPLIT_CSV = OUTPUT_DIR / "BGC_nonBGC_id_type_subtype_split.csv"

TRAIN_IDS = OUTPUT_DIR / "train_ids.txt"
TEST_IDS = OUTPUT_DIR / "test_ids.txt"

RANDOM_SEED = 20260915
TEST_RATIO = 0.20


# ============================================================
# 工具函数
# ============================================================

def clean_text(value):
    """删除首尾空格，并合并连续空格。"""
    return " ".join(str(value).strip().split())


def stable_seed(global_seed, subtype):
    """
    为每个 subtype 生成稳定随机种子。

    这样即使 CSV 中的行顺序发生变化，
    相同 subtype 的划分结果仍然可重复。
    """
    text = f"{global_seed}\0{subtype}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little")


# ============================================================
# 1. 读取 ID 表
# ============================================================

if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV}")

rows = []

with INPUT_CSV.open(
    "r",
    encoding="utf-8-sig",
    newline=""
) as handle:

    reader = csv.DictReader(handle)

    if not reader.fieldnames:
        raise ValueError("输入 CSV 没有表头。")

    # 允许列名大小写不同
    column_map = {
        column.strip().lower(): column
        for column in reader.fieldnames
    }

    required_columns = {
        "id",
        "type",
        "subtype",
        "source",
        "group_id",
    }

    missing_columns = required_columns - set(column_map)

    if missing_columns:
        raise ValueError(
            f"输入 CSV 缺少列：{sorted(missing_columns)}；"
            f"当前列为：{reader.fieldnames}"
        )

    for line_number, row in enumerate(reader, start=2):

        seq_id = clean_text(row[column_map["id"]])
        type_label = clean_text(row[column_map["type"]])
        subtype = clean_text(row[column_map["subtype"]])
        source = clean_text(row[column_map["source"]])
        group_id = clean_text(row[column_map["group_id"]])

        if not seq_id:
            raise ValueError(f"第 {line_number} 行的 id 为空。")

        if not type_label:
            raise ValueError(f"第 {line_number} 行的 type 为空。")

        if not subtype:
            raise ValueError(f"第 {line_number} 行的 subtype 为空。")

        if not source:
            raise ValueError(f"第 {line_number} 行的 source 为空。")

        if not group_id:
            raise ValueError(f"第 {line_number} 行的 group_id 为空。")

        rows.append({
            "id": seq_id,
            "type": type_label,
            "subtype": subtype,
            "source": source,
            "group_id": group_id,
        })


if not rows:
    raise ValueError("输入 CSV 中没有数据。")


# ============================================================
# 2. 检查重复 ID
# ============================================================

all_ids = [row["id"] for row in rows]

id_counts = Counter(all_ids)

duplicate_ids = sorted(
    seq_id
    for seq_id, count in id_counts.items()
    if count > 1
)

if duplicate_ids:
    raise ValueError(
        f"发现 {len(duplicate_ids)} 个重复 ID；"
        f"示例：{duplicate_ids[:10]}"
    )


# ============================================================
# 3. 检查 subtype/type/group 的层级关系
# ============================================================

subtype_to_types = defaultdict(set)

for row in rows:
    subtype_to_types[row["subtype"]].add(row["type"])

ambiguous_subtypes = {
    subtype: sorted(types)
    for subtype, types in subtype_to_types.items()
    if len(types) > 1
}

if ambiguous_subtypes:
    examples = list(ambiguous_subtypes.items())[:10]

    raise ValueError(
        "发现同一个 subtype 对应多个 type，无法保证层级标签一致。\n"
        f"示例：{examples}"
    )


group_to_labels = defaultdict(set)

for row in rows:
    group_to_labels[row["group_id"]].add(
        (row["type"], row["subtype"])
    )

ambiguous_groups = {
    group_id: sorted(labels)
    for group_id, labels in group_to_labels.items()
    if len(labels) > 1
}

if ambiguous_groups:
    examples = list(ambiguous_groups.items())[:10]
    raise ValueError(
        "发现同一个 group_id 对应多个 type/subtype，无法安全划分。\n"
        f"示例：{examples}"
    )


# ============================================================
# 4. 按 subtype 和 group_id 分组
# ============================================================

groups_by_subtype = defaultdict(lambda: defaultdict(list))

for row in rows:
    groups_by_subtype[row["subtype"]][row["group_id"]].append(row)

subtype_group_counts = {
    subtype: len(group_rows)
    for subtype, group_rows in groups_by_subtype.items()
}

singleton_subtypes = sorted(
    subtype
    for subtype, count in subtype_group_counts.items()
    if count < 2
)

if singleton_subtypes:
    raise ValueError(
        f"发现 {len(singleton_subtypes)} 个少于 2 个独立 group 的 subtype，"
        "无法保证 group 不泄漏并同时放入 train 和 test。\n"
        f"示例：{singleton_subtypes[:10]}"
    )


# ============================================================
# 5. 每个 subtype 的 group 按 8:2 划分
# ============================================================

train_rows = []
test_rows = []

for subtype in sorted(groups_by_subtype):

    subtype_groups = sorted(
        groups_by_subtype[subtype].items(),
        key=lambda item: item[0],
    )

    # 先按 group_id 排序，避免原始 CSV 行顺序影响划分结果

    rng = random.Random(
        stable_seed(RANDOM_SEED, subtype)
    )

    rng.shuffle(subtype_groups)

    number_of_groups = len(subtype_groups)

    # 常规四舍五入计算 20% test groups
    number_of_test_groups = int(
        math.floor(
            number_of_groups * TEST_RATIO + 0.5
        )
    )

    # 每个 subtype 的 train/test 至少各有 1 个完整 group
    number_of_test_groups = max(1, number_of_test_groups)
    number_of_test_groups = min(
        number_of_test_groups,
        number_of_groups - 1
    )

    subtype_test_groups = subtype_groups[:number_of_test_groups]
    subtype_train_groups = subtype_groups[number_of_test_groups:]

    subtype_test = [
        row
        for _, group_rows in subtype_test_groups
        for row in group_rows
    ]
    subtype_train = [
        row
        for _, group_rows in subtype_train_groups
        for row in group_rows
    ]

    if not subtype_train or not subtype_test:
        raise RuntimeError(
            f"subtype group 划分失败：{subtype}"
        )

    train_rows.extend(subtype_train)
    test_rows.extend(subtype_test)


# ============================================================
# 6. 恢复原始 CSV 顺序
# ============================================================

original_order = {
    row["id"]: index
    for index, row in enumerate(rows)
}

train_rows.sort(
    key=lambda row: original_order[row["id"]]
)

test_rows.sort(
    key=lambda row: original_order[row["id"]]
)


# ============================================================
# 7. 严格检查 train/test
# ============================================================

train_ids = {
    row["id"]
    for row in train_rows
}

test_ids = {
    row["id"]
    for row in test_rows
}

if train_ids & test_ids:
    overlap = sorted(train_ids & test_ids)

    raise RuntimeError(
        f"train/test 出现 ID 重叠：{overlap[:10]}"
    )

if train_ids | test_ids != set(all_ids):
    missing_ids = set(all_ids) - (train_ids | test_ids)

    raise RuntimeError(
        f"train/test 没有覆盖所有 ID；缺少：{sorted(missing_ids)[:10]}"
    )


train_group_ids = {
    row["group_id"]
    for row in train_rows
}

test_group_ids = {
    row["group_id"]
    for row in test_rows
}

if train_group_ids & test_group_ids:
    overlap = sorted(train_group_ids & test_group_ids)
    raise RuntimeError(
        f"train/test 出现 group_id 重叠：{overlap[:10]}"
    )

all_group_ids = set(group_to_labels)

if train_group_ids | test_group_ids != all_group_ids:
    missing_groups = all_group_ids - (train_group_ids | test_group_ids)
    raise RuntimeError(
        "train/test 没有覆盖所有 group_id；缺少："
        f"{sorted(missing_groups)[:10]}"
    )


# ============================================================
# 8. 检查所有 subtype 是否同时存在于两边
# ============================================================

all_subtypes = {
    row["subtype"]
    for row in rows
}

train_subtypes = {
    row["subtype"]
    for row in train_rows
}

test_subtypes = {
    row["subtype"]
    for row in test_rows
}

missing_train_subtypes = all_subtypes - train_subtypes
missing_test_subtypes = all_subtypes - test_subtypes

if missing_train_subtypes:
    raise RuntimeError(
        "以下 subtype 没有进入 train："
        f"{sorted(missing_train_subtypes)[:10]}"
    )

if missing_test_subtypes:
    raise RuntimeError(
        "以下 subtype 没有进入 test："
        f"{sorted(missing_test_subtypes)[:10]}"
    )


# ============================================================
# 9. 检查所有 type 是否同时存在于两边
# ============================================================

all_types = {
    row["type"]
    for row in rows
}

train_types = {
    row["type"]
    for row in train_rows
}

test_types = {
    row["type"]
    for row in test_rows
}

missing_train_types = all_types - train_types
missing_test_types = all_types - test_types

if missing_train_types:
    raise RuntimeError(
        "以下 type 没有进入 train："
        f"{sorted(missing_train_types)}"
    )

if missing_test_types:
    raise RuntimeError(
        "以下 type 没有进入 test："
        f"{sorted(missing_test_types)}"
    )


# ============================================================
# 10. 写出结果
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "id",
    "type",
    "subtype",
    "source",
    "group_id",
]


def write_label_csv(path, output_rows):

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)


write_label_csv(
    TRAIN_CSV,
    train_rows
)

write_label_csv(
    TEST_CSV,
    test_rows
)


# ============================================================
# 11. 写出包含 split 列的完整表
# ============================================================

split_by_id = {
    seq_id: "train"
    for seq_id in train_ids
}

split_by_id.update({
    seq_id: "test"
    for seq_id in test_ids
})

with SPLIT_CSV.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "id",
            "type",
            "subtype",
            "source",
            "group_id",
            "split",
        ]
    )

    writer.writeheader()

    for row in rows:
        writer.writerow({
            "id": row["id"],
            "type": row["type"],
            "subtype": row["subtype"],
            "source": row["source"],
            "group_id": row["group_id"],
            "split": split_by_id[row["id"]],
        })


# ============================================================
# 12. 写出 ID 文件
# ============================================================

TRAIN_IDS.write_text(
    "\n".join(row["id"] for row in train_rows) + "\n",
    encoding="utf-8"
)

TEST_IDS.write_text(
    "\n".join(row["id"] for row in test_rows) + "\n",
    encoding="utf-8"
)


# ============================================================
# 13. 输出统计信息
# ============================================================

print("=" * 60)
print("Train/test 划分完成")
print("=" * 60)

print(f"总序列数：       {len(rows):,}")
print(f"Train 序列数：   {len(train_rows):,}")
print(f"Test 序列数：    {len(test_rows):,}")

print(f"\n总 group 数：   {len(all_group_ids):,}")
print(f"Train group 数： {len(train_group_ids):,}")
print(f"Test group 数：  {len(test_group_ids):,}")

print(
    f"实际比例：       "
    f"{len(train_rows) / len(rows):.2%} train / "
    f"{len(test_rows) / len(rows):.2%} test"
)

print(f"\n总 type 数：     {len(all_types):,}")
print(f"Train type 数：  {len(train_types):,}")
print(f"Test type 数：   {len(test_types):,}")

print(f"\n总 subtype 数：    {len(all_subtypes):,}")
print(f"Train subtype 数： {len(train_subtypes):,}")
print(f"Test subtype 数：  {len(test_subtypes):,}")

print("\n检查结果：")
print("Train/test ID 无重叠：PASS")
print("Train/test group_id 无重叠：PASS")
print("所有 ID 完整覆盖：PASS")
print("所有 subtype 同时存在于 train/test：PASS")
print("所有 type 同时存在于 train/test：PASS")

print("\n输出文件：")
print(TRAIN_CSV)
print(TEST_CSV)
print(SPLIT_CSV)
print(TRAIN_IDS)
print(TEST_IDS)
