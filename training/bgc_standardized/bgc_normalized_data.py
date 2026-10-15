"""Read-only source audit and versioned BGC cluster-origin label normalization."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

BASE = Path(r"E:/1. ARGmodel/BGC数据处理")
SOURCE = BASE / "BGC_novel dataset"
OUT = Path(__file__).resolve().parent
DATASET = OUT / "Dataset"
NON = "nonBGC"
TYPE_MAP = {"pks": "PKS", "nrps": "NRPS", "ribosomal": "Ribosomal peptide",
            "terpene": "Terpene", "saccharide": "Saccharide", "other": "Other"}
TYPE_ORDER = ["PKS", "NRPS", "Ribosomal peptide", "Terpene", "Saccharide", "Other"]
POLICY_VERSION = "2026-10-06-cluster-origin-partial-labels-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def load_module(filename, alias):
    path = SOURCE / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def normalize_cluster(entry):
    classes = entry.get("biosynthesis", {}).get("classes", [])
    mapped = []
    for item in classes:
        name = item.get("class", "").lower()
        if name not in TYPE_MAP:
            raise ValueError(f"Unknown MIBiG class {name!r}: {entry['accession']}")
        mapped.append((TYPE_MAP[name], item))
    unique = sorted({name for name, _ in mapped}, key=TYPE_ORDER.index)
    if not unique:
        raise ValueError(f"No biosynthetic class: {entry['accession']}")
    raw_subclasses = " ; ".join(str(item.get("subclass", "")) for _, item in mapped)
    architecture = ""
    if len(unique) > 1:
        # A component composition, not a concatenation of heterogeneous fine labels.
        return "Hybrid", "Hybrid | " + " + ".join(unique), raw_subclasses, architecture
    type_name = unique[0]
    item = mapped[0][1]
    raw = " ".join(str(item.get("subclass", "")).split())
    lower = raw.lower()
    if type_name == "PKS":
        if "iterative" in lower:
            architecture = "Iterative"
        elif "trans-at" in lower or "trans at" in lower:
            architecture = "Trans-AT"
        elif "modular" in lower:
            architecture = "Modular"
        match = re.search(r"type\s+(iii|ii|i)(?![a-z])", lower)
        fine = "Type " + match.group(1).upper() if match else ""
    elif type_name == "Ribosomal peptide":
        if lower == "unmodified":
            fine = "Unmodified"
        elif lower == "ripp":
            ripp_type = str(item.get("ripp_type") or "").strip()
            fine = ripp_type if ripp_type.lower() not in {"", "other", "unknown"} else ""
        else:
            raise ValueError(f"Unrecognized ribosomal subclass {raw}: {entry['accession']}")
    else:
        fine = raw if lower not in {"", "unknown", "unclassified", "other", "na", "n/a"} else ""
    return type_name, f"{type_name} | {fine}" if fine else "", raw_subclasses, architecture


def prepare():
    label_path = SOURCE / "BGC_corrected_files/corrected_BGC_nonBGC_id_type_subtype.csv"
    split_path = SOURCE / "Long_fusion/model_split_metadata.csv"
    old = read_csv(label_path)
    assignments = {r["id"]: r for r in read_csv(split_path)}
    if len(assignments) != len(old) or set(assignments) != {r["id"] for r in old}:
        raise ValueError("Original labels and immutable split ID sets differ.")
    cluster_cache = {}
    rows = []
    source_json_digest = hashlib.sha256()
    for original in old:
        row = dict(original)
        split = assignments[row["id"]]
        if row["group_id"] != split["group_id"]:
            raise ValueError("BGC group assignment changed.")
        if row["source"] == "nonBGC":
            kind, candidate, raw, architecture = NON, NON, "", ""
        else:
            group = row["group_id"]
            if group not in cluster_cache:
                path = BASE / "mibig_json_4.0" / (group + ".json")
                raw_bytes = path.read_bytes()
                source_json_digest.update(group.encode() + b"\0" + raw_bytes)
                entry = json.loads(raw_bytes)
                if entry.get("accession") != group or entry.get("status") != "active":
                    raise ValueError(f"Unexpected source BGC/status: {group}")
                cluster_cache[group] = normalize_cluster(entry)
            kind, candidate, raw, architecture = cluster_cache[group]
        rows.append({"id": row["id"], "type": kind, "subtype": candidate,
                     "source": row["source"], "group_id": row["group_id"],
                     "external_split": split["external_split"], "model_split": split["model_split"],
                     "original_type": row["type"], "original_subtype": row["subtype"],
                     "raw_subclass": raw, "PKS_architecture": architecture,
                     "subtype_candidate": candidate, "subtype_evaluable": 0,
                     "fine_label_exclusion_reason": ""})
    groups = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["subtype_candidate"] and r["source"] == "BGC":
            groups[r["subtype_candidate"]][r["model_split"]].add(r["group_id"])
    eligible = set()
    support = []
    for label, splits in sorted(groups.items()):
        total_groups = set().union(*splits.values())
        # This is a predeclared structural support rule, never a prediction-score filter.
        reason = ("fewer_than_two_source_BGCs" if len(total_groups) < 2
                  else "absent_from_model_train" if not splits["train"] else "")
        if not reason:
            eligible.add(label)
        support.append({"subtype": label, "type": label.split(" | ")[0],
                        "source_BGC_groups": len(total_groups),
                        "train_BGC_groups": len(splits["train"]),
                        "validation_BGC_groups": len(splits["validation"]),
                        "test_BGC_groups": len(splits["test"]),
                        "retained_fine_class": int(not reason), "exclusion_reason": reason})
    for r in rows:
        if r["source"] == "nonBGC":
            r["subtype_evaluable"] = 1
        elif not r["subtype_candidate"]:
            r["fine_label_exclusion_reason"] = "missing_or_nonspecific_subtype_annotation"
        elif r["subtype_candidate"] in eligible:
            r["subtype_evaluable"] = 1
        else:
            match = next(x for x in support if x["subtype"] == r["subtype_candidate"])
            r["fine_label_exclusion_reason"] = match["exclusion_reason"]
        if not r["subtype_evaluable"]:
            r["subtype"] = ""
    group_splits = defaultdict(set)
    group_labels = defaultdict(set)
    for r in rows:
        group_splits[r["group_id"]].add(r["model_split"])
        group_labels[r["group_id"]].add((r["type"], r["subtype"]))
        if (r["external_split"] == "test") != (r["model_split"] == "test"):
            raise ValueError("External split leakage.")
    if any(len(s) != 1 for s in group_splits.values()) or any(len(s) != 1 for s in group_labels.values()):
        raise ValueError("A BGC spans splits or conflicting labels.")
    fine_types = {r["type"] for r in rows if r["subtype_evaluable"]}
    if fine_types != {r["type"] for r in rows}:
        raise ValueError(f"Types lack a trainable fine label: {set(r['type'] for r in rows)-fine_types}")
    DATASET.mkdir(parents=True, exist_ok=True)
    destination = DATASET / "normalized_BGC_nonBGC_labels_and_fixed_splits.csv"
    if destination.exists():
        existing = read_csv(destination)
        if [{k: str(v) for k, v in r.items()} for r in rows] != existing:
            raise FileExistsError("Prepared dataset differs; do not overwrite an experiment.")
    else:
        write_csv(destination, rows)
        write_csv(DATASET / "subtype_support_and_exclusions.csv", support)
        mappings = Counter((r["original_type"], r["original_subtype"], r["type"],
                            r["subtype"], r["fine_label_exclusion_reason"]) for r in rows)
        write_csv(DATASET / "old_to_new_label_audit.csv", [
            {"original_type": key[0], "original_subtype": key[1], "normalized_type": key[2],
             "normalized_subtype": key[3], "fine_label_exclusion_reason": key[4], "sequence_count": n}
            for key, n in sorted(mappings.items())])
    coverage = {}
    for split in ("train", "validation", "test"):
        local = [r for r in rows if r["model_split"] == split]
        positives = [r for r in local if r["source"] == "BGC"]
        fine = sum(r["subtype_evaluable"] for r in positives)
        coverage[split] = {"all_records": len(local), "BGC_records": len(positives),
                           "nonBGC_records": len(local)-len(positives),
                           "fine_annotated_BGC_records": fine,
                           "fine_BGC_annotation_coverage": fine/len(positives),
                           "fine_end_to_end_evaluation_records": fine+len(local)-len(positives)}
    manifest = {"policy_version": POLICY_VERSION, "source_labels_sha256": sha256(label_path),
                "source_fixed_split_sha256": sha256(split_path),
                "source_JSON_combined_sha256": source_json_digest.hexdigest(),
                "normalized_labels_sha256": sha256(destination),
                "total_records": len(rows), "source_counts": dict(Counter(r["source"] for r in rows)),
                "positive_type_counts": dict(Counter(r["type"] for r in rows if r["source"] == "BGC")),
                "positive_fine_class_count": len(eligible), "fine_classes": sorted(eligible),
                "coverage": coverage, "group_split_overlap": 0, "all_original_ID_assignments_preserved": True,
                "fine_label_exclusions": dict(Counter(r["fine_label_exclusion_reason"] for r in rows if not r["subtype_evaluable"])),
                "rare_class_rule": "At least two independent source BGCs and at least one model-training BGC; no filtering by model performance.",
                "validation_support_warning": "Some fine classes have no validation BGC; all support counts are disclosed.",
                "supervision": "Fine CE for annotated classes; marginal probability CE over type-compatible fine labels for coarse-only positive records; no new type head.",
                "prototype_alignment": "Fine prototypes for annotated positives and negative center for negatives; excluded for coarse-only positives.",
                "inference": "Binary gate followed by highest-probability fine label and predefined fine-to-type mapping, as in the original framework.",
                "evaluation": "Binary positive-class P/R/F1 and ACC/MCC on all records; type macro on all records; subtype weighted on fine-annotated positives plus all negatives; denominators disclosed."}
    if (DATASET / "dataset_manifest.json").exists():
        old_manifest = json.loads((DATASET / "dataset_manifest.json").read_text(encoding="utf-8"))
        if old_manifest != manifest:
            raise ValueError("Dataset manifest drift.")
    else:
        write_json(DATASET / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return rows, manifest


def load_features(rows):
    module = load_module("02_local_bgc_gated_fusion.py", "legacy_bgc_fusion")
    targets = [r["id"] for r in rows]
    paths = [SOURCE / "ESM2_BGC/BGC_nonBGC_esm2_mean.pt",
             SOURCE / "DNABERT2_BGC/BGC_nonBGC_dnabert2_mean.pt"]
    tensors = []
    for path, dimension in zip(paths, (1280, 768)):
        ids, embeddings = module.load_embedding_pt(path, dimension)
        if set(ids) != set(targets):
            raise ValueError(f"Embedding ID mismatch: {path}")
        index = {value: i for i, value in enumerate(ids)}
        tensors.append(embeddings[torch.tensor([index[value] for value in targets])].contiguous())
    return tensors


if __name__ == "__main__":
    prepare()
