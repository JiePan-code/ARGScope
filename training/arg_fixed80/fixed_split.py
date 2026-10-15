"""Read-only authority for the already-generated 80:20 grouped split."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_FINGERPRINT = "6d2d3cf7ad252f255dedf4e4bbeae693f5ea0c5112f39daec331aaa975d93991"
STRUCTURE_SHA256 = "8c1eb57cd6bb34cf3d38500daf4141d4ba46563f2abee13527058cca9fdb6a24"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path):
    values = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not values or any(not x or x != x.strip() for x in values) or len(set(values)) != len(values):
        raise ValueError(f"Empty, malformed or duplicate IDs: {path}")
    return set(values)


class FixedSplit:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "split_contract.json").read_text(encoding="utf-8-sig"))
        if self.metadata.get("split_fingerprint") != EXPECTED_FINGERPRINT:
            raise ValueError("Wrong split contract: this suite is pinned to the new 80:20 split.")
        required = ("split_manifest.csv", "population_ids.txt", "train_ids_subtype2.txt",
                    "test_ids_subtype2.txt", "model_train_ids.txt", "validation_ids.txt")
        for name in required:
            if sha256(self.directory / name) != self.metadata["output_sha256"].get(name):
                raise ValueError(f"Split file has changed: {name}")
        with (self.directory / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        canonical = json.dumps(self.rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != EXPECTED_FINGERPRINT:
            raise ValueError("Split manifest fingerprint mismatch")
        self.by_id = {r["sequence_id"]: r for r in self.rows}
        if len(self.by_id) != len(self.rows) or len(self.rows) != 22870:
            raise ValueError("Wrong population or duplicate IDs")
        self.ids = {s: {r["sequence_id"] for r in self.rows if r["model_split"] == s}
                    for s in ("train", "validation", "test")}
        self.external_train = self.ids["train"] | self.ids["validation"]
        expected_lists = {"population_ids.txt": set(self.by_id), "train_ids_subtype2.txt": self.external_train,
                          "test_ids_subtype2.txt": self.ids["test"], "model_train_ids.txt": self.ids["train"],
                          "validation_ids.txt": self.ids["validation"]}
        for name, expected in expected_lists.items():
            if read_ids(self.directory / name) != expected:
                raise ValueError(f"Manifest / ID-list disagreement: {name}")
        if {s: len(v) for s, v in self.ids.items()} != {"train": 14636, "validation": 3660, "test": 4574}:
            raise ValueError("Incorrect train/validation/test counts")
        self.type_mapping = {}
        seen_groups = {key: {} for key in ("sequence_group", "protein_sha256", "dna_sha256")}
        for r in self.rows:
            expected_ext = "test" if r["model_split"] == "test" else "train"
            if r["external_split"] != expected_ext:
                raise ValueError("Validation must be a subset of external train")
            for key, seen in seen_groups.items():
                value = r[key]
                if not value or seen.setdefault(value, r["model_split"]) != r["model_split"]:
                    raise ValueError(f"Cross-split sequence leakage: {key}")
            if r["ARG_or_nonARG"] == "ARG":
                if self.type_mapping.setdefault(r["Subtype"], r["Type"]) != r["Type"]:
                    raise ValueError("Conflicting subtype / Type labels")
            elif r["ARG_or_nonARG"] != "nonARG" or r["Subtype"] != "nonARG" or r["Type"] != "nonARG":
                raise ValueError("Invalid binary class / negative labels")
        for s in ("train", "test"):
            arg_rows = [r for r in self.rows if r["model_split"] == s and r["ARG_or_nonARG"] == "ARG"]
            if {r["Subtype"] for r in arg_rows} != set(self.type_mapping) or len(self.type_mapping) != 820:
                raise ValueError("Train and test must both contain all 820 ARG subtypes")
            if len({r["Type"] for r in arg_rows}) != 23:
                raise ValueError("Train and test must both contain all 23 ARG Types")
        for s, selected in self.ids.items():
            counts = Counter(self.by_id[x]["ARG_or_nonARG"] for x in selected)
            if counts["ARG"] != counts["nonARG"]:
                raise ValueError(f"Unbalanced binary classes: {s}")

    def aligned_splits(self, ids, subtypes, train_ids, test_ids, validation_fraction, seed):
        if float(validation_fraction) != .2 or int(seed) != 20260822:
            raise ValueError("This fixed split requires validation_fraction=.20 and seed=20260822")
        if len(ids) != len(set(ids)) or set(ids) != set(self.by_id) or len(subtypes) != len(ids):
            raise ValueError("Aligned data must contain exactly the fixed population, once each")
        if set(train_ids) != self.external_train or set(test_ids) != self.ids["test"]:
            raise ValueError("Old or different train/test IDs supplied")
        if any(self.by_id[sid]["Subtype"] != label for sid, label in zip(ids, subtypes)):
            raise ValueError("Labels differ from the fixed split manifest")
        return ([self.by_id[sid]["model_split"] for sid in ids],
                [self.by_id[sid]["external_split"] for sid in ids])

    def validate_structure(self, path):
        if sha256(path) != STRUCTURE_SHA256:
            raise ValueError("Structure file changed since the split was built")

    def validate_predictions(self, path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or len(rows) != len(self.ids["test"]):
            raise ValueError(f"Expected 4574 test predictions: {path}")
        id_key = "sequence_id" if "sequence_id" in rows[0] else "id"
        if len({r[id_key] for r in rows}) != len(rows) or {r[id_key] for r in rows} != self.ids["test"]:
            raise ValueError("Prediction IDs do not match the fixed test set")
        mapping = {**self.type_mapping, "nonARG": "nonARG"}
        for r in rows:
            truth = self.by_id[r[id_key]]
            for key, source in (("true_subtype", "Subtype"), ("true_type", "Type")):
                if r[key] != truth[source]:
                    raise ValueError(f"Prediction ground truth mismatch: {r[id_key]} / {key}")
            if r["predicted_subtype"] not in mapping or mapping[r["predicted_subtype"]] != r["predicted_type"]:
                raise ValueError("Unknown prediction label or inconsistent subtype / Type")
            if int(r["true_ARG"]) != int(truth["ARG_or_nonARG"] == "ARG"):
                raise ValueError("Inconsistent binary ground truth")
            if int(r["predicted_ARG"]) != int(r["predicted_subtype"] != "nonARG"):
                raise ValueError("Inconsistent binary / subtype prediction")
        return rows

    def validate_student_manifest(self, path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(self.rows) or {r["sequence_id"] for r in rows} != set(self.by_id):
            raise ValueError("Student saved an incomplete or duplicate split manifest")
        for r in rows:
            truth = self.by_id[r["sequence_id"]]
            for key in ("model_split", "external_split"):
                if r[key] != truth[key]:
                    raise ValueError(f"Student regenerated or changed the split: {key}")
