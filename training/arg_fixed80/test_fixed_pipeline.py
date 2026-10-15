"""Regression checks; no full model training. Uses the real fixed manifest."""
import argparse
import copy
import importlib
import json
import random
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from fixed_split import FixedSplit, EXPECTED_FINGERPRINT
from suite_runtime import Context, TRANSFER_FIELDS, SUITE, load_base, apply_no_gate, apply_no_transfer
from run_pipeline import plan

DATA_ROOT = SUITE.parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = Context(DATA_ROOT)
        cls.authority = cls.context.split

    def setUp(self):
        sys.path.insert(0, str(SUITE / "core"))
        import newsplit_contract
        self.contract = importlib.reload(newsplit_contract)
        self.contract.configure(self.context.split_dir, "full")

    def fusion_manifest(self):
        rows = self.authority.rows
        ids = [r["sequence_id"] for r in rows]
        labels = [r["Subtype"] for r in rows]
        model, external = self.authority.aligned_splits(ids, labels, self.authority.external_train,
                                                      self.authority.ids["test"], .2, 20260822)
        obj = {"schema": self.contract.SCHEMA, "ids": ids, "subtypes": labels,
               "splits": external, "external_splits": external, "model_splits": model,
               "exact_ids": {s: [i for i, split in zip(ids, model) if split == s] for s in self.authority.ids},
               "subtype_to_type": self.authority.type_mapping, "fusion_chain": "full",
               "authoritative_split_fingerprint": EXPECTED_FINGERPRINT}
        obj["split_fingerprint"] = self.contract.fingerprint(obj)
        return obj

    def test_exact_counts_and_all_820_subtypes(self):
        self.assertEqual({s: len(ids) for s, ids in self.authority.ids.items()},
                         {"train": 14636, "validation": 3660, "test": 4574})
        self.assertEqual(len(self.authority.type_mapping), 820)

    def test_reordering_never_regenerates_validation(self):
        ids = list(self.authority.by_id)
        for seed in range(5):
            random.Random(seed).shuffle(ids)
            labels = [self.authority.by_id[i]["Subtype"] for i in ids]
            model, _ = self.contract.build_model_splits_from_new_ids(
                ids, labels, self.authority.external_train, self.authority.ids["test"], .2, 20260822)
            self.assertEqual({i for i, s in zip(ids, model) if s == "validation"}, self.authority.ids["validation"])

    def test_old_ids_and_changed_labels_are_rejected(self):
        obj = self.fusion_manifest()
        self.contract.validate_fusion(obj)
        with self.assertRaises(ValueError):
            self.contract.build_model_splits_from_new_ids(obj["ids"], obj["subtypes"],
                                                         self.authority.ids["train"], self.authority.ids["test"], .2, 20260822)
        bad = copy.deepcopy(obj)
        bad["subtypes"][0] = "wrong label"
        bad["split_fingerprint"] = self.contract.fingerprint(bad)
        with self.assertRaises(ValueError):
            self.contract.validate_fusion(bad)

    def test_wrong_chain_and_old_artifact_rejected(self):
        obj = self.fusion_manifest()
        for key, value in (("fusion_chain", "wo_gate"), ("schema", "argscope-subtype2-newsplit-v1"),
                           ("authoritative_split_fingerprint", "old")):
            bad = copy.deepcopy(obj)
            bad[key] = value
            with self.assertRaises(ValueError):
                self.contract.validate_fusion(bad)

    def test_validation_tampering_rejected_even_with_rehashed_artifact(self):
        obj = self.fusion_manifest()
        index = obj["model_splits"].index("validation")
        obj["model_splits"][index] = "train"
        obj["split_fingerprint"] = self.contract.fingerprint(obj)
        with self.assertRaises(ValueError):
            self.contract.validate_fusion(obj)

    def test_plan_includes_new_full_and_all_eight_students(self):
        tasks = plan(("full", "wo_gate", "wo_graph", "wo_transfer"))
        self.assertEqual(len(tasks), 14)
        self.assertEqual(len([s for v, s in tasks if s in ("protein", "dna")]), 8)
        self.assertNotIn(("full", "graph"), plan(("wo_graph",)))
        self.assertEqual(len(tasks), len(set(tasks)))

    def test_no_gate_is_equal_weight_and_gate_has_no_gradient(self):
        import torch
        m = load_base("fusion", self.context, "wo_gate")
        apply_no_gate(m)
        model = m.SupervisedGatedFusion(protein_dim=1280, dna_dim=768, fusion_dim=512,
                                        number_subtypes=821, dropout=0., protein_mean=torch.zeros(1280),
                                        protein_std=torch.ones(1280), dna_mean=torch.zeros(768), dna_std=torch.ones(768))
        protein, dna = torch.randn(3, 1280), torch.randn(3, 768)
        z, gate = model.encode(protein, dna)
        expected = model.fusion_norm(.5 * model.protein_projection(protein) + .5 * model.dna_projection(dna))
        self.assertTrue(torch.equal(z, expected))
        self.assertTrue(torch.all(gate == .5))
        z.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.gate.parameters()))

    def test_transfer_coefficients_zero_and_dna_progress_signature(self):
        import inspect
        captured = []
        fake = SimpleNamespace(train_student=lambda teacher, data, config, *args: captured.append(config))
        apply_no_transfer(fake)
        self.assertIn("progress_dir", inspect.signature(fake.train_teacher).parameters)
        config = SimpleNamespace(**{field: .5 for field in TRANSFER_FIELDS}, student_binary_ce=1., student_subtype_ce=1.3)
        fake.train_student(None, {}, config, False, None, [], None)
        self.assertTrue(all(getattr(captured[0], field) == 0 for field in TRANSFER_FIELDS))
        self.assertEqual(config.student_subtype_ce, 1.3)

    def test_bundled_protein_parameters_contain_no_old_metrics(self):
        import csv
        with (SUITE / "protein_parameters.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertTrue(all(k.startswith("param_") or k.startswith("selected_teacher_") or
                            k == "selected_binary_ARG_probability_threshold" for k in row))
        m = load_base("protein", self.context, "full")
        config, meta, _ = m.load_grid_best_parameters(SUITE / "protein_parameters.csv", m.Config())
        self.assertEqual((config.teacher_epochs, config.student_epochs), (100, 200))
        self.assertEqual(meta["reference_binary_threshold"], .8)
        self.assertEqual(config.seed, 20260822)

    def test_five_metric_definitions(self):
        import math
        from summarize_results import five_metrics
        truth = [1] * 4 + [0] * 6
        predicted = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        metrics = five_metrics(truth, predicted, "binary")
        for key, expected in {"Accuracy": .7, "Precision": .6, "Recall": .75,
                              "F1": 2/3, "MCC": 10/math.sqrt(600)}.items():
            self.assertAlmostEqual(metrics[key], expected)
        for average in ("macro", "weighted", "micro"):
            row = five_metrics(truth, predicted, average)
            self.assertEqual(row["Accuracy"], metrics["Accuracy"])
            self.assertEqual(row["MCC"], metrics["MCC"])
            if average == "micro":
                for key in ("Precision", "Recall", "F1"):
                    self.assertAlmostEqual(row[key], .7)

    def test_original_core_model_implementations_unchanged(self):
        from fixed_split import sha256
        reference = json.loads((SUITE / "core_source_manifest.json").read_text())
        for name, expected in reference["sha256"].items():
            self.assertEqual(sha256(SUITE / "core" / name), expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    unittest.main(argv=[sys.argv[0]], verbosity=2)
