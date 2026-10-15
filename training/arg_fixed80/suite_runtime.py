"""Paths, stage receipts and small, explicit ablation hooks. No model redesign."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from fixed_split import EXPECTED_FINGERPRINT, FixedSplit, sha256

SUITE = Path(__file__).resolve().parent
VARIANTS = ("full", "wo_gate", "wo_graph", "wo_transfer")
STAGES = {"fusion": "01_gated_fusion.py", "raw": "02_raw_prototypes.py",
          "graph": "03_refine_gat.py", "protein": "04_protein_student.py", "dna": "05_dna_student.py"}
ARTIFACT_NAMES = {"fusion": "fusion.pt", "raw": "raw_prototypes.pt", "graph": "refined_prototypes.pt",
                  "protein": "protein_100pct_epoch_features_run_best_model.pt", "dna": "DNA_student_long_model.pt"}
TRANSFER_FIELDS = ("student_binary_kd", "student_subtype_kd", "student_feature_align",
                   "student_prototype_align", "student_head_anchor")


def add_path_arguments(parser):
    parser.add_argument("--data-root", type=Path, default=SUITE.parent)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="One-epoch integration test; NOT scientific results")


@dataclass
class Context:
    data_root: Path
    output_root: Path | None = None
    smoke_test: bool = False

    def __post_init__(self):
        self.data_root = self.data_root.resolve()
        name = "ARGScope_80_20_SMOKE_ONLY" if self.smoke_test else "ARGScope_80_20_results"
        self.output_root = (self.output_root or self.data_root / name).resolve()
        if self.output_root == self.data_root or self.output_root in self.data_root.parents or self.output_root == SUITE:
            raise ValueError("Results must be in a separate child directory, never the data/workspace root")
        self.split_dir = self.data_root / "split_80_20_grouped"
        self.split = FixedSplit(self.split_dir)
        self.structure = self.data_root / "all_ARG_nonARG_1to1_structure.txt"
        self.split.validate_structure(self.structure)
        self.protein = self.data_root / "ESM2_EMBEDDING/percent_100/full_1to1_esm2_mean.pt"
        self.dna = self.data_root / "DNABERT2_embedding/percent_100/full_1to1_dnabert2_mean.pt"
        # No searching neighbouring experiments for weights or old split files.
        for path in (self.protein, self.dna, SUITE / "protein_parameters.csv"):
            if not path.is_file():
                raise FileNotFoundError(path)

    def directory(self, variant, stage):
        return self.output_root / variant / stage

    def artifact(self, variant, stage):
        return self.directory(variant, stage) / ARTIFACT_NAMES[stage]

    def dependencies(self, variant, stage):
        chain = "wo_gate" if variant == "wo_gate" else "full"
        if stage == "fusion":
            return {"protein_embedding": self.protein, "dna_embedding": self.dna}
        deps = {"fusion": self.artifact(chain, "fusion")}
        if stage == "graph" or (stage in ("protein", "dna") and variant == "wo_graph"):
            deps["raw"] = self.artifact(chain, "raw")
        elif stage in ("protein", "dna"):
            deps["refined"] = self.artifact(chain, "graph")
        if stage in ("protein", "dna"):
            deps["single_modality_embedding"] = getattr(self, stage)
        return deps

    def identity(self, variant, stage):
        code = [SUITE / name for name in ("suite_runtime.py", "stage_runner.py", "fixed_split.py")]
        code += sorted((SUITE / "core").glob("*.py")) + [SUITE / "protein_parameters.csv"]
        return {"suite": "ARGScope-fixed80-v1", "split_fingerprint": EXPECTED_FINGERPRINT,
                "variant": variant, "stage": stage, "smoke_test_only": self.smoke_test,
                "code_sha256": {p.relative_to(SUITE).as_posix(): sha256(p) for p in code},
                "input_sha256": {key: sha256(path) for key, path in self.dependencies(variant, stage).items()}}


def load_base(stage, context, variant):
    core = SUITE / "core"
    sys.path.insert(0, str(core))
    import newsplit_contract as contract
    contract.configure(context.split_dir, "wo_gate" if variant == "wo_gate" else "full")
    name = f"argscope80_{variant}_{stage}"
    spec = importlib.util.spec_from_file_location(name, core / STAGES[stage])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_no_gate(module):
    def equal_weight_encode(self, protein, dna):
        protein, dna = self.standardize(protein, dna)
        protein_projected = self.protein_projection(protein)
        dna_projected = self.dna_projection(dna)
        gate = module.torch.full_like(protein_projected, .5)
        return self.fusion_norm(.5 * protein_projected + .5 * dna_projected), gate
    module.SupervisedGatedFusion.encode = equal_weight_encode
    original = module.contract.stamp_fusion

    def stamp(obj, *args):
        obj["fusion_method"] = "Subtype-supervised equal-weight fusion (w/o Adaptive Gating)"
        obj["fusion_formula"] = "LN(0.5*protein_projection + 0.5*DNA_projection)"
        return original(obj, *args)
    module.contract.stamp_fusion = stamp


def apply_no_graph(module, raw_path):
    original_load = module.load_pt
    original_validate = module.contract.validate_derived

    def load(path):
        obj = original_load(path)
        if Path(path).resolve() == raw_path.resolve():
            if obj.get("artifact_kind") != "raw":
                raise ValueError("w/o Graph requires new model-train raw prototypes")
            obj = dict(obj)
            obj["refined_arg_subtype_prototypes"] = obj["raw_arg_subtype_prototypes"]
        return obj

    def validate(obj, fused, fusion_path, kind):
        if kind == "refined":
            # Strictly require raw; do not silently accept a refined artifact.
            return original_validate(obj, fused, fusion_path, "raw")
        return original_validate(obj, fused, fusion_path, kind)
    module.load_pt, module.contract.validate_derived = load, validate


def apply_no_transfer(module):
    original_student = module.train_student

    def untrained_teacher(data, config, use_amp, hb, progress_dir=None):
        for field in TRANSFER_FIELDS:
            setattr(config, field, 0.)
        teacher = module.FusionTeacher(data["refined_gpu"], config).cuda()
        # Replace BOTH prototype-initialized heads with random weights.
        # The temporary teacher is not optimized; no learned head is copied.
        with module.torch.no_grad():
            module.nn.init.xavier_uniform_(teacher.binary_head.weight)
            module.nn.init.xavier_uniform_(teacher.arg_head.weight)
        teacher.requires_grad_(False)
        print("w/o Transfer: no Teacher training; all five transfer coefficients = 0.", flush=True)
        return teacher, [], 0, float("nan")

    def supervised_student(teacher, data, config, use_amp, hb, snapshot_epochs, snapshot_dir):
        for field in TRANSFER_FIELDS:
            setattr(config, field, 0.)
        return original_student(teacher, data, config, use_amp, hb, snapshot_epochs, snapshot_dir)
    module.train_teacher, module.train_student = untrained_teacher, supervised_student


def stamp_checkpoints(module, identity):
    name = "safe_save" if hasattr(module, "safe_save") else "save_checkpoint"
    if not hasattr(module, name):
        return  # Fusion uses its own writer; stage receipt also binds its hash.
    original = getattr(module, name)

    def save(obj, path):
        if isinstance(obj, dict):
            obj["experiment_provenance"] = identity
        return original(obj, path)
    setattr(module, name, save)


def check_complete(context, variant, stage, identity):
    directory = context.directory(variant, stage)
    marker = directory / "COMPLETE.json"
    if not marker.is_file():
        return False
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    if receipt.get("identity") != identity or receipt.get("status") != "complete":
        raise ValueError(f"Stale/different result, not reusable: {directory}")
    if receipt.get("artifact_name") != ARTIFACT_NAMES[stage] or not receipt.get("outputs"):
        raise ValueError(f"Incomplete receipt: {marker}")
    for name, expected in receipt["outputs"].items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or sha256(path) != expected:
            raise ValueError(f"Completed output changed: {path}")
    if stage in ("protein", "dna"):
        context.split.validate_predictions(directory / "test_sequence_predictions.csv")
        context.split.validate_student_manifest(directory / "split_manifest.csv")
    return True


def save_complete(context, variant, stage, identity):
    directory = context.directory(variant, stage)
    paths = [context.artifact(variant, stage)]
    if stage in ("protein", "dna"):
        path = directory / "test_sequence_predictions.csv"
        context.split.validate_predictions(path)
        context.split.validate_student_manifest(directory / "split_manifest.csv")
        paths.extend([path, directory / "split_manifest.csv"])
    receipt = {"status": "complete", "identity": identity, "artifact_name": ARTIFACT_NAMES[stage],
               "outputs": {p.name: sha256(p) for p in paths}}
    with (directory / "COMPLETE.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
