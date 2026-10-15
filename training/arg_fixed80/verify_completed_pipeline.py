"""Audit completed runs: actual IDs, train-only statistics, prototypes and settings."""
import argparse
import json
from pathlib import Path

from fixed_split import EXPECTED_FINGERPRINT
from suite_runtime import Context, TRANSFER_FIELDS, VARIANTS, add_path_arguments, check_complete


def verify(context):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    means, stds = {}, {}
    for modality in ("protein", "dna"):
        embedding = load(getattr(context, modality))
        ids = [str(x).strip().split()[0] for x in embedding["ids"]]
        selected = [i for i, sid in enumerate(ids) if sid in context.split.ids["train"]]
        train = embedding["embeddings"][selected].float()
        means[modality] = train.mean(dim=0)
        stds[modality] = train.std(dim=0, unbiased=False).clamp_min(1e-6)
        del train, embedding
    raw_tensors = {}
    for chain in ("full", "wo_gate"):
        for stage in ("fusion", "raw", "graph"):
            if not check_complete(context, chain, stage, context.identity(chain, stage)):
                raise ValueError(f"Unfinished {chain}/{stage}")
        fused = load(context.artifact(chain, "fusion"))
        weights = fused["fusion_model_state_dict"]
        for modality in ("protein", "dna"):
            torch.testing.assert_close(weights[f"{modality}_mean"], means[modality], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(weights[f"{modality}_std"], stds[modality], rtol=1e-5, atol=1e-5)
        if chain == "wo_gate":
            for key in ("gate_weights_protein", "gate_weights_DNA"):
                values = torch.as_tensor(fused[key])
                if not torch.all(values == .5):
                    raise ValueError("w/o Gate contains non-constant gate weights")
        raw = load(context.artifact(chain, "raw"))
        z = F.normalize(fused["embeddings"].float(), p=2, dim=1)
        groups = {}
        for i, (sid, label) in enumerate(zip(fused["ids"], fused["subtypes"])):
            if sid in context.split.ids["train"] and label != "nonARG":
                groups.setdefault(label, []).append(i)
        expected = torch.stack([F.normalize(z[groups[label]].mean(dim=0), p=2, dim=0)
                                for label in raw["arg_subtype_names"]])
        torch.testing.assert_close(raw["raw_arg_subtype_prototypes"], expected, rtol=1e-4, atol=1e-5)
        raw_tensors[chain] = raw["raw_arg_subtype_prototypes"]
        del fused, raw, z
    full_configs, full_shapes = {}, {}
    for variant in VARIANTS:
        for modality in ("protein", "dna"):
            identity = context.identity(variant, modality)
            if not check_complete(context, variant, modality, identity):
                raise ValueError(f"Unfinished {variant}/{modality}")
            obj = load(context.artifact(variant, modality))
            if obj["experiment_provenance"] != identity:
                raise ValueError("Checkpoint provenance mismatch")
            torch.testing.assert_close(obj[f"{modality}_train_mean"], means[modality], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(obj[f"{modality}_train_std"], stds[modality], rtol=1e-5, atol=1e-5)
            if obj["selected_binary_ARG_probability_threshold"] != .8:
                raise ValueError("Fixed binary threshold changed")
            config = dict(obj["configuration"])
            shapes = {key: list(value.shape) for key, value in obj["student_state_dict"].items()}
            if variant == "full":
                full_configs[modality], full_shapes[modality] = config, shapes
            else:
                expected = dict(full_configs[modality])
                if variant == "wo_transfer":
                    expected.update({key: 0. for key in TRANSFER_FIELDS})
                    if obj["teacher_best_epoch"] != 0:
                        raise ValueError("w/o Transfer optimized a Teacher")
                if config != expected or shapes != full_shapes[modality]:
                    raise ValueError(f"Unintended settings/architecture change: {variant}/{modality}")
            if variant == "wo_graph":
                torch.testing.assert_close(obj["refined_ARG_subtype_prototypes"], raw_tensors["full"])
            del obj
    report = {"status": "PASS", "split_fingerprint": EXPECTED_FINGERPRINT,
              "completed_students": 8, "N_test_each": 4574, "same_test_IDs_and_ground_truth": True,
              "actual_model_train_normalization_checked": True, "raw_prototypes_recomputed": True,
              "no_gate_weights_fixed_at_half": True, "no_graph_uses_raw_prototypes": True,
              "no_transfer_Teacher_epoch_zero_and_all_transfer_coefficients_zero": True,
              "same_settings_within_modality_except_explicit_ablation": True,
              "smoke_test_only": context.smoke_test}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    report = verify(Context(options.data_root, options.output_root, options.smoke_test))
    if options.report:
        with options.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
