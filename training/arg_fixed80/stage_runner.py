"""One stage per fresh process, to isolate ablation hooks and GPU memory."""
import argparse
import json
import sys
from types import SimpleNamespace

from suite_runtime import (Context, STAGES, SUITE, VARIANTS, add_path_arguments, apply_no_gate,
                           apply_no_graph, apply_no_transfer, check_complete, load_base,
                           save_complete, stamp_checkpoints)


def execute(context, variant, stage):
    if variant in ("wo_graph", "wo_transfer") and stage not in ("protein", "dna"):
        raise ValueError("These ablations reuse the new Full upstream stages")
    identity = context.identity(variant, stage)
    output = context.directory(variant, stage)
    if check_complete(context, variant, stage, identity):
        print(f"VERIFIED COMPLETE; skip {variant}/{stage}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete/unrecognized results: {output}\n"
                              "Nothing was deleted. Rename this stage folder to keep it, then rerun.")
    m = load_base(stage, context, variant)
    args = []
    deps = context.dependencies(variant, stage)
    if stage == "fusion":
        args = ["--protein-pt", context.protein, "--dna-pt", context.dna,
                "--output", context.artifact(variant, stage)]
        if variant == "wo_gate":
            apply_no_gate(m)
    elif stage == "raw":
        args = ["--input-pt", deps["fusion"], "--output-pt", context.artifact(variant, stage)]
    elif stage == "graph":
        args = ["--raw-pt", deps["raw"], "--fused-pt", deps["fusion"],
                "--output-pt", context.artifact(variant, stage)]
    else:
        prototype = deps["raw"] if variant == "wo_graph" else deps["refined"]
        args = [f"--{stage}-pt", getattr(context, stage), "--fusion-pt", deps["fusion"],
                "--refined-pt", prototype, "--output-dir", output]
        if stage == "protein":
            args += ["--best-params", SUITE / "protein_parameters.csv",
                     "--teacher-checkpoint", output / "OLD_TEACHER_FORBIDDEN.pt"]
            # This old interface argument is never loaded. Fail even if a future
            # base edit accidentally tries to load a selected historical teacher.
            def reject_old_teacher(*_args, **_kwargs):
                raise RuntimeError("Historical Teacher checkpoint loading is forbidden")
            m.load_selected_fusion_teacher = reject_old_teacher
        if variant == "wo_graph":
            apply_no_graph(m, prototype)
        if variant == "wo_transfer":
            apply_no_transfer(m)
        # These entrypoints never authorize deleting any existing directory.
        def reject_delete(*_args, **_kwargs):
            raise RuntimeError("Recursive deletion is disabled by the 80:20 suite")
        m.shutil = SimpleNamespace(**(vars(m.shutil) | {"rmtree": reject_delete}))
    if stage in ("fusion", "protein", "dna"):
        args += ["--train-ids", context.split_dir / "train_ids_subtype2.txt",
                 "--test-ids", context.split_dir / "test_ids_subtype2.txt"]
    if stage != "raw":
        args += ["--structure", context.structure]
    if context.smoke_test:
        if stage in ("fusion", "graph"):
            args += ["--epochs", 1]
        if stage == "graph":
            args += ["--k-candidates", 5]
        if stage in ("protein", "dna"):
            args += ["--snapshot-epochs", 0, 1]
            if stage == "dna":
                args += ["--teacher-epochs", 1, "--student-epochs", 1]
            else:
                original = m.load_grid_best_parameters

                def smoke_parameters(path, config):
                    config, meta, row = original(path, config)
                    config.teacher_epochs = config.student_epochs = 1
                    return config, meta, row
                m.load_grid_best_parameters = smoke_parameters
    stamp_checkpoints(m, identity)
    sys.argv = [STAGES[stage]] + [str(a) for a in args]
    print(f"RUN {variant}/{stage} | fixed test=4574 | smoke_only={context.smoke_test}", flush=True)
    m.main()
    if stage in ("fusion", "raw", "graph"):
        obj = m.torch.load(context.artifact(variant, stage), map_location="cpu", weights_only=False)
        if stage == "fusion":
            m.contract.validate_fusion(obj)
        else:
            fused = m.torch.load(deps["fusion"], map_location="cpu", weights_only=False)
            m.contract.validate_derived(obj, fused, deps["fusion"], "raw" if stage == "raw" else "refined")
    save_complete(context, variant, stage, identity)
    print(f"COMPLETE {variant}/{stage}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("stage", choices=STAGES)
    add_path_arguments(parser)
    options = parser.parse_args()
    execute(Context(options.data_root, options.output_root, options.smoke_test), options.variant, options.stage)
