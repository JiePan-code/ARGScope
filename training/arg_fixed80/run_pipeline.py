"""Run new Full and/or ablations serially. Verified completed stages can be reused."""
import argparse
import datetime
import os
import subprocess
import sys

from suite_runtime import Context, SUITE, VARIANTS, add_path_arguments


def plan(variants, modality="both"):
    stages = []
    for variant in variants:
        chain = "wo_gate" if variant == "wo_gate" else "full"
        upstream = ["fusion", "raw"] if variant == "wo_graph" else ["fusion", "raw", "graph"]
        requested = [(chain, s) for s in upstream]
        requested += [(variant, s) for s in (("protein", "dna") if modality == "both" else (modality,))]
        for item in requested:
            if item not in stages:
                stages.append(item)
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all",) + VARIANTS, default="all")
    parser.add_argument("--modality", choices=("both", "protein", "dna"), default="both")
    parser.add_argument("--preflight-only", action="store_true")
    add_path_arguments(parser)
    options = parser.parse_args()
    context = Context(options.data_root, options.output_root, options.smoke_test)
    from preflight import preflight
    preflight(context)
    if options.preflight_only:
        return
    variants = VARIANTS if options.variant == "all" else (options.variant,)
    tasks = plan(variants, options.modality)
    print("Execution plan: " + " -> ".join(f"{v}/{s}" for v, s in tasks), flush=True)
    logs = context.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for variant, stage in tasks:
        command = [sys.executable, "-u", str(SUITE / "stage_runner.py"), variant, stage,
                   "--data-root", str(context.data_root), "--output-root", str(context.output_root)]
        if options.smoke_test:
            command.append("--smoke-test")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = logs / f"{variant}_{stage}_{timestamp}.log"
        with path.open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", env=env) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code != 0:
            raise RuntimeError(f"Stopped at {variant}/{stage}, exit={code}. See {path}")
    print(f"Requested stages complete. Results: {context.output_root}")
    print("After all eight models finish, run 05_summarize_results.bat.")


if __name__ == "__main__":
    main()
