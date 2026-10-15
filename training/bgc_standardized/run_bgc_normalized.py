"""Three fixed-split seeds. Outputs are isolated from all original experiments."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from bgc_normalized_data import OUT, SOURCE, prepare, load_features, sha256, write_csv, write_json
from train_partial_labels import AA, fit_fusion, setup_data, run_branch, self_test


def stage_script(filename,arguments,log):
    print(f"Running {filename}, log={log}",flush=True)
    environment=os.environ.copy()
    environment["PYTHONUTF8"]="1"
    environment["OMP_NUM_THREADS"]="4"
    with log.open("w",encoding="utf-8") as handle:
        subprocess.run([sys.executable,"-u","-X","utf8",str(SOURCE/filename),*map(str,arguments)],
                       stdout=handle,stderr=subprocess.STDOUT,env=environment,check=True)


def aggregate():
    summaries=[]
    for seed in (20260915,20261006,20261007):
        for branch in ("AA","NT"):
            path=OUT/f"seed_{seed}"/branch/"run_summary.json"
            if path.exists():
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rows=[]
    for s in summaries:
        for method,key in (("ARGScope","student_test_metrics"),("Linear probe","linear_probe_test_metrics")):
            if key not in s:
                continue
            m=s[key]
            for task,fields in (("binary",("accuracy","positive_precision","positive_recall","positive_f1","mcc")),
                                ("type",("accuracy","macro_precision","macro_recall","macro_f1","mcc")),
                                ("subtype",("accuracy","weighted_precision","weighted_recall","weighted_f1","mcc"))):
                record=m["end_to_end_all_classes" if task=="subtype" else task]
                for metric in fields:
                    rows.append({"method":method,"branch":s["branch"],"seed":s["seed"],"task":task,"metric":metric,
                                 "value":record[metric],"evaluation_records":record["n"]})
    if rows:
        write_csv(OUT/"Results/all_seed_metrics.csv",rows)
        averaged=[]
        keys=sorted({(r["method"],r["branch"],r["task"],r["metric"]) for r in rows})
        for key in keys:
            subset=[r for r in rows if (r["method"],r["branch"],r["task"],r["metric"])==key]
            values=[r["value"] for r in subset]
            averaged.append({"method":key[0],"branch":key[1],"task":key[2],"metric":key[3],"seeds_completed":len(values),
                             "mean":statistics.mean(values),"sample_std":statistics.stdev(values) if len(values)>1 else "",
                             "evaluation_records":subset[0]["evaluation_records"]})
        write_csv(OUT/"Results/mean_std_metrics.csv",averaged)
    write_json(OUT/"Results/completion_status.json",{"completed_branch_runs":len(summaries),"expected_branch_runs":6,
                                                      "complete":len(summaries)==6,"test_access":"Evaluation after validation selection; no test-selected seed."})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[20260915,20261006,20261007])
    parser.add_argument("--self-test-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(4)
    self_test()
    if args.self_test_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This full experiment requires the verified CUDA environment.")
    rows,manifest=prepare()
    features=load_features(rows)
    provenance={"dataset_manifest":manifest,"feature_sha256":{
        "protein":sha256(SOURCE/"ESM2_BGC/BGC_nonBGC_esm2_mean.pt"),
        "DNA":sha256(SOURCE/"DNABERT2_BGC/BGC_nonBGC_dnabert2_mean.pt")},
        "legacy_source_sha256":{p.name:sha256(p) for p in SOURCE.glob("0[23456]_*.py")},
        "new_source_sha256":{p.name:sha256(p) for p in OUT.glob("*.py")},
        "runtime":{"python":sys.version,"torch":torch.__version__,"CUDA":torch.version.cuda,
                   "GPU":torch.cuda.get_device_name(),"platform":platform.platform()},
        "seeds":[20260915,20261006,20261007],
        "changes":["Normalized cluster-origin labels and fine-label masks.",
                   "Type-marginal NLL for positive records without a usable fine label.",
                   "No fine prototype alignment/KD for unlabelled positives.",
                   "Fixed predefined macro-class set and explicit fine-evaluation coverage."],
        "preserved":["All sequence IDs and group-disjoint assignments.","Frozen encoder features.",
                     "Fusion/GAT/branch-specific teacher/student architectures and original loss coefficients.",
                     "Original patience, epoch limits, learning rates and validation selection/threshold rules."]}
    provenance_path=OUT/"experiment_provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text(encoding="utf-8"))!=provenance:
            raise RuntimeError("Provenance drift detected. Use a new experiment directory.")
    else:
        write_json(provenance_path,provenance)
    for seed in args.seeds:
        seed_root=OUT/f"seed_{seed}"
        seed_root.mkdir(parents=True,exist_ok=True)
        fusion_dir=seed_root/"Fusion"
        fused_path=fusion_dir/"long_fusion_full.pt"
        data=setup_data(rows,manifest["fine_classes"])
        if fused_path.exists():
            obj=torch.load(fused_path,map_location="cpu",weights_only=False)
            if obj["provenance"]!=provenance or obj["seed"]!=seed:
                raise RuntimeError("Cached fusion belongs to different labels/split/runtime.")
            fused=obj["embeddings"]
        else:
            fused=fit_fusion(*features,data,seed,fusion_dir,provenance)
        graph_dir=seed_root/"Graph"
        graph_dir.mkdir(exist_ok=True)
        raw=graph_dir/"raw_prototypes.pt"
        refined=graph_dir/"refined_prototypes.pt"
        if not raw.exists():
            stage_script("03_build_raw_bgc_prototypes.py",["--input-pt",fusion_dir/"graph_input_fusion.pt","--output-pt",raw,"--device","cuda"],graph_dir/"raw_prototypes.log")
        if not refined.exists():
            stage_script("04_refine_bgc_prototypes_gat.py",["--raw-pt",raw,"--fusion-pt",fusion_dir/"graph_input_fusion.pt",
                         "--output-pt",refined,"--seed",seed,"--device","cuda"],graph_dir/"graph_training.log")
        obj=torch.load(refined,map_location="cpu",weights_only=False)
        data=setup_data(rows,obj["prototype_subtypes"])
        data["refined"]=F.normalize(obj["refined_prototypes"].float(),dim=1)
        data["nonbgc_centroid"]=F.normalize(obj["nonbgc_reference_centroid"].float().reshape(-1),dim=0)
        for branch,feature in zip(("AA","NT"),features):
            result=seed_root/branch/"run_summary.json"
            if result.exists():
                print(f"Already completed {seed} {branch}; preserved.",flush=True)
            else:
                run_branch(branch,feature,fused,data,seed,seed_root/branch)
                aggregate()
        write_json(seed_root/"completion.json",{"seed":seed,"complete":True,"selected_graph_k":obj["selected_k"]})
    # Recheck immutable labels, split and legacy source hashes before declaring completion.
    if sha256(SOURCE/"BGC_corrected_files/corrected_BGC_nonBGC_id_type_subtype.csv")!=manifest["source_labels_sha256"]:
        raise RuntimeError("Original labels unexpectedly changed.")
    if sha256(SOURCE/"Long_fusion/model_split_metadata.csv")!=manifest["source_fixed_split_sha256"]:
        raise RuntimeError("Original split unexpectedly changed.")
    for name,digest in provenance["legacy_source_sha256"].items():
        if sha256(SOURCE/name)!=digest:
            raise RuntimeError("Original training code unexpectedly changed.")
    aggregate()
    print("REQUESTED RUNS COMPLETED. Aggregate results:",OUT/"Results/mean_std_metrics.csv",flush=True)


if __name__=="__main__":
    main()
