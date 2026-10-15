"""Preserved ARGScope architectures with explicit, audited partial-label supervision."""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bgc_normalized_data import NON, load_module, write_csv, write_json

AA = load_module("05_train_bgc_protein_student_local.py", "normalized_legacy_AA")
NT = load_module("06_train_bgc_dna_student_local.py", "normalized_legacy_NT")
FU = load_module("02_local_bgc_gated_fusion.py", "normalized_legacy_fusion")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_data(rows, names):
    name_to_index = {n: i for i, n in enumerate(names)}
    type_names = [NON] + sorted({r["type"] for r in rows if r["type"] != NON})
    type_to_index = {n: i for i, n in enumerate(type_names)}
    prototype_types = [n.split(" | ")[0] for n in names]
    proto_type_index = torch.tensor([type_to_index[t] for t in prototype_types])
    compatible = torch.stack([proto_type_index == i for i in range(len(type_names))])
    return {
        "rows": rows, "bgc_names": names, "type_names": type_names,
        "prototype_types": prototype_types, "prototype_type_indices": proto_type_index,
        "compatible": compatible,
        "binary_targets": torch.tensor([int(r["type"] != NON) for r in rows]),
        "bgc_targets": torch.tensor([name_to_index.get(r["subtype"], -1) for r in rows]),
        "type_targets": torch.tensor([type_to_index[r["type"]] for r in rows]),
        "split_indices": {s: torch.tensor([i for i, r in enumerate(rows) if r["model_split"] == s])
                          for s in ("train", "validation", "test")},
    }


def metrics_array(true, pred, count, per_class=False):
    true, pred = np.asarray(true, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    if len(true) == 0:
        raise ValueError("Empty evaluation cohort.")
    if min(true.min(), pred.min()) < 0 or max(true.max(), pred.max()) >= count:
        raise ValueError("Predictions/labels outside the predefined label set.")
    matrix = np.bincount(true * count + pred, minlength=count * count).reshape(count, count).astype(float)
    support, predicted_count, tp = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    precision = np.divide(tp, predicted_count, out=np.zeros(count), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(count), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(count), where=(precision + recall) != 0)
    n = len(true)
    denom = math.sqrt(max(0, n*n - float(predicted_count @ predicted_count)) * max(0, n*n - float(support @ support)))
    result = {"n": n, "accuracy": float(tp.sum()/n),
              "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
              "macro_f1": float(f1.mean()), "weighted_precision": float(precision @ support/n),
              "weighted_recall": float(recall @ support/n), "weighted_f1": float(f1 @ support/n),
              "mcc": float((tp.sum()*n - support @ predicted_count)/denom) if denom else 0.0}
    if per_class:
        result["per_class"] = [{"index": i, "support": int(support[i]), "predicted_count": int(predicted_count[i]),
                                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                               for i in range(count)]
        result["confusion_matrix"] = matrix.astype(int).tolist()
    return result


def cached_outputs(outputs, data):
    ix = outputs["indices"].cpu()
    fine = data["bgc_targets"][ix].numpy()
    binary = data["binary_targets"][ix].numpy()
    probability = torch.softmax(outputs["binary_logits"].float(), dim=1)[:, 1].numpy()
    pred_fine = outputs["bgc_logits"].argmax(dim=1).numpy()
    return {"indices": ix.numpy(), "binary": binary, "fine": fine,
            "type": data["type_targets"][ix].numpy(), "probability": probability,
            "pred_fine": pred_fine, "fine_evaluable": (binary == 0) | (fine >= 0)}


def evaluate_cached(cache, data, threshold, per_class=False):
    pred_binary = (cache["probability"] >= threshold).astype(np.int64)
    true_binary, fine, pred_fine = cache["binary"], cache["fine"], cache["pred_fine"]
    eligible, known = cache["fine_evaluable"], fine >= 0
    true_full = np.where(true_binary == 0, 0, fine + 1)
    pred_full = np.where(pred_binary == 0, 0, pred_fine + 1)
    types = data["prototype_type_indices"].numpy()
    pred_type = np.where(pred_binary == 0, 0, types[pred_fine])
    binary_metrics = metrics_array(true_binary, pred_binary, 2, per_class)
    tp = int(((true_binary == 1) & (pred_binary == 1)).sum())
    fp = int(((true_binary == 0) & (pred_binary == 1)).sum())
    fn = int(((true_binary == 1) & (pred_binary == 0)).sum())
    p, r = tp/max(1,tp+fp), tp/max(1,tp+fn)
    binary_metrics.update(positive_precision=p, positive_recall=r,
                          positive_f1=2*p*r/max(1e-12,p+r), TP=tp, FP=fp, FN=fn)
    metrics = {"binary": binary_metrics,
               "type": metrics_array(cache["type"], pred_type, len(data["type_names"]), per_class),
               "end_to_end_all_classes": metrics_array(true_full[eligible], pred_full[eligible], len(data["bgc_names"])+1, per_class),
               "BGC_subtype_oracle_binary": metrics_array(fine[known], pred_fine[known], len(data["bgc_names"]), per_class),
               "coverage": {"all_records": len(fine), "BGC_records": int(true_binary.sum()),
                            "fine_annotated_BGC_records": int(known.sum()),
                            "fine_BGC_annotation_coverage": float(known.sum()/max(1,true_binary.sum())),
                            "fine_end_to_end_records": int(eligible.sum())}}
    arrays = {"predicted_binary": pred_binary, "predicted_fine": pred_fine,
              "predicted_type": pred_type, "predicted_full": pred_full}
    return metrics, arrays


def calibrate(outputs, data, minimum_recall=.70):
    cache = cached_outputs(outputs, data)
    choices = []
    for integer in range(30,91):
        threshold = integer/100
        metrics, _ = evaluate_cached(cache, data, threshold)
        choices.append({"threshold": threshold, "score": AA.validation_score(metrics), "metrics": metrics})
    eligible = [x for x in choices if x["metrics"]["binary"]["positive_recall"] >= minimum_recall]
    best = max(eligible or choices, key=lambda x: (x["score"], x["metrics"]["binary"]["positive_precision"],
                                                   x["metrics"]["binary"]["accuracy"], -abs(x["threshold"]-.5)))
    return {**best, "minimum_recall_satisfied": bool(eligible)}


def partial_loss(logits, fine, types, compatible, weights=None, smoothing=.002):
    """One per-positive objective: exact CE if known, otherwise type-marginal NLL."""
    positive = types > 0
    known = (fine >= 0) & positive
    coarse = (fine < 0) & positive
    zero = logits.float().sum() * 0
    n = int(positive.sum())
    if n == 0:
        return zero
    loss = zero
    if known.any():
        loss = loss + F.cross_entropy(logits[known].float(), fine[known], weight=weights,
                                     label_smoothing=smoothing) * (known.sum()/n)
    if coarse.any():
        logprob = F.log_softmax(logits[coarse].float(), dim=1)
        allowed = compatible[types[coarse]]
        if not bool(allowed.any(dim=1).all()):
            raise ValueError("Coarse positive has no compatible fine class.")
        marginal = -torch.logsumexp(logprob.masked_fill(~allowed, -torch.inf), dim=1)
        loss = loss + marginal.mean() * (coarse.sum()/n)
    return loss


@torch.inference_mode()
def collect(model, inputs, indices, batch=512):
    model.eval()
    outputs = {"indices": [], "binary_logits": [], "bgc_logits": [], "features": []}
    for chunk in indices.split(batch):
        with torch.autocast("cuda", dtype=torch.float16):
            binary, fine, features = model(inputs[chunk.to(inputs.device)])
        for key, value in zip(outputs, (chunk, binary, fine, features)):
            outputs[key].append(value.detach().float().cpu() if key != "indices" else value.cpu())
    return {key: torch.cat(values) for key, values in outputs.items()}


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_fusion(protein, dna, data, seed, outdir, provenance):
    seed_all(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    tr, va = (data["split_indices"][s] for s in ("train", "validation"))
    pmean, pstd = FU.fit_standardizer(protein, tr)
    dmean, dstd = FU.fit_standardizer(dna, tr)
    pp = ((protein-pmean)/pstd).cuda()
    dd = ((dna-dmean)/dstd).cuda()
    names = data["bgc_names"]
    fine = torch.where(data["binary_targets"] == 0, 0, data["bgc_targets"]+1)
    known = (data["binary_targets"] == 0) | (data["bgc_targets"] >= 0)
    fine = torch.where(known, fine, -1)
    weights = FU.calculate_class_weights(fine[tr][known[tr]], len(names)+1).cuda()
    ft, types = fine.cuda(), data["type_targets"].cuda()
    compatible = F.pad(data["compatible"], (1,0)).cuda()
    compatible[0,0] = True
    model = FU.SupervisedGatedFusion(1280,768,512,len(names)+1,.20).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=4,min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    history, best, state, stale, best_epoch = [], -math.inf, None, 0, 0
    for epoch in range(1,101):
        model.train()
        total = torch.zeros((),device="cuda")
        order = tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
        for chunk in order.split(512):
            ix = chunk.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                _, logits, _ = model(pp[ix],dd[ix])
                target, kind = ft[ix],types[ix]
                labelled = target >= 0
                coarse = ~labelled
                loss = logits.float().sum()*0
                if labelled.any():
                    loss = loss+F.cross_entropy(logits[labelled].float(),target[labelled],weight=weights,label_smoothing=.05)*labelled.float().mean()
                if coarse.any():
                    lp = F.log_softmax(logits[coarse].float(),dim=1)
                    loss = loss-torch.logsumexp(lp.masked_fill(~compatible[kind[coarse]],-torch.inf),dim=1).mean()*coarse.float().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Fusion loss nonfinite.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()*len(chunk)
        model.eval()
        pred = []
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):
            for chunk in va.split(1024):
                pred.append(model(pp[chunk.cuda()],dd[chunk.cuda()])[1].argmax(1).cpu())
        pred = torch.cat(pred)
        mask = known[va]
        score = metrics_array(fine[va][mask].numpy(),pred[mask].numpy(),len(names)+1)["macro_f1"]
        scheduler.step(score)
        history.append({"epoch":epoch,"loss":float(total)/len(tr),"validation_fine_macro_f1":score,"lr":optimizer.param_groups[0]["lr"]})
        if score>best+1e-8:
            best,state,best_epoch,stale = score,cpu_state(model),epoch,0
        else:
            stale += 1
        if epoch%5==0 or epoch==1:
            print(f"Fusion seed={seed} epoch={epoch} val_macroF1={score:.4f} best={best_epoch}",flush=True)
        if stale>=12:
            break
    model.load_state_dict(state)
    model.eval()
    fused, gates = [], []
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for start in range(0,len(protein),1024):
            z,_,g = model(pp[start:start+1024],dd[start:start+1024])
            fused.append(z.float().cpu())
            gates.append(g.float().mean(1).cpu())
    fused = torch.cat(fused)
    payload = {"ids":[r["id"] for r in data["rows"]],"types":[r["type"] for r in data["rows"]],
               "subtypes":[r["subtype"] for r in data["rows"]],
               "model_splits":[r["model_split"] for r in data["rows"]],
               "splits":[r["external_split"] for r in data["rows"]],
               "embeddings":fused,"protein_gate":torch.cat(gates),"provenance":provenance,"seed":seed}
    AA.atomic_torch_save(payload,outdir/"long_fusion_full.pt")
    graph_indices = torch.where(known)[0]
    graph = {k:[v[i] for i in graph_indices.tolist()] for k,v in payload.items() if k in ("ids","types","subtypes","model_splits","splits")}
    graph.update(embeddings=fused[graph_indices],provenance=provenance,seed=seed)
    AA.atomic_torch_save(graph,outdir/"graph_input_fusion.pt")
    AA.atomic_torch_save({"state_dict":state,"protein_mean":pmean,"protein_std":pstd,"dna_mean":dmean,"dna_std":dstd,
                          "class_names":[NON]+names,"seed":seed,"best_epoch":best_epoch,"provenance":provenance},outdir/"fusion_checkpoint.pt")
    write_csv(outdir/"fusion_history.csv",history)
    write_json(outdir/"fusion_summary.json",{"seed":seed,"best_epoch":best_epoch,"validation_score":best,"epochs_ran":len(history)})
    del pp,dd,model,optimizer
    torch.cuda.empty_cache()
    return fused


class LinearProbe(nn.Module):
    def __init__(self,mean,std,count):
        super().__init__()
        self.register_buffer("mean",mean)
        self.register_buffer("std",std)
        self.binary_head = nn.Linear(len(mean),2)
        self.bgc_head = nn.Linear(len(mean),count)

    def forward(self,features):
        x=(features-self.mean)/self.std
        return self.binary_head(x),self.bgc_head(x),x


def train_stage(stage, data, inputs, fusion, module, config, outdir, teacher=None, linear=False):
    device=torch.device("cuda")
    tr,va=(data["split_indices"][s] for s in ("train","validation"))
    is_teacher=stage=="teacher"
    if is_teacher:
        model=module.FusionTeacher(data["refined"],config).cuda()
        module.initialize_binary_centers(model,fusion.cpu(),data["binary_targets"],tr)
        maximum,patience,lr=config.teacher_epochs,config.teacher_patience,config.teacher_lr
        power=config.teacher_class_weight_power
    else:
        mean=inputs[tr.cuda()].mean(0)
        std=inputs[tr.cuda()].std(0,unbiased=False).clamp_min(1e-6)
        if linear:
            model=LinearProbe(mean,std,len(data["bgc_names"])).cuda()
        elif module is AA:
            model=module.ProteinStudent(mean,std,len(data["bgc_names"]),config).cuda()
        else:
            model=module.DNAStudent(mean,std,len(data["bgc_names"]),config).cuda()
        if not linear:
            with torch.no_grad():
                model.binary_head.weight.copy_(teacher.binary_head.weight)
                model.bgc_head.weight.copy_(teacher.bgc_head.weight)
        maximum,patience,lr=config.student_epochs,config.student_patience,config.student_lr
        if linear:
            maximum,patience,lr=200,25,1e-3
        power=config.class_weight_power
    train_fine=data["bgc_targets"][tr]
    weights=module.mild_class_weights(train_fine[train_fine>=0],len(data["bgc_names"]),power).cuda()
    binary_y,fine_y,type_y=(data[k].cuda() for k in ("binary_targets","bgc_targets","type_targets"))
    refined=data["refined"].cuda()
    centroid=data["nonbgc_centroid"].cuda()
    compatible=data["compatible"].cuda()
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        t_binary=teacher.binary_head.normalized_weight().detach()
        t_fine=teacher.bgc_head.normalized_weight().detach()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=5 if is_teacher else 7,min_lr=1e-6)
    scaler=torch.amp.GradScaler("cuda")
    best_key=(-math.inf,-math.inf,-math.inf)
    best_epoch,stale,state,best_calibration=0,0,None,None
    history=[]
    for epoch in range(1,maximum+1):
        begun=time.perf_counter()
        model.train()
        total=torch.zeros((),device=device)
        offsets=0 if is_teacher else 100000
        order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(config.seed+offsets+epoch))]
        for chunk in order.split(config.batch_size):
            ix=chunk.cuda()
            b,f,t=binary_y[ix],fine_y[ix],type_y[ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                binary,fine,representation=model(inputs[ix])
                binary_ce=F.cross_entropy(binary.float(),b,label_smoothing=config.label_smoothing)
                subtype_ce=partial_loss(fine,f,t,compatible,weights,config.label_smoothing)
                if is_teacher:
                    anchor=(1-(model.bgc_head.normalized_weight()*refined).sum(1)).mean()
                    loss=config.teacher_binary_ce*binary_ce+config.teacher_subtype_ce*subtype_ce+config.teacher_refined_anchor*anchor
                elif linear:
                    loss=binary_ce+subtype_ce
                else:
                    with torch.no_grad():
                        tb,tf,trep=teacher(fusion[ix])
                    known=f>=0
                    subtype_kd=module.kd_loss(fine[known],tf[known],config.temperature) if known.any() else fine.float().sum()*0
                    binary_kd=module.kd_loss(binary,tb,config.temperature)
                    feature=module.cosine_pull(representation,trep)
                    eligible=known|(b==0)
                    target=centroid.expand(len(ix),-1).clone()
                    if known.any():
                        target[known]=refined[f[known]]
                    proto=module.cosine_pull(representation[eligible],target[eligible]) if eligible.any() else feature*0
                    head=(1-(model.binary_head.normalized_weight()*t_binary).sum(1)).mean()+(1-(model.bgc_head.normalized_weight()*t_fine).sum(1)).mean()
                    loss=(config.student_binary_ce*binary_ce+config.student_subtype_ce*subtype_ce+
                          config.student_binary_kd*binary_kd+config.student_subtype_kd*subtype_kd+
                          config.student_feature_align*feature+config.student_prototype_align*proto+config.student_head_anchor*head)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {stage} loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            total+=loss.detach()*len(chunk)
        outputs=collect(model,inputs,va,config.eval_batch_size)
        calibration=calibrate(outputs,data,config.minimum_bgc_recall)
        score=calibration["score"]
        scheduler.step(score)
        full=calibration["metrics"]["end_to_end_all_classes"]
        key=(score,full["weighted_precision"],full["accuracy"])
        improved=score>best_key[0]+config.min_delta
        if not is_teacher:
            improved=improved or (abs(score-best_key[0])<=config.min_delta and key[1]>best_key[1]+1e-8) or (
                abs(score-best_key[0])<=config.min_delta and abs(key[1]-best_key[1])<=1e-8 and key[2]>best_key[2]+1e-8)
        if improved:
            best_key,best_epoch,state,best_calibration,stale=key,epoch,cpu_state(model),copy.deepcopy(calibration),0
            # Recovery checkpoints contain the validation-selected state only.
            module.atomic_torch_save({"state_dict":state,"epoch":epoch,"config":asdict(config),"threshold":calibration["threshold"],
                                      "validation_score":score,"label_names":data["bgc_names"],"type_names":data["type_names"]},outdir/f"{stage}_checkpoint.pt")
        else:
            stale+=1
        binary_metrics=calibration["metrics"]["binary"]
        history.append({"epoch":epoch,"loss":float(total)/len(tr),"validation_score":score,"threshold":calibration["threshold"],
                        "validation_binary_P":binary_metrics["positive_precision"],"validation_binary_R":binary_metrics["positive_recall"],
                        "validation_subtype_ACC":full["accuracy"],"validation_type_macroF1":calibration["metrics"]["type"]["macro_f1"],
                        "lr":optimizer.param_groups[0]["lr"],"epoch_seconds":time.perf_counter()-begun})
        if epoch==1 or epoch%5==0 or stale>=patience:
            print(f"{outdir.name} {stage} seed={config.seed} epoch={epoch} val_score={score:.4f} subtype_ACC={full['accuracy']:.4f} best={best_epoch} seconds={history[-1]['epoch_seconds']:.1f}",flush=True)
            write_csv(outdir/f"{stage}_history.csv",history)
        if stale>=patience:
            break
    model.load_state_dict(state)
    model.eval()
    write_csv(outdir/f"{stage}_history.csv",history)
    summary={"stage":stage,"seed":config.seed,"selected_epoch":best_epoch,"epochs_ran":len(history),"validation_score":best_key[0],
             "threshold":best_calibration["threshold"],"minimum_validation_recall_satisfied":best_calibration["minimum_recall_satisfied"],
             "config":asdict(config),"selection_rule":"Original validation weighted-precision score and threshold grid; fine-labelled cohort only for fine metrics."}
    write_json(outdir/f"{stage}_training_summary.json",summary)
    del optimizer
    return model,summary


def save_evaluation(model,inputs,data,threshold,outdir,tag):
    test=data["split_indices"]["test"]
    outputs=collect(model,inputs,test)
    cache=cached_outputs(outputs,data)
    metrics,arrays=evaluate_cached(cache,data,threshold,per_class=True)
    write_json(outdir/f"{tag}_test_metrics.json",metrics)
    rows=[]
    for j,ix in enumerate(cache["indices"]):
        r=data["rows"][ix]
        pred_binary=int(arrays["predicted_binary"][j])
        rows.append({"id":r["id"],"group_id":r["group_id"],"true_binary":int(cache["binary"][j]),
                     "true_type":r["type"],"true_subtype":r["subtype"],"fine_evaluable":int(cache["fine_evaluable"][j]),
                     "pred_binary":pred_binary,"pred_type":data["type_names"][arrays["predicted_type"][j]],
                     "pred_subtype":data["bgc_names"][arrays["predicted_fine"][j]] if pred_binary else NON,
                     "BGC_probability":float(cache["probability"][j]),"binary_threshold":threshold})
    write_csv(outdir/f"{tag}_test_predictions.csv",rows)
    for kind,names in (("type",data["type_names"]),("end_to_end_all_classes",[NON]+data["bgc_names"])):
        write_csv(outdir/f"{tag}_{kind}_per_class.csv",[{"label":names[c["index"]],**c} for c in metrics[kind]["per_class"]])
    return metrics


def run_branch(branch,features,fused,data,seed,outdir,linear=True):
    module=AA if branch=="AA" else NT
    config=module.Config(seed=seed)
    outdir.mkdir(parents=True,exist_ok=True)
    seed_all(seed)
    inputs=features.cuda()
    fusion=fused.cuda()
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    teacher,teacher_summary=train_stage("teacher",data,fusion,fusion,module,config,outdir)
    student,student_summary=train_stage("student",data,inputs,fusion,module,config,outdir,teacher=teacher)
    # Both model/threshold choices are final before any test metrics are calculated.
    metrics=save_evaluation(student,inputs,data,student_summary["threshold"],outdir,"student")
    teacher_metrics=save_evaluation(teacher,fusion,data,teacher_summary["threshold"],outdir,"teacher")
    summary={"branch":branch,"seed":seed,"teacher_training":teacher_summary,"student_training":student_summary,
             "student_test_metrics":metrics,"teacher_test_metrics":teacher_metrics,
             "elapsed_seconds":time.perf_counter()-started,"peak_allocated_GPU_GB":torch.cuda.max_memory_allocated()/1024**3}
    del student,teacher
    if linear:
        seed_all(seed)
        probe,probe_summary=train_stage("linear_probe",data,inputs,fusion,module,config,outdir,linear=True)
        summary["linear_probe_training"]=probe_summary
        summary["linear_probe_test_metrics"]=save_evaluation(probe,inputs,data,probe_summary["threshold"],outdir,"linear_probe")
        del probe
    write_json(outdir/"run_summary.json",summary)
    del inputs,fusion
    torch.cuda.empty_cache()
    return summary


def self_test():
    # Marginal loss must reward either compatible fine class and reject the other type.
    logits=torch.tensor([[2.,1.,-2.],[0.,1.,3.]],requires_grad=True)
    compatible=torch.tensor([[False,False,False],[True,True,False],[False,False,True]])
    loss=partial_loss(logits,torch.tensor([-1,2]),torch.tensor([1,2]),compatible,smoothing=0)
    expected=(-torch.logsumexp(F.log_softmax(logits,1)[0,:2],0)+F.cross_entropy(logits[1:],torch.tensor([2])))/2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # Negative rows do not become coarse positives or fine-class targets.
    assert partial_loss(logits.detach(),torch.tensor([-1,-1]),torch.tensor([0,0]),compatible).item()==0
    metric=metrics_array([0,1],[0,1],3)
    assert metric["macro_recall"]==2/3 and metric["weighted_f1"]==1
    print("Partial-label and predefined-class metric self-tests passed.",flush=True)
