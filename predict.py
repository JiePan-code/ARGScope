"""ARGScope fixed80 Full inference from precomputed embeddings.

Release wrapper added 2026-10-15. The original architecture is in student_models.py.
This is not an embedding extractor and does not retrain or tune the model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from student_models import DNAStudent, ProteinStudent

FULL_SHA256 = {
    "protein": "a7ce4f09fbca42c67b11a7133a716814465d29176ae5dfb2fff987320785b8bc",
    "dna": "229bc847bc2438fa21a1bd3539f0d3318723b39853d36edeb9ba57aca468d222",
}
SPLIT = "6d2d3cf7ad252f255dedf4e4bbeae693f5ea0c5112f39daec331aaa975d93991"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path, modality, device):
    if sha256(path) != FULL_SHA256[modality]:
        raise ValueError("Checkpoint is not the byte-verified fixed80 Full model for this modality. Consult MODEL_INDEX.csv for other experiment versions.")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["experiment_provenance"]["split_fingerprint"] != SPLIT:
        raise ValueError("Unexpected checkpoint split provenance")
    config = SimpleNamespace(**ckpt["configuration"])
    dim = 1280 if modality == "protein" else 768
    mean, std = ckpt[modality + "_train_mean"], ckpt[modality + "_train_std"]
    if mean.shape != (dim,) or std.shape != (dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid saved model-training normalization")
    labels = ckpt["arg_subtype_names"]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate subtype labels")
    cls = ProteinStudent if modality == "protein" else DNAStudent
    model = cls(dim, mean, std, len(labels), config)
    model.load_state_dict(ckpt["student_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    threshold = float(ckpt["selected_binary_ARG_probability_threshold"])
    if threshold != .8:
        raise ValueError("Unexpected saved decision threshold")
    return model, ckpt, int(config.eval_batch_size), threshold


@torch.inference_mode()
def annotate(model, checkpoint, embeddings, ids, batch_size, threshold, device):
    dim = model.protein_mean.numel() if hasattr(model, "protein_mean") else model.dna_mean.numel()
    if embeddings.ndim != 2 or tuple(embeddings.shape) != (len(ids), dim):
        raise ValueError(f"Expected embeddings of shape (N, {dim})")
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Embedding IDs must be unique, nonempty strings")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite input embeddings")
    binary, subtype = [], []
    for begin in range(0, len(ids), batch_size):
        x = embeddings[begin:begin + batch_size].float().to(device)
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                b, s, _ = model(x)
        else:
            b, s, _ = model(x)
        binary.append(b.float().cpu())
        subtype.append(s.float().cpu())
    arg_prob = torch.softmax(torch.cat(binary), dim=1)[:, 1]
    sub_prob, sub_idx = torch.softmax(torch.cat(subtype), dim=1).max(dim=1)
    rows = []
    for i, sid in enumerate(ids):
        label = checkpoint["arg_subtype_names"][int(sub_idx[i])]
        is_arg = int(arg_prob[i] >= threshold)
        rows.append({"sequence_id": sid, "predicted_ARG": is_arg,
                     "predicted_subtype": label if is_arg else "nonARG",
                     "predicted_type": checkpoint["subtype_to_type"][label] if is_arg else "nonARG",
                     "ARG_probability": float(arg_prob[i]),
                     "ARG_subtype_max_probability": float(sub_prob[i]),
                     "ARG_subtype_before_binary_gate": label,
                     "binary_threshold": threshold})
    return rows


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("protein", "dna"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Trusted .pt dictionary with 'ids' (list[str]) and 'embeddings' (N x 1280 or N x 768)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing predictions are never overwritten")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for the unverified CPU reporting path")
    configure_runtime()
    device = torch.device(args.device)
    model, ckpt, bs, threshold = load_model(args.checkpoint, args.modality, device)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        bs = args.batch_size
    data = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    rows = annotate(model, ckpt, data["embeddings"], data["ids"], bs, threshold, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Annotated {len(rows)} sequences -> {args.output}")


if __name__ == "__main__":
    main()
