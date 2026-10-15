"""Check the REAL fixed split and original embedding inputs without training."""
import argparse
import json

from suite_runtime import Context, add_path_arguments
from fixed_split import EXPECTED_FINGERPRINT


def preflight(context):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run with D:\\Anaconda\\envs\\dnabert2_5060\\python.exe")
    print(f"CUDA: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}", flush=True)
    for path, dimension in ((context.protein, 1280), (context.dna, 768)):
        # Only trusted, locally supplied PyTorch files are accepted.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "ids" not in obj or "embeddings" not in obj:
            raise ValueError(f"Embedding requires ids and embeddings: {path}")
        ids = [str(x).strip().split()[0] for x in obj["ids"]]
        tensor = obj["embeddings"]
        if len(ids) != len(set(ids)) or tensor.shape != (len(ids), dimension):
            raise ValueError(f"Duplicate IDs / unexpected embedding dimensions: {path}")
        if not set(context.split.by_id) <= set(ids) or not torch.isfinite(tensor).all():
            raise ValueError(f"Missing IDs / non-finite embeddings: {path}")
        if any(key in obj for key in ("fusion_model_state_dict", "model_splits", "refined_arg_subtype_prototypes")):
            raise ValueError("A learned downstream artifact was supplied as an original embedding")
        print(f"PASS {path.name}: {tuple(tensor.shape)}; all fixed-population IDs present", flush=True)
        del obj, tensor
    print(json.dumps({"external_train": 18296, "model_train": 14636, "validation": 3660, "test": 4574,
                      "ARG_subtypes_in_model_train_and_test": 820, "ARG_types": 23,
                      "ID_and_exact_sequence_overlap": 0, "split_fingerprint": EXPECTED_FINGERPRINT,
                      "mode": "SMOKE ONLY" if context.smoke_test else "FORMAL TRAINING",
                      "output_root": str(context.output_root)}, indent=2), flush=True)
    print("PREFLIGHT PASS (no model training performed by this check)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    options = parser.parse_args()
    preflight(Context(options.data_root, options.output_root, options.smoke_test))
