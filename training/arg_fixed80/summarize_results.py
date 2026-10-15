"""Only summarize new, completed, identical-split Full + ablation results."""
import argparse
import csv
import datetime
import json

from suite_runtime import Context, VARIANTS, add_path_arguments, check_complete
from fixed_split import EXPECTED_FINGERPRINT


def five_metrics(truth, prediction, average):
    from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average=average, zero_division=0)
    return {"Accuracy": accuracy_score(truth, prediction), "Precision": precision,
            "Recall": recall, "F1": f1, "MCC": matthews_corrcoef(truth, prediction)}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(context):
    binary, types, subtypes, binary_averages = [], [], [], []
    for variant in VARIANTS:
        for modality in ("protein", "dna"):
            identity = context.identity(variant, modality)
            if not check_complete(context, variant, modality, identity):
                raise FileNotFoundError(f"Not complete: {variant}/{modality}. Finish all eight models first.")
            rows = context.split.validate_predictions(context.directory(variant, modality) / "test_sequence_predictions.csv")
            info = {"Model": variant, "Modality": modality, "N_test": len(rows)}
            y = [int(r["true_ARG"]) for r in rows]
            p = [int(r["predicted_ARG"]) for r in rows]
            binary.append({**info, "Average": "binary_ARG_positive", **five_metrics(y, p, "binary")})
            for average in ("macro", "weighted", "micro"):
                binary_averages.append({**info, "Average": average, **five_metrics(y, p, average)})
                for level, destination in (("type", types), ("subtype", subtypes)):
                    truth = [r[f"true_{level}"] for r in rows]
                    prediction = [r[f"predicted_{level}"] for r in rows]
                    destination.append({**info, "Average": average, **five_metrics(truth, prediction, average)})
    prefix = "SMOKE_NOT_FOR_PAPER_" if context.smoke_test else "metrics_"
    output = context.output_root / (prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "01_ARG_nonARG_ARG_positive_five_metrics.csv", binary)
    write_csv(output / "02_Type_overall_macro_weighted_micro.csv", types)
    write_csv(output / "03_subtype_overall_macro_weighted_micro.csv", subtypes)
    write_csv(output / "04_ARG_nonARG_macro_weighted_micro_optional.csv", binary_averages)
    combined = [{"Level": level, **r} for level, selected in
                (("ARG_nonARG", binary), ("Type", types), ("Subtype", subtypes)) for r in selected]
    write_csv(output / "00_all_models_summary.csv", combined)
    report = {"split_fingerprint": EXPECTED_FINGERPRINT, "models": 8, "N_per_model": 4574,
              "type_and_subtype_include_nonARG": True, "ARG_is_positive": True,
              "Accuracy_and_MCC": "global single-label metrics, repeated unchanged across averaging rows",
              "zero_division": 0, "old_Full_results_used": False, "smoke_test_only": context.smoke_test}
    (output / "calculation_notes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"All eight NEW models summarized: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    options = parser.parse_args()
    summarize(Context(options.data_root, options.output_root, options.smoke_test))
