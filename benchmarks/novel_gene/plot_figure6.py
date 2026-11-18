"""Render the five panels of Figure 6 from the audited result counts."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
MODELS = ("ARGScope", "DeepARG")
THRESHOLDS = ("0.50", "0.80")
METRICS = ("ARG/non-ARG", "Type assignment")
BLUE = LinearSegmentedColormap.from_list("paper_blue", ("#ffffff", "#d9e8f3"))
INK = "#171e27"
RULE = "#dfe7ed"
PURPLE = "#736295"
PINK = {"B": "#d8a1b4", "C": "#bf3e70"}
COUNTS = ("N_ARG", "N_non_ARG", "TP", "FN", "TN", "FP", "correct_type")
UNITS = {"A": 16, "B": 1, "C": 1, "D": 45, "E": 30}


def load_results(path):
    panels = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            panel = row["panel"]
            slot = (int(row["row_order"]), row["threshold"], row["model"])
            if slot in panels[panel]:
                raise ValueError(f"Duplicate result for panel {panel}, {slot}")
            for field in COUNTS:
                row[field] = int(row[field]) if row[field] else None
            if row["TP"] + row["FN"] != row["N_ARG"] or row["TN"] + row["FP"] != row["N_non_ARG"]:
                raise ValueError(f"Inconsistent support for panel {panel}, {slot}")
            if panel != "A" and not 0 <= row["correct_type"] <= row["TP"]:
                raise ValueError(f"Inconsistent type assignments for panel {panel}, {slot}")
            panels[panel][slot] = row
    if set(panels) != set(UNITS):
        raise ValueError("Expected panels A through E")
    for panel, units in UNITS.items():
        expected = {(i, t, m) for i in range(1, units + 1) for t in THRESHOLDS for m in MODELS}
        if panels[panel].keys() != expected:
            raise ValueError(f"Incomplete rows for panel {panel}")
    for summary, detail in (("B", "D"), ("C", "E")):
        for threshold in THRESHOLDS:
            for model in MODELS:
                total = panels[summary][1, threshold, model]
                block = [panels[detail][i, threshold, model] for i in range(1, UNITS[detail] + 1)]
                if any(total[key] != sum(row[key] for row in block) for key in COUNTS):
                    raise ValueError(f"Panel {summary} does not pool counts from panel {detail}")
    return panels


def value(row, metric):
    numerator = row["TP"] if metric == METRICS[0] else row["correct_type"]
    return numerator / row["N_ARG"]


def printed(rate):
    return str(int(rate)) if rate in (0, 1) else f"{rate:.4f}"


def label(fig, panel, x, y):
    fig.text(x, y, panel, fontsize=17, fontweight="bold", va="top")


def add_colorbar(fig, box):
    ax = fig.add_axes(box)
    normal = matplotlib.colors.Normalize(0, 1)
    mapper = matplotlib.cm.ScalarMappable(norm=normal, cmap=BLUE)
    bar = fig.colorbar(mapper, cax=ax, ticks=[0, .5, 1])
    bar.ax.set_yticklabels(("0", "0.5", "1"))
    bar.ax.tick_params(length=2, width=.4, pad=2, labelsize=7)
    bar.outline.set_edgecolor("#a7b7c5")
    bar.outline.set_linewidth(.55)


def heatmap(fig, panel, rows, box, colorbar):
    is_type = panel == "A"
    metrics = METRICS[:1] if is_type else METRICS
    columns = [(metric, threshold, model) for metric in metrics
               for threshold in THRESHOLDS for model in MODELS]
    x, y, w, h = box
    ax = fig.add_axes(box)
    n_rows, n_cols = UNITS[panel], len(columns)
    ax.set(xlim=(0, n_cols), ylim=(n_rows, 0))
    ax.axis("off")
    for i in range(1, n_rows + 1):
        name = rows[i, THRESHOLDS[0], MODELS[0]]["held_out_label"]
        ax.text(-.035 if is_type else -.08, i - .5, name, va="center",
                ha="right", fontsize=6.8 if is_type else 6.55)
        for j, (metric, threshold, model) in enumerate(columns):
            rate = value(rows[i, threshold, model], metric)
            ax.add_patch(Rectangle((j, i - 1), 1, 1, color=BLUE(rate), linewidth=0))
            ax.text(j + .5, i - .5, printed(rate), ha="center", va="center",
                    fontsize=6.85 if is_type else 6.55)
    for j, (_, _, model) in enumerate(columns):
        header = model if is_type else ("AS" if model == "ARGScope" else "DA")
        fig.text(x + w * (j + .5) / n_cols, y+h+.007, header, ha="center", fontsize=7.1, weight="bold")
    fig.text(x-.003, y+h+.007, "Resistance Type" if is_type else "Subtype",
             ha="right", fontsize=7.2, weight="bold")
    for j in range(n_cols // 2):
        fig.text(x + w * (2*j + 1) / n_cols, y+h+.025, THRESHOLDS[j % 2],
                 ha="center", fontsize=7.8, weight="bold")
    for j, metric in enumerate(metrics):
        fig.text(x + w * (4*j + 2) / n_cols, y+h+.045, metric,
                 ha="center", fontsize=9.1, weight="bold")
    ax.plot([0, n_cols], [n_rows, n_rows], color=RULE, lw=.85, clip_on=False)
    add_colorbar(fig, colorbar)


def bars(fig, panel, rows, y):
    color = PINK[panel]
    for midpoint, text in ((.68, METRICS[0]), (2.13, METRICS[1])):
        fig.text(.454, y + .098 * (1 - (midpoint + .16) / 2.98),
                 text, fontsize=8, va="center")
    for column, threshold in enumerate(THRESHOLDS):
        x = .557 + column * .207
        ax = fig.add_axes((x, y, .190, .098))
        ax.set(xlim=(0, 1), ylim=(2.82, -.16),
               xticks=(0, .5, 1), xticklabels=("0", "0.5", "1"), yticks=())
        ax.grid(axis="x", color=RULE, linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(threshold, fontsize=8.4, weight="bold", pad=5)
        for j, metric in enumerate(METRICS):
            for k, model in enumerate(MODELS):
                rate = value(rows[1, threshold, model], metric)
                center = .42 + 1.45*j + .52*k
                if model == "DeepARG":
                    ax.barh(center, rate, height=.48, color=PURPLE, hatch="////",
                            edgecolor="#cfc2dc", linewidth=.25)
                else:
                    ax.barh(center, rate, height=.48, color=color)
                inside = rate >= .36
                ax.text(rate + (-.023 if inside else .019), center, f"{rate:.3f}",
                        ha="right" if inside else "left", va="center", weight="bold",
                        fontsize=7.8, color="#15191d")


def figure(panels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "xtick.color": INK, "hatch.linewidth": .25,
                         "pdf.fonttype": 42, "svg.fonttype": "path"})
    fig = plt.figure(figsize=(12.4, 11.8), facecolor="white")
    for panel, x, y in (("A",.016,.985), ("B",.454,.985), ("C",.454,.807),
                        ("D",.016,.616), ("E",.504,.616)):
        label(fig, panel, x, y)
    heatmap(fig, "A", panels["A"], (.138, .673, .276, .229),
            (.421, .673, .008, .229))
    fig.legend(handles=[Patch(facecolor=PINK["C"], label="AS  ARGScope"),
                        Patch(facecolor=PURPLE, edgecolor="#cfc2dc", hatch="////",
                              linewidth=.25, label="DA  DeepARG")],
               loc="upper center", bbox_to_anchor=(.699,.977), ncol=2,
               frameon=False, handlelength=1.1, handletextpad=.4, columnspacing=6,
               fontsize=8)
    bars(fig, "B", panels["B"], .837)
    bars(fig, "C", panels["C"], .654)
    heatmap(fig, "D", panels["D"], (.159,.026,.282,.514),
            (.464,.026,.008,.514))
    heatmap(fig, "E", panels["E"], (.632,.026,.301,.514),
            (.957,.026,.008,.514))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "figure6_results.csv")
    parser.add_argument("--output", type=Path, default=Path("Figure_6.pdf"),
                        help="Output path ending in .pdf, .svg or .png")
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".pdf", ".svg", ".png"}:
        parser.error("output must end in .pdf, .svg or .png")
    fig = figure(load_results(args.results))
    fig.savefig(args.output, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
