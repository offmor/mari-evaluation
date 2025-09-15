import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Configuration (change your CSV paths here)
# CSVs must contain: index,node_address,count,mean_ms,std_ms
PATHS: Dict[str, Path] = {
    "tiny":   LOGS_BASE_DIR / "new_tiny_10_10mins_load20/latency_summary_175341_180341_idx_index_map.csv",
    "medium": LOGS_BASE_DIR / "new_medium_44_10mins_load20/latency_summary_145152_150151_idx_line_index_map.csv",
    "big":    LOGS_BASE_DIR / "new_big_66_10mins_load20/latency_summary_153145_154145_idx_line_index_map.csv",
    "huge":   LOGS_BASE_DIR / "new_test_2_100/latency_summary_165658_170657_idx_line_index_map.csv",
}

# Image and Export Settings
FIG_SIZE = (8.2, 4.2)
DPI_PNG  = 300
SAVE_PDF = True
X_LABEL  = "Schedule"
Y_LABEL  = "Latency (ms)"
TITLE    = ""           # Keep empty for papers; caption is used instead

# Bar Chart Appearance
BAR_WIDTH = 0.6
BAR_COLOR = "tab:orange"
SHOW_VALUE_LABELS = False
VALUE_FMT = "{:.2f}"
XTICK_SHOW_NODES = False
TOP_MARGIN_RATIO = 0.10
NO_LABEL_MARGIN_RATIO = 0.06
SHOW_LEGEND = True
LEGEND_OUTSIDE = True

OUT_BASENAME = "overall_latency_by_schedule"

@dataclass
class NodeStat:
    count: int
    mean_ms: float
    std_ms: float

def read_index_map_csv(path: str) -> List[NodeStat]:
    """Reads the index map CSV and returns a list of NodeStat objects."""
    rows: List[NodeStat] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        def norm(s: str) -> str:
            return (s or "").strip().lower()
        colmap = {norm(c): c for c in (reader.fieldnames or [])}
        need = ["count", "mean_ms", "std_ms"]
        for k in need:
            if k not in colmap:
                raise RuntimeError(f"{path} is missing column `{k}`")
        for row in reader:
            try:
                c = int(str(row[colmap["count"]]).strip())
                m = float(str(row[colmap["mean_ms"]]).strip())
                s = float(str(row[colmap["std_ms"]]).strip())
            except Exception:
                continue
            if c <= 0 or not math.isfinite(m) or not math.isfinite(s):
                continue
            rows.append(NodeStat(count=c, mean_ms=m, std_ms=s))
    if not rows:
        raise RuntimeError(f"{path} contains no valid data.")
    return rows

def pooled_mean_std(stats: List[NodeStat]) -> Tuple[float, float, int, int]:
    """Calculates the pooled mean and standard deviation."""
    K = len(stats)
    N = sum(st.count for st in stats)
    if N <= 1:
        return (stats[0].mean_ms if K else 0.0, 0.0, N, K)
    
    M = sum(st.count * st.mean_ms for st in stats) / N
    
    SS_within  = sum((st.count - 1) * (st.std_ms ** 2) for st in stats if st.count > 1)
    SS_between = sum(st.count * ((st.mean_ms - M) ** 2) for st in stats)
    
    var = (SS_within + SS_between) / (N - 1)
    std = math.sqrt(var)
    return M, std, N, K

def summarize_all(paths: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Processes all CSVs and returns a summary dictionary."""
    summary: Dict[str, Dict[str, float]] = {}
    for label, p in paths.items():
        stats = read_index_map_csv(p)
        M, S, N, K = pooled_mean_std(stats)
        summary[label] = {
            "mean_ms": M,
            "std_ms": S,
            "total_samples": N,
            "num_nodes": K,
        }
    return summary

def save_summary_csv(summary: Dict[str, Dict[str, float]], out_dir: str, basename: str) -> str:
    """Saves the summary data to a new CSV file."""
    path = os.path.join(out_dir, basename + "_summary.csv")
    labels = ["tiny", "medium", "big", "huge"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "mean_ms", "std_ms", "num_nodes", "total_samples"])
        for lbl in labels:
            if lbl in summary:
                s = summary[lbl]
                w.writerow([lbl, f"{s['mean_ms']:.3f}", f"{s['std_ms']:.3f}", s["num_nodes"], s["total_samples"]])
    return path

def plot_summary(summary: Dict[str, Dict[str, float]], out_dir: str, basename: str):
    """Generates and saves the summary bar chart."""
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    labels = [lbl for lbl in ["tiny", "medium", "big", "huge"] if lbl in summary]
    x = list(range(len(labels)))
    means = [summary[l]["mean_ms"] for l in labels]
    stds  = [summary[l]["std_ms"]  for l in labels]

    xticklabels = [f"{l}\n({summary[l]['num_nodes']} nodes)" for l in labels] if XTICK_SHOW_NODES else labels

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bars = ax.bar(
        x, means, yerr=stds, width=BAR_WIDTH, capsize=3,
        linewidth=0.8, edgecolor="black", color=BAR_COLOR,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"}
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    if TITLE:
        ax.set_title(TITLE)

    y_max = max((m + s) for m, s in zip(means, stds)) if stds else max(means)
    margin = TOP_MARGIN_RATIO if SHOW_VALUE_LABELS else NO_LABEL_MARGIN_RATIO
    ax.set_ylim(0, y_max * (1 + margin))

    if SHOW_VALUE_LABELS:
        pad = y_max * 0.02
        for rect, m in zip(bars, means):
            ax.text(rect.get_x() + rect.get_width() / 2.0,
                    rect.get_height() + pad,
                    VALUE_FMT.format(m),
                    ha="center", va="bottom", fontsize=10)

    if SHOW_LEGEND:
        avg_patch = Patch(facecolor=BAR_COLOR, edgecolor="black", label="Average")
        std_line = Line2D([0], [0], linestyle='-', linewidth=1.0, color="black", label="Std. Dev.")
        if LEGEND_OUTSIDE:
            ax.legend(handles=[avg_patch, std_line], loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
        else:
            ax.legend(handles=[avg_patch, std_line], loc="upper right", frameon=False)

    fig.tight_layout()

    # Save figures
    png_path = os.path.join(out_dir, basename + ".png")
    fig.savefig(png_path, dpi=DPI_PNG, bbox_inches="tight")
    print(f"Saved figure (PNG): {png_path}")

    if SAVE_PDF:
        pdf_path = os.path.join(out_dir, basename + ".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved figure (PDF): {pdf_path}")

    plt.show()
    plt.close(fig)

def main():
    first_csv = next(iter(PATHS.values()))
    if not os.path.isfile(first_csv):
        raise FileNotFoundError(f"First CSV not found: {first_csv}")
    out_dir = os.path.dirname(os.path.abspath(first_csv))
    os.makedirs(out_dir, exist_ok=True)

    for k, v in PATHS.items():
        if not os.path.isfile(v):
            raise FileNotFoundError(f"{k} CSV not found: {v}")

    summary = summarize_all(PATHS)
    csv_out = save_summary_csv(summary, out_dir, OUT_BASENAME)
    print(f"Saved summary CSV: {csv_out}")

    plot_summary(summary, out_dir, OUT_BASENAME)

if __name__ == "__main__":
    main()
