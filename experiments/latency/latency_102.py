import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Configuration
CSV_PATH = "LOGS_BASE_DIR /new_test_1_100/node_metrics_164428.csv"
TS_START = "2025-09-05T16:44:51.499495"
TS_END   = "2025-09-05T16:54:50.539215"

# Exclude nodes with fewer samples than this from the plot.
MIN_SAMPLES_FOR_CHART = 5

# If more nodes are found, keep only the top N by sample count.
ENFORCE_MAX_NODES = True
MAX_NODES_FOR_CHART = 102

# Plotting Appearance
FIG_SIZE = (12, 6)
DPI_PNG = 150
USE_SHADED_STD = True  # Use a shaded band for std. dev.
SHOW_FIG = True        # True: plt.show(); False: only save files
MAX_XTICK_LABELS = 12  # Max number of x-axis labels to show
MARKERSIZE = 3         # Marker size for the line plot

def parse_iso_ts(s: str) -> datetime:
    """Parses an ISO 8601 timestamp string."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)

def load_latency(csv_path: str, t0: datetime, t1: datetime) -> Dict[str, List[float]]:
    """Reads the CSV and returns a dict of node_address to its latency values."""
    latencies = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "node_address", "latency_ms"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"CSV is missing required columns: {required}")

        for row in reader:
            try:
                # Filter by time window
                ts = parse_iso_ts(row["timestamp"])
                if not (t0 <= ts <= t1):
                    continue

                node = (row.get("node_address") or "").strip()
                if not node:
                    continue

                # Filter out zero or negative latencies
                lat_ms = float((row.get("latency_ms") or "0").strip())
                if lat_ms > 0:
                    latencies[node].append(lat_ms)
            except (ValueError, TypeError):
                continue
    return latencies

def mean_std(vals: List[float]) -> Tuple[float, float]:
    """Returns the mean and sample standard deviation (std=0 for n<2)."""
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    m = sum(vals) / n
    if n >= 2:
        var = sum((x - m) ** 2 for x in vals) / (n - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return m, s

def plot_means_line_indexed(node_stats: Dict[str, Tuple[float, float]],
                            counts: Dict[str, int],
                            csv_path: str, t0: datetime, t1: datetime):
    """
    Plots a line chart of mean latencies with an indexed x-axis and exports a mapping file.
    """
    if not node_stats:
        print("No latency data to plot (or all data was filtered out).")
        return

    nodes = sorted(node_stats.keys())
    means = [node_stats[n][0] for n in nodes]
    stds  = [node_stats[n][1] for n in nodes]
    N = len(nodes)

    plt.figure(figsize=FIG_SIZE)
    
    xs = list(range(1, N + 1))
    
    # Calculate step to avoid overcrowded x-axis labels
    step = max(1, math.ceil(N / MAX_XTICK_LABELS))
    show_idx = list(range(0, N, step))

    # Plot line and shaded band; also thin out markers
    plt.plot(xs, means, marker='o', markersize=MARKERSIZE, markevery=step, linewidth=1)
    if USE_SHADED_STD:
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.fill_between(xs, lower, upper, alpha=0.2)

    # Set the thinned-out ticks and labels
    plt.xticks([xs[i] for i in show_idx], [str(xs[i]) for i in show_idx], rotation=0, ha="center")

    plt.xlim(0.5, N + 0.5)
    plt.xlabel("Node Index (sorted by address)")
    plt.ylabel("Mean Latency (ms)")
    plt.title(
        f"Per-Node Mean Latency (line with ±std band)\n"
        f"Window: {t0.isoformat()}  →  {t1.isoformat()}"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    # Save files to the same directory as the CSV
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = f"latency_summary_{t0.strftime('%H%M%S')}_{t1.strftime('%H%M%S')}_indexed_line"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    plt.savefig(png_path, dpi=DPI_PNG)
    plt.savefig(pdf_path)
    print(f"Saved figure:\n- {png_path}\n- {pdf_path}")

    # Export an index-to-node mapping CSV
    map_path = os.path.join(out_dir, base + "_index_map.csv")
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "node_address", "count", "mean_ms", "std_ms"])
        for i, n in enumerate(nodes, start=1):
            m, s = node_stats[n]
            w.writerow([i, n, counts.get(n, 0), f"{m:.3f}", f"{s:.3f}"])
    print(f"Saved index-to-node mapping CSV:\n- {map_path}")

    if SHOW_FIG:
        plt.show()
    else:
        plt.close()

def main():
    t0 = parse_iso_ts(TS_START)
    t1 = parse_iso_ts(TS_END)
    if t1 < t0:
        t0, t1 = t1, t0

    all_latencies = load_latency(CSV_PATH, t0, t1)

    # Print a full summary to the console
    print("node_address,count,mean_latency_ms,std_latency_ms")
    for node in sorted(all_latencies.keys()):
        m, s = mean_std(all_latencies[node])
        print(f"{node},{len(all_latencies[node])},{m:.3f},{s:.3f}")

    # For plotting: filter out nodes with too few samples
    dropped = {n: len(v) for n, v in all_latencies.items() if len(v) < MIN_SAMPLES_FOR_CHART}
    plot_input = {n: v for n, v in all_latencies.items() if len(v) >= MIN_SAMPLES_FOR_CHART}

    if dropped:
        print(f"\n# Dropped from chart (sample count < {MIN_SAMPLES_FOR_CHART}):")
        for n in sorted(dropped.keys()):
            print(f"{n}: {dropped[n]}")

    # If too many nodes, trim to the top N by sample count
    if ENFORCE_MAX_NODES and len(plot_input) > MAX_NODES_FOR_CHART:
        counts_all = {n: len(v) for n, v in plot_input.items()}
        sorted_nodes = sorted(plot_input.keys(), key=lambda n: (-counts_all[n], n))
        keep = set(sorted_nodes[:MAX_NODES_FOR_CHART])
        trimmed = [n for n in sorted_nodes[MAX_NODES_FOR_CHART:]]
        plot_input = {n: plot_input[n] for n in sorted(keep)}
        print(f"\n# Trimmed to top {MAX_NODES_FOR_CHART} nodes by sample count:")
        for n in trimmed:
            print(f"{n}: {counts_all[n]}")

    # Calculate stats for the nodes to be plotted
    counts_map = {n: len(v) for n, v in plot_input.items()}
    node_stats_for_plot = {n: mean_std(v) for n, v in plot_input.items()}

    # Plot the results using an indexed x-axis
    plot_means_line_indexed(node_stats_for_plot, counts_map, CSV_PATH, t0, t1)

if __name__ == "__main__":
    main()
