import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Configuration
CSV_PATH = "LOGS_BASE_DIR /new_huge_10_10mins_load20/node_metrics_171253.csv"
TS_START = "2025-09-05T17:13:01.377394"
TS_END   = "2025-09-05T17:23:01.365027"
# Plotting: nodes with fewer samples than this are excluded.
MIN_SAMPLES_FOR_CHART = 5
FIG_SIZE = (12, 6)
DPI_PNG = 150

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

def plot_latency_summary(node_stats: Dict[str, Tuple[float, float]],
                         counts_map: Dict[str, int],
                         csv_path: str, t0: datetime, t1: datetime):
    """
    Plots a bar chart of mean latencies with an indexed x-axis
    and exports a mapping file from index to node address.
    """
    if not node_stats:
        print("No latency data to plot (or all data was filtered out).")
        return

    nodes = sorted(node_stats.keys())
    means = [node_stats[n][0] for n in nodes]
    stds  = [node_stats[n][1] for n in nodes]

    plt.figure(figsize=FIG_SIZE)
    
    xs = list(range(1, len(nodes) + 1))
    plt.bar(xs, means, yerr=stds, capsize=4)
    plt.xticks(xs, xs, rotation=0, ha="center")

    plt.xlabel("Node Index (1 to N)")
    plt.ylabel("Mean Latency (ms)")
    plt.title(
        f"Per-Node Mean Latency (with Std. Dev. Error Bars)\n"
        f"Window: {t0.isoformat()}  →  {t1.isoformat()}"
    )
    plt.tight_layout()

    # Prepare output directory and filenames
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = f"latency_summary_{t0.strftime('%H%M%S')}_{t1.strftime('%H%M%S')}"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    
    plt.savefig(png_path, dpi=DPI_PNG)
    plt.savefig(pdf_path)
    print(f"Saved figure:\n- {png_path}\n- {pdf_path}")

    # Export a mapping CSV from index to node address
    map_path = os.path.join(out_dir, base + "_index_map.csv")
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "node_address", "count", "mean_ms", "std_ms"])
        for i, node_addr in enumerate(nodes, start=1):
            mean, std = node_stats[node_addr]
            count = counts_map.get(node_addr, 0)
            w.writerow([i, node_addr, count, f"{mean:.3f}", f"{std:.3f}"])
    print(f"Saved index-to-node mapping CSV:\n- {map_path}")

    plt.show()

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

    # Filter out nodes with too few samples for plotting
    dropped = {n: len(v) for n, v in all_latencies.items() if len(v) < MIN_SAMPLES_FOR_CHART}
    plot_input = {n: v for n, v in all_latencies.items() if len(v) >= MIN_SAMPLES_FOR_CHART}

    if dropped:
        print(f"\n# Dropped from chart (sample count < {MIN_SAMPLES_FOR_CHART}):")
        for n in sorted(dropped.keys()):
            print(f"{n}: {dropped[n]}")

    # Calculate stats for the nodes to be plotted
    counts_map = {n: len(v) for n, v in plot_input.items()}
    node_stats_for_plot = {n: mean_std(v) for n, v in plot_input.items()}

    # Plot the results
    plot_latency_summary(node_stats_for_plot, counts_map, CSV_PATH, t0, t1)

if __name__ == "__main__":
    main()
