import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Configuration
CSV_PATH = "LOGS_BASE_DIR /new_medium_44_10mins_load20/node_metrics_145141.csv"
TS_START = "2025-09-05T14:51:52.313178"
TS_END   = "2025-09-05T15:01:51.498434"

MIN_SAMPLES      = 5    # Drop nodes with < MIN_SAMPLES samples
ANNOTATE_TOP_K   = 5    # Annotate top-K std outliers (0 to disable)

# Visual Parameters
MARKER_SIZE      = 32   # Marker size for all points
ALPHA            = 0.6  # Point transparency
LINEWIDTH        = 0.4  # Marker edge width
JITTER_FRACTION  = 0.005  # Jitter amount as a fraction of axis range
RNG_SEED         = 2025   # Fixed random seed for reproducible jitter

def parse_iso_ts(s: str) -> datetime:
    """Parses an ISO 8601 timestamp string."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)

def load_latency(csv_path: str, t0: datetime, t1: datetime) -> Dict[str, List[float]]:
    """Reads CSV and returns a dict of node_address to its latency values."""
    latencies = defaultdict(list)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "node_address", "latency_ms"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(f"CSV missing columns: {required}; found: {reader.fieldnames}")

        for row in reader:
            try:
                # Filter by time window
                ts = parse_iso_ts(row["timestamp"])
                if not (t0 <= ts <= t1):
                    continue

                node = (row.get("node_address") or "").strip()
                if not node:
                    continue
                
                # Exclude zeros and non-finite values
                lat_ms = float((row.get("latency_ms") or "0").strip())
                if lat_ms > 0 and math.isfinite(lat_ms):
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

def summarize_per_node(lat: Dict[str, List[float]], min_samples: int):
    """Compute per-node stats (count, mean, std) and filter by min_samples."""
    stats = {}
    dropped = {}
    for node, vals in lat.items():
        cnt = len(vals)
        if cnt < min_samples:
            dropped[node] = cnt
            continue
        m, s = mean_std(vals)
        stats[node] = (cnt, m, s)
    return stats, dropped

def plot_scatter_mean_std(stats: Dict[str, Tuple[int, float, float]],
                          csv_path: str, t0: datetime, t1: datetime,
                          annotate_top_k: int):
    """Plots a scatter chart of mean vs. std with jitter to separate overlapping points."""
    if not stats:
        print("No nodes passed the MIN_SAMPLES filter — scatter plot skipped.")
        return

    # Prepare data for plotting
    nodes  = list(stats.keys())
    means  = np.array([stats[n][1] for n in nodes], dtype=float)
    stds   = np.array([stats[n][2] for n in nodes], dtype=float)

    # Add jitter to prevent points from overlapping
    rng = np.random.default_rng(RNG_SEED)
    xr = (means.max() - means.min()) or 1.0
    yr = (stds.max()  - stds.min())  or 1.0
    xj = means + rng.normal(0, xr * JITTER_FRACTION, size=len(means))
    yj = stds  + rng.normal(0, yr * JITTER_FRACTION, size=len(stds))

    plt.figure(figsize=(10, 6))
    plt.scatter(xj, yj, s=MARKER_SIZE, alpha=ALPHA, linewidths=LINEWIDTH)

    # Annotate top-K outliers by standard deviation
    if annotate_top_k > 0:
        top_idx = np.argsort(stds)[::-1][:annotate_top_k]
        for i in top_idx:
            label = nodes[i][-6:]  # Show last 6 chars of address
            plt.annotate(label, (xj[i], yj[i]),
                         xytext=(5, 5), textcoords="offset points", fontsize=9)

    plt.xlabel("Mean latency (ms)")
    plt.ylabel("Std latency (ms)")
    plt.title(
        "Per-node mean vs std (zeros excluded, n≥{})\n"
        "All markers same size | Window: {} → {}".format(
            MIN_SAMPLES, t0.isoformat(), t1.isoformat()
        )
    )
    plt.grid(True, axis="both", linestyle=":", alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = f"latency_scatter_{t0.strftime('%H%M%S')}_{t1.strftime('%H%M%S')}"
    png_path = os.path.join(out_dir, base + ".png")
    pdf_path = os.path.join(out_dir, base + ".pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    print(f"Saved scatter plot:\n- {png_path}\n- {pdf_path}")
    plt.show()

def summarize(lat: Dict[str, List[float]], min_samples: int):
    """Prints a summary table of node statistics."""
    stats, dropped = summarize_per_node(lat, min_samples)
    print("node_address,count,mean_latency_ms,std_latency_ms")
    for node in sorted(stats.keys()):
        cnt, m, s = stats[node]
        print(f"{node},{cnt},{m:.3f},{s:.3f}")
    if dropped:
        print(f"\n# Dropped nodes (sample count < {min_samples}):")
        for node in sorted(dropped.keys()):
            print(f"{node}: {dropped[node]}")
    return stats

def main():
    t0 = parse_iso_ts(TS_START)
    t1 = parse_iso_ts(TS_END)
    if t1 < t0:
        t0, t1 = t1, t0

    latencies = load_latency(CSV_PATH, t0, t1)
    stats = summarize(latencies, MIN_SAMPLES)

    # Plot the scatter chart
    plot_scatter_mean_std(stats, CSV_PATH, t0, t1, ANNOTATE_TOP_K)

if __name__ == "__main__":
    main()
