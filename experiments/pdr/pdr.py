import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plotting Scale Parameters
ZOOM_Y_MODE  = "fixed"        # "fixed" or "auto"
ZOOM_Y_FIXED = (0.95, 1.01)   # y-axis range for 'fixed' mode
ZOOM_X_WINDOW = None          # e.g., (2.0, 8.0) to limit x-axis in minutes; None for no limit

# Configuration
CONFIG = [
    {
        "label": "20 nodes",
        "path": "LOGS_BASE_DIR /PDR_20_huge_10mins_edge/gateway_metrics_171659.csv",
        "start": "2025-09-11T17:17:04.168130",
        "end":   "2025-09-11T17:27:04.232283",
        "multi_gw": False,
    },
    {
        "label": "40 nodes",
        "path": "LOGS_BASE_DIR /PDR_40_huge_10mins_edge/gateway_metrics_173832.csv",
        "start": "2025-09-11T17:38:36.392375",
        "end":   "2025-09-11T17:48:36.234710",
        "multi_gw": False,
    },
    {
        "label": "60 nodes",
        "path": "LOGS_BASE_DIR /new_60/gateway_metrics_143924.csv",
        "start": "2025-09-12T14:39:46.057679",
        "end":   "2025-09-12T14:49:46.568881",
        "multi_gw": False,
    },
    {
        "label": "80 nodes",
        "path": "LOGS_BASE_DIR /new_80/gateway_metrics_151837.csv",
        "start": "2025-09-12T15:18:50.894943",
        "end":   "2025-09-12T15:28:50.123697",
        "multi_gw": False,
    },
    {
        "label": "100 nodes",
        "path": "LOGS_BASE_DIR /new_100/gateway_metrics_145836.csv",
        "start": "2025-09-12T14:58:41.678612",
        "end":   "2025-09-12T15:08:41.816574",
        "multi_gw": False,
    },
    # 200 nodes: single file, multi-gateway
    {
        "label": "200 nodes",
        "path": "LOGS_BASE_DIR /200_10mins_2gateway/run_20250911_192037/gateway_metrics_192037.csv",
        "start": "2025-09-11T19:28:30.212169",
        "end":   "2025-09-11T19:38:30.782289",
        "multi_gw": True,
    },
]

# Column names
COL_TS      = "timestamp"
COL_NODE_TX = "latest_node_tx_count"
COL_NODE_RX = "latest_node_rx_count"
COL_GW_TX   = "latest_gw_tx_count"
COL_GW_RX   = "latest_gw_rx_count"
REQUIRED_COLS = {COL_TS, COL_NODE_TX, COL_NODE_RX, COL_GW_TX, COL_GW_RX}

GW_COL_CANDIDATES = ["gateway_address", "gw_address", "gateway", "gw_addr", "gatewayAddr", "gw"]


@dataclass
class SeriesResult:
    label: str
    t_minutes: np.ndarray
    pdr_overall: np.ndarray
    final_pdr: float
    file: str


def parse_ts(ts: Union[str, float, int, pd.Timestamp]) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return ts
    if isinstance(ts, (int, float)):
        return pd.to_datetime(ts, unit="s")
    return pd.to_datetime(ts)


def read_and_prepare_csv(path: str | Path) -> pd.DataFrame:
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(fpath)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df[COL_TS] = pd.to_datetime(df[COL_TS], errors="coerce")
    df = df.dropna(subset=[COL_TS]).sort_values(COL_TS).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"CSV {path} has no valid timestamp data.")
    return df


def detect_gateway_column(df: pd.DataFrame) -> Optional[str]:
    for c in GW_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def last_row_leq(df: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    df_le = df[df[COL_TS] <= t]
    return df_le.iloc[-1] if not df_le.empty else df.iloc[0]


def dedup_by_ts(df: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate by timestamp, keeping the last entry for each timestamp."""
    return df.sort_values(COL_TS).drop_duplicates(subset=[COL_TS], keep="last").reset_index(drop=True)


def compute_overall_pdr_series_single_gw(path: str, start, end) -> Tuple[np.ndarray, np.ndarray, float]:
    df = read_and_prepare_csv(path)
    start_ts = parse_ts(start)
    end_ts   = parse_ts(end)

    base = last_row_leq(df, start_ts)
    win  = df[(df[COL_TS] >= start_ts) & (df[COL_TS] <= end_ts)].copy()
    if win.empty:
        return np.array([]), np.array([]), float("nan")

    win = dedup_by_ts(win)

    t_minutes = (win[COL_TS] - start_ts).dt.total_seconds().to_numpy() / 60.0

    d_node_tx = (win[COL_NODE_TX].to_numpy(dtype="int64") - int(base[COL_NODE_TX]))
    d_gw_rx   = (win[COL_GW_RX].to_numpy(dtype="int64")   - int(base[COL_GW_RX]))
    d_gw_tx   = (win[COL_GW_TX].to_numpy(dtype="int64")   - int(base[COL_GW_TX]))
    d_node_rx = (win[COL_NODE_RX].to_numpy(dtype="int64") - int(base[COL_NODE_RX]))

    invalid = (d_node_tx < 0) | (d_gw_rx < 0) | (d_gw_tx < 0) | (d_node_rx < 0)
    denom = d_node_tx + d_gw_tx
    numer = d_gw_rx + d_node_rx

    pdr = np.full_like(denom, np.nan, dtype=float)
    ok = (denom > 0) & (~invalid)
    pdr[ok] = np.clip(numer[ok] / denom[ok], 0.0, 1.0)

    final_pdr = float(pdr[ok][-1]) if np.any(ok) else float("nan")
    return t_minutes, pdr, final_pdr


def compute_overall_pdr_series_multi_gw(path: str, start, end) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes a single PDR curve for a multi-gateway CSV."""
    df = read_and_prepare_csv(path)
    start_ts = parse_ts(start)
    end_ts   = parse_ts(end)

    gw_col = detect_gateway_column(df)
    if gw_col is None:
        raise ValueError(f"Gateway address column not found. Candidates: {GW_COL_CANDIDATES}")

    df = df[df[COL_TS] <= end_ts].copy()
    if df.empty:
        return np.array([]), np.array([]), float("nan")

    groups, baselines = [], {}
    time_union = pd.DatetimeIndex([start_ts])
    acc_cols = [COL_NODE_TX, COL_GW_RX, COL_GW_TX, COL_NODE_RX]

    for gw_val, sub in df.groupby(gw_col):
        sub = dedup_by_ts(sub)
        base = last_row_leq(sub, start_ts)
        baselines[gw_val] = base
        sub_win = sub[sub[COL_TS] >= pd.to_datetime(base[COL_TS])].copy()
        time_union = time_union.union(sub_win[COL_TS])
        groups.append((gw_val, sub_win))

    time_union = time_union[(time_union >= start_ts) & (time_union <= end_ts)]
    if len(time_union) == 0:
        return np.array([]), np.array([]), float("nan")

    acc_sum = {c: np.zeros(len(time_union), dtype="float64") for c in acc_cols}
    any_negative_mask = np.zeros(len(time_union), dtype=bool)

    for gw_val, sub_win in groups:
        base = baselines[gw_val]
        sub_aug = pd.concat([base.to_frame().T, sub_win], ignore_index=True)
        sub_aug = dedup_by_ts(sub_aug)
        sub_aug = sub_aug.set_index(COL_TS).reindex(time_union).ffill()

        deltas = sub_aug[acc_cols].to_numpy(dtype="float64") - base[acc_cols].astype("int64").to_numpy()

        any_negative_mask |= (deltas < 0).any(axis=1)
        for j, c in enumerate(acc_cols):
            acc_sum[c] += deltas[:, j]

    numer = acc_sum[COL_GW_RX] + acc_sum[COL_NODE_RX]
    denom = acc_sum[COL_NODE_TX] + acc_sum[COL_GW_TX]

    pdr = np.full_like(denom, np.nan, dtype=float)
    ok = (denom > 0) & (~any_negative_mask)
    pdr[ok] = np.clip(numer[ok] / denom[ok], 0.0, 1.0)

    t_minutes = (time_union - start_ts).total_seconds().to_numpy() / 60.0
    final_pdr = float(pdr[ok][-1]) if np.any(ok) else float("nan")
    return t_minutes, pdr, final_pdr


def make_line_plot(series_list: List[SeriesResult], outdir: Path):
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=False)

    for s in series_list:
        ax.plot(s.t_minutes, s.pdr_overall, linewidth=1.8, marker='o', markersize=2.5, markevery=25, label=s.label)

    if ZOOM_X_WINDOW is not None:
        ax.set_xlim(*ZOOM_X_WINDOW)

    if ZOOM_Y_MODE.lower() == "fixed":
        ax.set_ylim(*ZOOM_Y_FIXED)
    else:
        all_y = np.concatenate([s.pdr_overall[np.isfinite(s.pdr_overall)] for s in series_list if len(s.pdr_overall)])
        if all_y.size:
            lo, hi = float(np.nanpercentile(all_y, 5)), float(np.nanpercentile(all_y, 95))
            pad = max(0.003, (hi - lo) * 0.1)
            ax.set_ylim(max(0.0, lo - pad), min(1.05, hi + pad))
        else:
            ax.set_ylim(0.0, 1.05)

    ax.set_xlabel("Time since start (min)")
    ax.set_ylabel("Overall PDR")
    ax.grid(True, axis="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True)
    fig.subplots_adjust(top=0.80)

    plt.show()

    out_png = outdir / "pdr_over_time.png"
    out_pdf = outdir / "pdr_over_time.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def compute_overall_pdr_series(path: str, start, end, multi_gw: bool):
    if multi_gw:
        return compute_overall_pdr_series_multi_gw(path, start, end)
    return compute_overall_pdr_series_single_gw(path, start, end)


def main():
    if not CONFIG:
        print("CONFIG list is empty. Please edit the script.")
        sys.exit(1)

    outdir = Path(".").resolve()
    results: List[SeriesResult] = []

    for item in CONFIG:
        t_min, pdr, final_pdr = compute_overall_pdr_series(
            item["path"], item["start"], item["end"], bool(item.get("multi_gw", False))
        )
        results.append(SeriesResult(
            label=item["label"],
            t_minutes=t_min,
            pdr_overall=pdr,
            final_pdr=final_pdr,
            file=str(Path(item["path"]).resolve()),
        ))

    df_sum = pd.DataFrame([{
        "label": r.label,
        "final_overall_pdr": r.final_pdr,
        "points": len(r.t_minutes),
        "file": r.file,
    } for r in results])

    with pd.option_context("display.max_colwidth", 120):
        print(df_sum.to_string(index=False, justify="left", formatters={
            "final_overall_pdr": lambda v: "NaN" if pd.isna(v) else f"{v:.2%}",
        }))

    make_line_plot(results, outdir)


if __name__ == "__main__":
    main()
