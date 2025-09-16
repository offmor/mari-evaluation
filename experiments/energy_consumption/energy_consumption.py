from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG
CSV_CURRENT = "LOGS_BASE_DIR / Main current - Ace.csv"
CSV_VOLTAGE = "LOGS_BASE_DIR / Main voltage - Ace.csv"
CSV_POWER   = "LOGS_BASE_DIR / Main power - Ace.csv"

# Time window(s) to plot/integrate (seconds)
SLICES = [
    (1.12632, 1.15664),
]

# Axis/time offsets
OFFSET_VOLTAGE_S = 0.0
OFFSET_POWER_S   = 0.0

# Current axis options
CURRENT_YLIM_MA   = (0.0, 20.0)
CURRENT_YTICKS_MA = None
CURRENT_YLIM_A    = None
CURRENT_YTICKS_A  = None

# Voltage axis options
VOLTAGE_YLIM   = (0.0, 5.0)
VOLTAGE_YTICKS = [0, 1, 2, 3, 4, 5]

# ---- Background phase annotation (17 slots) ----
# Sequence: BBB U SS D SSSSS D SSSS  -> total 17 slots
SCHEDULE_SLOTS = "BBBUSSDSSSSSDSSSS"
SHOW_CELL_ANNOTATIONS = True
USE_SLICE_EQUAL_FIT = True    # Equal partitioning within the slice

# Used only when USE_SLICE_EQUAL_FIT=False
SF_DURATION_S  = 0.0303
PHASE_OFFSET_S = 0.0
BEACON_ANCHOR_S = None

# Colors & labels
CELL_ALPHA = 0.45
COLOR_B = "#9575cd"    # Beacon
COLOR_D = "#b0b0b0"    # Downlink
COLOR_U = "#ffcc80"    # Uplink
COLOR_S = "#81d4fa"    # Scan
LABEL_SIZE = 10
LABEL_WEIGHT = "bold"

# Label anchors (y value to place text at)
LABEL_ANCHOR_MA_MAIN = 15.0   # Beacon/Uplink/Downlink
LABEL_ANCHOR_MA_SCAN = 17.5   # Background scan
LABEL_Y_OFFSET_FRAC  = 0.0

# Put energy summary text on the figure?
SHOW_ENERGY_BOX = False  # False = print energy only in terminal

LABEL_MAP = {"B": "Beacon", "D": "Downlink", "U": "Uplink", "S": "Background scan"}
COLOR_MAP = {"B": COLOR_B,  "D": COLOR_D,  "U": COLOR_U,  "S": COLOR_S}
# ============================================================


# ---------- File helpers ----------
def resolve_csv_path(p: str | None, prefer_keywords: list[str] | None = None) -> Path | None:
    """Resolve a path that may be a file, directory, or stem; pick latest CSV in a directory."""
    if not p:
        return None
    path = Path(p).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        csvs = sorted(path.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not csvs:
            return None
        if prefer_keywords:
            pri = [c for c in csvs if any(k in c.name.lower() for k in prefer_keywords)]
            if pri:
                print(f"[info] Directory mode: auto-selected CSV -> {pri[0]}")
                return pri[0]
        print(f"[info] Directory mode: auto-selected CSV -> {csvs[0]}")
        return csvs[0]
    with_csv = path.with_suffix(".csv")
    return with_csv if with_csv.exists() else None

def auto_find_in_dir(base_csv: Path, keywords: tuple[str, ...]) -> Path | None:
    """In the same directory as base_csv, pick the most recent CSV matching keywords."""
    base = base_csv.parent
    candidates = sorted(base.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    pri = [c for c in candidates if any(k in c.name.lower() for k in keywords)]
    if pri:
        print(f"[info] Auto-found matching CSV in same directory -> {pri[0]}")
        return pri[0]
    for c in candidates:
        if c != base_csv:
            print(f"[warn] No file matches {keywords}; falling back to -> {c}")
            return c
    return None

def guess_columns(df: pd.DataFrame, role_hint: str = ""):
    """Heuristic column detection: one time-like + one value-like column."""
    cols = [c for c in df.columns if isinstance(c, str)]
    low = {c.lower(): c for c in cols}
    time_keys = ["timestamp", "time", "time (s)", "time_s", "seconds", "sec", "t", "date", "datetime"]
    val_keys  = ["value", "current", "current (a)", "current(a)", "current_a",
                 "i", "i(a)", "current (ma)", "ma", "amperage", "voltage", "v", "power", "w", "mw"]
    time_col = next((low[k] for k in time_keys if k in low), None)
    val_col  = next((low[k] for k in val_keys  if k in low), None)
    if (time_col is None or val_col is None) and len(cols) == 2:
        c1, c2 = cols
        if "time" in c1.lower() or "stamp" in c1.lower():
            time_col, val_col = c1, c2
        elif "time" in c2.lower() or "stamp" in c2.lower():
            time_col, val_col = c2, c1
    if time_col is None or val_col is None:
        raise RuntimeError(f"[{role_hint}] Unable to detect time/value columns: {list(df.columns)}")
    return time_col, val_col

def load_csv(path: Path) -> pd.DataFrame:
    """CSV loader that first tries comma-decimal, then falls back to dot-decimal."""
    try:
        return pd.read_csv(path, sep=None, engine="python", decimal=',',
                           comment='#', skipinitialspace=True)
    except Exception as e:
        print(f"[warn] Comma-decimal parse failed ({e}); falling back to '.' ...")
        return pd.read_csv(path, sep=None, engine="python",
                           comment='#', skipinitialspace=True)

def to_seconds_general(series: pd.Series):
    """Normalize a time-like column to seconds (supports numeric/us/ms and datetimes)."""
    try:
        tnum = pd.to_numeric(series, errors="raise")
        span = float(tnum.max()) - float(tnum.min())
        if span > 1e6: return tnum/1e6, "us→s"
        if span > 1e3: return tnum/1e3, "ms→s"
        return tnum, "s"
    except Exception:
        pass
    tdt = pd.to_datetime(series, errors="coerce", utc=False)
    if tdt.isna().all():
        raise ValueError("Time column is neither numeric nor a parseable timestamp")
    t0 = tdt.iloc[0]
    return (tdt - t0).dt.total_seconds(), "datetime→s(Δt)"


# ---------- Background segmentation ----------
def _compute_phase_with_beacon_anchor(t_start, slots, sf_dur, cell_dt, anchor_s):
    """Compute phase so beacon slot centers align with an anchor if provided."""
    if anchor_s is None:
        return PHASE_OFFSET_S
    be_idx = [i for i, c in enumerate(slots) if c == "B"]
    if not be_idx:
        return PHASE_OFFSET_S
    base0 = t_start + PHASE_OFFSET_S
    k = round((anchor_s - base0) / sf_dur)
    base = base0 + k * sf_dur
    centers = [base + i * cell_dt + 0.5 * cell_dt for i in be_idx]
    c_near = centers[int(np.argmin([abs(anchor_s - c) for c in centers]))]
    return PHASE_OFFSET_S + (anchor_s - c_near)

def prepare_slot_segments_equal_fit(t_start: float, t_end: float, slots: str):
    """Evenly divide [t_start, t_end] into len(slots) cells; merge consecutive same-kind cells."""
    n = len(slots)
    if n == 0 or t_end <= t_start:
        return []
    dt = (t_end - t_start) / n
    segs = [[t_start + i * dt, t_start + (i + 1) * dt, ch] for i, ch in enumerate(slots)]
    merged = []
    for t0, t1, kind in segs:
        if not merged or kind != merged[-1][2]:
            merged.append([t0, t1, kind])
        else:
            merged[-1][1] = t1
    return merged

def prepare_slot_segments_phased(t_start, t_end, slots, sf_duration_s, phase_offset_s, anchor_beacon_s=None):
    """Phase-aligned segmentation within a superframe (when not using equal-fit)."""
    if not slots:
        return []
    n = len(slots)
    cell_dt = float(sf_duration_s) / n
    phase = _compute_phase_with_beacon_anchor(t_start, slots, sf_duration_s, cell_dt, anchor_beacon_s)
    k0 = int(np.floor((t_start - (t_start + phase)) / sf_duration_s))
    frame_start = t_start + phase + k0 * sf_duration_s
    runs = []
    t = frame_start
    while t < t_end + sf_duration_s:
        for j in range(n):
            t0, t1 = t + j * cell_dt, t + (j + 1) * cell_dt
            if t1 < t_start:
                continue
            if t0 > t_end:
                break
            runs.append((max(t0, t_start), min(t1, t_end), slots[j]))
        t += sf_duration_s
        if t > t_end:
            break
    merged = []
    for seg in runs:
        if not merged or seg[2] != merged[-1][2] or abs(seg[0] - merged[-1][1]) > 1e-12:
            merged.append(list(seg))
        else:
            merged[-1][1] = seg[1]
    return merged

def draw_background(ax, merged_segments):
    """Draw colored spans for background segments."""
    for t0, t1, kind in merged_segments:
        ax.axvspan(t0, t1, color=COLOR_MAP.get(kind, "#cccccc"),
            alpha=CELL_ALPHA, linewidth=0, zorder=0)

def place_labels_fixed_by_kind(ax, merged_segments, unit_is_mA: bool):
    """Place labels at fixed y anchors per kind."""
    y_min, y_max = ax.get_ylim()
    yr = max(y_max - y_min, 1e-9)
    for t0, t1, kind in merged_segments:
        anchor_ma = 17.5 if kind == "S" else 15.0
        y_anchor = anchor_ma if unit_is_mA else (anchor_ma / 1000.0)
        y_text = y_anchor + LABEL_Y_OFFSET_FRAC * yr
        y_text = max(y_min + 0.02*yr, min(y_text, y_max - 0.03*yr))
        x_text = 0.5 * (t0 + t1)
        ax.text(x_text, y_text, LABEL_MAP.get(kind, kind),
                ha="center", va="bottom",
                fontsize=LABEL_SIZE, fontweight=LABEL_WEIGHT,
                color="black", zorder=7)

# ---------- Energy / power ----------
def integrate_power_between(t_s: np.ndarray, p_w: np.ndarray, t0: float, t1: float) -> tuple[float, float]:
    """Integrate power P(t) over [t0, t1]; return (energy_J, average_power_W)."""
    if t1 <= t0:
        raise ValueError("t1 must be > t0")
    m = np.isfinite(t_s) & np.isfinite(p_w)
    t, p = np.asarray(t_s[m], float), np.asarray(p_w[m], float)
    if t.size < 2:
        raise ValueError("Insufficient power samples")
    o = np.argsort(t); t, p = t[o], p[o]
    if t0 < t[0] or t1 > t[-1]:
        raise ValueError("Integration window is outside the power data range")
    sel = (t >= t0) & (t <= t1); ts, ps = t[sel], p[sel]
    if ts.size == 0 or ts[0] > t0:
        ps0 = float(np.interp(t0, t, p)); ts = np.insert(ts, 0, t0); ps = np.insert(ps, 0, ps0)
    if ts[-1] < t1:
        ps1 = float(np.interp(t1, t, p)); ts = np.append(ts, t1); ps = np.append(ps, ps1)
    E = float(np.trapz(ps, ts))
    return E, E / (t1 - t0)

def compute_power_series_from_IV(t_i: np.ndarray, i_a: np.ndarray,
                                 t_v: np.ndarray | None, v_v: np.ndarray | None,
                                 offset_v_s: float = 0.0):
    """Build P(t)=V(t)*I(t) on I timestamps using interpolated V(t)."""
    if t_v is None or v_v is None or t_i.size == 0:
        return None
    v_interp = np.interp(t_i, t_v + offset_v_s, v_v, left=np.nan, right=np.nan)
    m = np.isfinite(v_interp) & np.isfinite(i_a)
    return (t_i[m], v_interp[m] * i_a[m]) if m.sum() >= 2 else None

def summarize_energy_by_kind(t_power: np.ndarray, p_watts: np.ndarray, merged_segments):
    """Sum energy per background kind over provided segments."""
    by_kind = {}
    for t0, t1, kind in merged_segments:
        E, _ = integrate_power_between(t_power, p_watts, float(t0), float(t1))
        by_kind[kind] = by_kind.get(kind, 0.0) + E
    return by_kind

# ---------- Plotting / compute one slice ----------
def plot_one_segment(current_csv: Path,
                     voltage_csv: Path | None,
                     t_start: float, t_end: float,
                     df_i: pd.DataFrame, ti_col: str, i_col: str, t_i_sec,
                     df_v: pd.DataFrame | None, tv_col: str | None, v_col: str | None, t_v_sec,
                     df_p: pd.DataFrame | None, tp_col: str | None, p_col: str | None, t_p_sec):
    """Plot current (and voltage if available), draw background, compute energy over [t_start,t_end]."""
    if t_end <= t_start:
        raise ValueError("End time must be greater than start time")

    # Current segment
    i_vals_all = pd.to_numeric(df_i[i_col], errors="coerce").to_numpy()
    m_i = (t_i_sec >= t_start) & (t_i_sec <= t_end)
    seg_i = pd.DataFrame({"time_s": t_i_sec[m_i], "current_A_raw": i_vals_all[m_i]}).dropna()
    if seg_i.empty:
        raise ValueError(f"No current samples in [{t_start}, {t_end}] s")

    # Voltage segment
    seg_v = None
    t_v_all, v_all = None, None
    if voltage_csv and df_v is not None and tv_col and v_col:
        v_all = pd.to_numeric(df_v[v_col], errors="coerce").to_numpy()
        t_v_all = np.asarray(t_v_sec, dtype=float) + float(OFFSET_VOLTAGE_S)
        m_v = (t_v_all >= t_start) & (t_v_all <= t_end)
        seg_v = pd.DataFrame({"time_s": t_v_all[m_v], "voltage_V_raw": v_all[m_v]}).dropna()
        if seg_v.empty:
            seg_v = None

    # Background slots
    merged_segments = []
    if SHOW_CELL_ANNOTATIONS:
        if USE_SLICE_EQUAL_FIT:
            merged_segments = prepare_slot_segments_equal_fit(t_start, t_end, SCHEDULE_SLOTS)
        else:
            merged_segments = prepare_slot_segments_phased(
                t_start, t_end, SCHEDULE_SLOTS, SF_DURATION_S, PHASE_OFFSET_S, BEACON_ANCHOR_S
            )

    # Plot
    fig, ax1 = plt.subplots(figsize=(9.6, 3.8))
    fig.patch.set_facecolor("white"); ax1.set_facecolor("white")
    if merged_segments:
        draw_background(ax1, merged_segments)

    # Current trace
    use_mA = seg_i["current_A_raw"].abs().max() < 0.2
    y_i, y_label = (seg_i["current_A_raw"] * 1e3, "Current (mA)") if use_mA else (seg_i["current_A_raw"], "Current (A)")
    ln1 = ax1.plot(seg_i["time_s"], y_i, color="#d35400", linewidth=1.9, label=y_label, zorder=5)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel(y_label); ax1.grid(True, linestyle="--", alpha=0.3, zorder=2)
    
    if use_mA and CURRENT_YLIM_MA: ax1.set_ylim(*CURRENT_YLIM_MA)
    if use_mA and CURRENT_YTICKS_MA: ax1.set_yticks(CURRENT_YTICKS_MA)
    if not use_mA and CURRENT_YLIM_A: ax1.set_ylim(*CURRENT_YLIM_A)
    if not use_mA and CURRENT_YTICKS_A: ax1.set_yticks(CURRENT_YTICKS_A)
    
    if merged_segments:
        place_labels_fixed_by_kind(ax1, merged_segments, unit_is_mA=use_mA)

    # Voltage trace (twin y)
    ln2 = []
    if seg_v is not None:
        ax2 = ax1.twinx()
        ln2 = ax2.plot(seg_v["time_s"], seg_v["voltage_V_raw"], color="#0b3c7c", linewidth=1.7, label="Voltage (V)", zorder=6)
        ax2.set_ylabel("Voltage (V)")
        if VOLTAGE_YLIM: ax2.set_ylim(*VOLTAGE_YLIM)
        if VOLTAGE_YTICKS: ax2.set_yticks(VOLTAGE_YTICKS)

    # Energy (prefer power CSV; else compute V*I)
    t_power, p_watts = None, None
    if df_p is not None and not df_p.empty and tp_col and p_col and t_p_sec is not None:
        t_power = np.asarray(t_p_sec, float) + float(OFFSET_POWER_S)
        p_watts = pd.to_numeric(df_p[p_col], errors="coerce").to_numpy()
        print("[info] Using power CSV for integration.")
    elif voltage_csv and t_v_all is not None and v_all is not None:
        iv_result = compute_power_series_from_IV(t_i_sec, i_vals_all, t_v_all, v_all)
        if iv_result:
            t_power, p_watts = iv_result
            print("[info] No power CSV: built instantaneous power via V×I for integration.")

    if t_power is not None and p_watts is not None:
        E_J, P_avg_W = integrate_power_between(t_power, p_watts, t_start, t_end)
        nWh, mJ, mW = E_J * (1e9 / 3600.0), E_J * 1e3, P_avg_W * 1e3
        print(f"[Energy Total] {t_start:.6f}–{t_end:.6f}s : {nWh:.3f} nWh | {mJ:.3f} mJ (avg {mW:.2f} mW)")

        if merged_segments:
            by_kind = summarize_energy_by_kind(t_power, p_watts, merged_segments)
            print("[Energy by kind]")
            for k in ("B", "U", "D", "S"):
                if k in by_kind:
                    Ek_nWh, Ek_mJ = by_kind[k] * (1e9 / 3600.0), by_kind[k] * 1e3
                    print(f"  {LABEL_MAP[k]:<16s}: {Ek_nWh:8.3f} nWh | {Ek_mJ:8.3f} mJ")

    # Legend & save
    lines = ln1 + ln2
    if lines:
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True, framealpha=0.85)
    
    out_base = f"segment_IV_{t_start:.6f}_{t_end:.6f}"
    out_dir = current_csv.parent
    fig.savefig(out_dir / f"{out_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{out_base}.pdf", bbox_inches="tight")
    print(f"[saved] {out_dir / (out_base + '.png')}")
    print(f"[saved] {out_dir / (out_base + '.pdf')}")
    plt.show(); plt.close(fig)

def main():
    # Resolve CSVs (current is required; voltage/power optional)
    cur_csv = resolve_csv_path(CSV_CURRENT, prefer_keywords=["current", "main"])
    if not cur_csv:
        raise FileNotFoundError(f"Current CSV not found: {CSV_CURRENT}")
    print(f"[info] Using current CSV: {cur_csv}")

    vol_csv = resolve_csv_path(CSV_VOLTAGE, prefer_keywords=["volt", "voltage"])
    if not vol_csv:
        print(f"[warn] Voltage CSV not found: {CSV_VOLTAGE}. Trying auto-search...")
        vol_csv = auto_find_in_dir(cur_csv, keywords=("volt", "voltage"))

    pow_csv = resolve_csv_path(CSV_POWER, prefer_keywords=["power"])
    if not pow_csv:
        print(f"[warn] Power CSV not found: {CSV_POWER}. Trying auto-search...")
        pow_csv = auto_find_in_dir(cur_csv, keywords=("power", "pwr"))

    # Load & detect columns
    df_i = load_csv(cur_csv)
    ti_col, i_col = guess_columns(df_i, role_hint="current")
    t_i_sec, _ = to_seconds_general(df_i[ti_col])

    df_v, tv_col, v_col, t_v_sec = None, None, None, None
    if vol_csv:
        df_v = load_csv(vol_csv)
        tv_col, v_col = guess_columns(df_v, role_hint="voltage")
        t_v_sec, _ = to_seconds_general(df_v[tv_col])

    df_p, tp_col, p_col, t_p_sec = None, None, None, None
    if pow_csv:
        df_p = load_csv(pow_csv)
        tp_col, p_col = guess_columns(df_p, role_hint="power")
        t_p_sec, _ = to_seconds_general(df_p[tp_col])

    # Process all slices
    for (ts, te) in SLICES:
        plot_one_segment(cur_csv, vol_csv, float(ts), float(te),
                         df_i, ti_col, i_col, t_i_sec,
                         df_v, tv_col, v_col, t_v_sec,
                         df_p, tp_col, p_col, t_p_sec)

if __name__ == "__main__":
    main()
