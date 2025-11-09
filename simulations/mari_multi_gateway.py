import numpy as np
import matplotlib.pyplot as plt
import time

# Configuration
SCHEDULES = {
    1: {
        "name": "huge",
        "slots": "BBB" + ("UUSDUUUUSDUUU" * 11),
        "max_nodes": 102,
        "d_down": 22,
        "sf_duration": 256.88,  # ms
    },
    6: {
        "name": "tiny",
        "slots": "BBB" + ("UUSDUUUUSDUUUU" * 1),
        "max_nodes": 10,
        "d_down": 2,
        "sf_duration": 28.9,  # ms
    },
    7: {
        "name": "tiny2",
        "slots": "BSDUSDU",
        "ch_offets": [0, 0, 1, 2, 3, 4, 5],
        "max_nodes": 2,
        "d_down": 2,
        "sf_duration": 11.9,  # ms
    },
}

CHANNELS_MAP = {
    "beacon": {37, 38, 39},
    "regular": list(range(0, 37)),
}

SCHEDULE_ID        = 7
SLOT_WIDTH_MS      = 1.70      # slot width (paper)
RADIO_ON_MAX_MS    = 1.02
HALF_FRAME_MS      = 0.51      # ~128B "half-frame"

# - "fixed_full"     -> 1.02 ms
# - "fixed_half"     -> 0.51 ms
# - "random_uniform" -> Uniform(0, 1.02 ms)
TX_LEN_MODE_GATEWAY_TX   = "fixed_half"   # for Beacon/Downlink (gateway TX)

# - "fixed_full"   -> 1.02 ms
# - "fixed_half"   -> 0.51 ms
# - "slot_full"    -> full slot
RX_LEN_MODE_GATEWAY_RX   = "fixed_full"

# Whether all gateways share the same per-slot TX/RX "shape" (length pattern)
TX_SAME_SHAPE_FOR_ALL = True

N_CYCLES           = 10        # number of slotframe repetitions
# N_DELTA_STEPS      = 341        # δ sweep resolution (~5 µs)
N_DELTA_STEPS      = 68        # δ sweep resolution (~25 µs)
N_GATEWAYS         = 2          # Set N here (>=2)

# k sampling:
K_MODE             = "random"   # "random" or "exhaustive"
N_K_SAMPLES        = 200        # only used when K_MODE="random"
RNG_SEED           = 12345

# g0=0; g1 uses (k*slot+δ); others evenly spaced in [0, T_slot)
FIXED_OFFSETS_MS = None

# Collision model toggles
INCLUDE_GW_TX_vs_GW_TX      = True   # B/D vs B/D
INCLUDE_GW_RX_vs_GW_TX      = True   # (UL RX) vs other GW's B/D
INCLUDE_GW_RX_vs_GW_RX      = True   # both gateways in UL (uplink vs uplink)
ASSUME_SAME_FREQUENCY       = True   # if later you add channel-hopping, adjust here

# Cell type sets
GATEWAY_TX_CELLS = {'B', 'D'}  # Beacon/Downlink -> GW TX
UPLINK_CELLS     = {'U', 'S'}  # Uplink (scheduled/shared) -> GW RX


def _randgen():
    return np.random.default_rng(RNG_SEED) if RNG_SEED is not None else np.random.default_rng()

def sample_len_ms(mode: str, rng: np.random.Generator) -> float:
    """Sampler for TX/RX duration according to `mode`, capped by RADIO_ON_MAX_MS when appropriate."""
    if mode == "fixed_full":
        return RADIO_ON_MAX_MS
    elif mode == "fixed_half":
        return min(HALF_FRAME_MS, RADIO_ON_MAX_MS)
    elif mode == "random_uniform":
        return float(rng.uniform(0.0, RADIO_ON_MAX_MS))
    elif mode == "slot_full":
        return SLOT_WIDTH_MS
    else:
        raise ValueError(f"Unknown length mode: {mode}")

def build_intervals_one_cycle_with_channel_offsets(schedule_str, slot_width_ms, cells_set, len_mode, rng, channel_offsets):
    """
    Build intervals [start, end) for a single cycle for the cells in cells_set,
    using the duration sampler len_mode.
    """
    ivs = []
    for idx, c in enumerate(schedule_str):
        if c in cells_set:
            ch_offset = channel_offsets[idx]
            start = idx * slot_width_ms
            L = sample_len_ms(len_mode, rng)
            end = start + L
            ivs.append(("beacon" if c == "B" else "regular", ch_offset, start, end))
    n_cells = len(schedule_str)
    period_ms = n_cells * slot_width_ms
    return ivs, n_cells, period_ms

def build_intervals_one_cycle(schedule_str, slot_width_ms, cells_set, len_mode, rng):
    """
    Build intervals [start, end) for a single cycle for the cells in cells_set,
    using the duration sampler len_mode.
    """
    ivs = []
    for idx, c in enumerate(schedule_str):
        if c in cells_set:
            start = idx * slot_width_ms
            L = sample_len_ms(len_mode, rng)
            end = start + L
            ivs.append((start, end))
    n_cells = len(schedule_str)
    period_ms = n_cells * slot_width_ms
    return ivs, n_cells, period_ms

def repeat_intervals(intervals_one_cycle, period_ms, n_cycles):
    out = []
    for k in range(n_cycles):
        off = k * period_ms
        for (a, b) in intervals_one_cycle:
            out.append((a + off, b + off))
    out.sort(key=lambda x: x[0])
    return out

def shift_intervals(intervals, delta_ms):
    return [(a + delta_ms, b + delta_ms) for (a, b) in intervals]

def union_of_many_interval_lists(list_of_interval_lists):
    all_ints = [iv for sub in list_of_interval_lists for iv in sub]
    if not all_ints:
        return []
    all_ints.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e = all_ints[0]
    for s, e in all_ints[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def overlap_time_and_count_two_lists(A, B):
    """
    Overlap between two sorted, non-overlapping lists A, B.
    Returns (total_overlap_ms, #overlap_segments).
    """
    i, j = 0, 0
    total = 0.0
    count = 0
    in_segment = False
    while i < len(A) and j < len(B):
        a0, a1 = A[i]
        b0, b1 = B[j]
        s = max(a0, b0)
        e = min(a1, b1)
        if e > s:
            total += (e - s)
            if not in_segment:
                count += 1
                in_segment = True
        # advance
        if a1 <= b1:
            i += 1
            in_segment = False
        else:
            j += 1
            in_segment = False
    return total, count

def sample_k_values(n_cells, mode, n_samples, rng):
    if mode == "random":
        return rng.integers(low=0, high=n_cells, size=n_samples)
    elif mode == "exhaustive":
        return np.arange(n_cells)
    else:
        raise ValueError("K_MODE must be 'random' or 'exhaustive'.")

def make_other_gateways_lists(base_shapes_one_cycle, sf_period, n_cycles,
                              n_gateways, slot_width_ms, fixed_offsets, rng,
                              same_shape=True):
    """
    Build intervals (already repeated n_cycles) for gateways 0..N-1
    (gateway #1 will be shifted by k*slot+δ later).
    """
    lists_all = {'gw_tx': [], 'gw_rx': []}

    # Gateway 0: base (offset 0.0)
    for key in lists_all.keys():
        base_full = repeat_intervals(base_shapes_one_cycle[key], sf_period, n_cycles)
        lists_all[key].append(shift_intervals(base_full, 0.0))

    # Gateways 2..N-1
    for i in range(2, n_gateways):
        # offset for gateway i
        if fixed_offsets is not None:
            if len(fixed_offsets) != n_gateways:
                raise ValueError("FIXED_OFFSETS_MS length must equal N_GATEWAYS.")
            off = fixed_offsets[i]
            if off is None:
                frac = i / n_gateways
                off = frac * slot_width_ms
        else:
            frac = i / n_gateways
            off = frac * slot_width_ms

        if same_shape:
            # reuse base shape
            for key in lists_all.keys():
                base_full = repeat_intervals(base_shapes_one_cycle[key], sf_period, n_cycles)
                lists_all[key].append(shift_intervals(base_full, float(off)))
        else:
            # regenerate shape for each gateway i
            sch = SCHEDULES[SCHEDULE_ID]["slots"]
            gw_tx_i, _, _ = build_intervals_one_cycle(
                sch, slot_width_ms, GATEWAY_TX_CELLS, TX_LEN_MODE_GATEWAY_TX, rng
            )
            gw_rx_i, _, _ = build_intervals_one_cycle(
                sch, slot_width_ms, UPLINK_CELLS, RX_LEN_MODE_GATEWAY_RX, rng
            )
            for key, src in zip(('gw_tx','gw_rx'), (gw_tx_i, gw_rx_i)):
                base_full = repeat_intervals(src, sf_period, n_cycles)
                lists_all[key].append(shift_intervals(base_full, float(off)))

    return lists_all

# Main
def main():
    rng = _randgen()
    start_time = time.time()
    sch = SCHEDULES[SCHEDULE_ID]
    slots_str = sch["slots"]
    name = sch["name"]

    # Build ONE-CYCLE shapes for gateway 0
    print("Building one-cycle shapes for gateway 0 TX")
    gw_tx_1c, n_cells, sf_period = build_intervals_one_cycle(
        schedule_str=slots_str, slot_width_ms=SLOT_WIDTH_MS,
        cells_set=GATEWAY_TX_CELLS, len_mode=TX_LEN_MODE_GATEWAY_TX, rng=rng
    )
    print(f"One-cycle shapes for gateway 0 TX: {gw_tx_1c}")

    # --- CH: Build ONE-CYCLE shapes for gateway 0 with channel offsets
    print("Building one-cycle shapes for gateway 0 TX with channel offsets")
    gw_tx_1c_ch_off, n_cells, sf_period = build_intervals_one_cycle_with_channel_offsets(
        schedule_str=slots_str, slot_width_ms=SLOT_WIDTH_MS,
        cells_set=GATEWAY_TX_CELLS, len_mode=TX_LEN_MODE_GATEWAY_TX, rng=rng, channel_offsets=sch["ch_offets"]
    )
    print(f"One-cycle shapes for gateway 0 TX with channel offsets: {gw_tx_1c_ch_off}")

    print("Building one-cycle shapes for gateway 0 RX")
    gw_rx_1c, _, _ = build_intervals_one_cycle(
        schedule_str=slots_str, slot_width_ms=SLOT_WIDTH_MS,
        cells_set=UPLINK_CELLS, len_mode=RX_LEN_MODE_GATEWAY_RX, rng=rng
    )

    total_window_ms = N_CYCLES * sf_period

    base_shapes_one_cycle = {'gw_tx': gw_tx_1c, 'gw_rx': gw_rx_1c}

    # Precompute other gateways (0 and 2..N-1).
    print("Building other gateways lists")
    lists_others = make_other_gateways_lists(
        base_shapes_one_cycle=base_shapes_one_cycle,
        sf_period=sf_period,
        n_cycles=N_CYCLES,
        n_gateways=N_GATEWAYS,
        slot_width_ms=SLOT_WIDTH_MS,
        fixed_offsets=FIXED_OFFSETS_MS,
        rng=rng,
        same_shape=TX_SAME_SHAPE_FOR_ALL
    )

    # Union across OTHER gateways
    print("Union across OTHER gateways")
    union_others_gw_tx = union_of_many_interval_lists(lists_others['gw_tx'])
    print(f"Union across OTHER gateways TX: {union_others_gw_tx}")
    union_others_gw_rx = union_of_many_interval_lists(lists_others['gw_rx'])

    # Prepare δ grid
    print("Preparing δ grid")
    deltas = np.linspace(0.0, SLOT_WIDTH_MS - (SLOT_WIDTH_MS/(N_DELTA_STEPS-1)), N_DELTA_STEPS)

    # arrays to store averages over k for each δ
    ratio_avg_total    = np.zeros_like(deltas)
    count_avg_total    = np.zeros_like(deltas)
    overlap_avg_ms_tot = np.zeros_like(deltas)

    comp_overlap = {
        'gwtx_gwtx': np.zeros_like(deltas),  # B/D vs B/D
        'gwrx_gwtx': np.zeros_like(deltas),  # UL(RX) vs B/D
        'gwrx_gwrx': np.zeros_like(deltas),  # UL(RX) vs UL(RX)
    }
    comp_count = {
        'gwtx_gwtx': np.zeros_like(deltas),
        'gwrx_gwtx': np.zeros_like(deltas),
        'gwrx_gwrx': np.zeros_like(deltas),
    }

    # k-set (random or exhaustive)
    ks = sample_k_values(n_cells, K_MODE, N_K_SAMPLES, rng)
    n_k_used = len(ks)

    # Gateway #1 base full-cycle lists (before offset off = k*T_slot + δ)
    if TX_SAME_SHAPE_FOR_ALL:
        g1_gw_tx_1c = gw_tx_1c
        g1_gw_rx_1c = gw_rx_1c
    else:
        g1_gw_tx_1c, _, _ = build_intervals_one_cycle(
            slots_str, SLOT_WIDTH_MS, GATEWAY_TX_CELLS, TX_LEN_MODE_GATEWAY_TX, rng
        )
        g1_gw_rx_1c, _, _ = build_intervals_one_cycle(
            slots_str, SLOT_WIDTH_MS, UPLINK_CELLS, RX_LEN_MODE_GATEWAY_RX, rng
        )

    g1_gw_tx_full = repeat_intervals(g1_gw_tx_1c, sf_period, N_CYCLES)
    g1_gw_rx_full = repeat_intervals(g1_gw_rx_1c, sf_period, N_CYCLES)

    # Sweep δ
    for idx, delta in enumerate(deltas):
        # components per δ averaged over k
        ov_gwtx_gwtx, ct_gwtx_gwtx = [], []
        ov_gwrx_gwtx, ct_gwrx_gwtx = [], []
        ov_gwrx_gwrx, ct_gwrx_gwrx = [], []

        for k_int in ks:
            off = (int(k_int) % n_cells) * SLOT_WIDTH_MS + float(delta)  # k*T_slot + δ

            # Shift gateway #1 lists
            g1_tx_shift = shift_intervals(g1_gw_tx_full, off)
            g1_rx_shift = shift_intervals(g1_gw_rx_full, off)

            # 1) GW_TX vs GW_TX
            if INCLUDE_GW_TX_vs_GW_TX:
                ov, ct = overlap_time_and_count_two_lists(g1_tx_shift, union_others_gw_tx)
                ov_gwtx_gwtx.append(ov); ct_gwtx_gwtx.append(ct)

            # 2) GW_RX(UL) vs other GW_TX
            if INCLUDE_GW_RX_vs_GW_TX:
                ov, ct = overlap_time_and_count_two_lists(g1_rx_shift, union_others_gw_tx)
                ov_gwrx_gwtx.append(ov); ct_gwrx_gwtx.append(ct)

            # 3) GW_RX(UL) vs GW_RX(UL)
            if INCLUDE_GW_RX_vs_GW_RX:
                ov, ct = overlap_time_and_count_two_lists(g1_rx_shift, union_others_gw_rx)
                ov_gwrx_gwrx.append(ov); ct_gwrx_gwrx.append(ct)

        # Average components over k
        m1_ov = np.mean(ov_gwtx_gwtx) if ov_gwtx_gwtx else 0.0
        m1_ct = np.mean(ct_gwtx_gwtx) if ct_gwtx_gwtx else 0.0

        m2_ov = np.mean(ov_gwrx_gwtx) if ov_gwrx_gwtx else 0.0
        m2_ct = np.mean(ct_gwrx_gwtx) if ct_gwrx_gwtx else 0.0

        m3_ov = np.mean(ov_gwrx_gwrx) if ov_gwrx_gwrx else 0.0
        m3_ct = np.mean(ct_gwrx_gwrx) if ct_gwrx_gwrx else 0.0

        total_overlap = m1_ov + m2_ov + m3_ov
        total_count   = m1_ct + m2_ct + m3_ct

        overlap_avg_ms_tot[idx] = total_overlap
        ratio_avg_total[idx]    = 100.0 * total_overlap / total_window_ms if total_window_ms > 0 else 0.0
        count_avg_total[idx]    = total_count

        # store components for later summary
        comp_overlap['gwtx_gwtx'][idx] = m1_ov
        comp_overlap['gwrx_gwtx'][idx] = m2_ov
        comp_overlap['gwrx_gwrx'][idx] = m3_ov

        comp_count['gwtx_gwtx'][idx] = m1_ct
        comp_count['gwrx_gwtx'][idx] = m2_ct
        comp_count['gwrx_gwrx'][idx] = m3_ct

    # Totals across all (δ, k)
    total_overlap_time_ms_all = float(np.sum(overlap_avg_ms_tot) * len(ks))
    total_collisions_all      = float(np.sum(count_avg_total) * len(ks))

    # Component totals
    comp_totals = {k: float(np.sum(v) * len(ks)) for k, v in comp_overlap.items()}
    comp_counts = {k: float(np.sum(v) * len(ks)) for k, v in comp_count.items()}

    # Summary print
    k_desc = f"{K_MODE}, samples={len(ks)}"
    print(f"[Schedule] {name}, n_cells={n_cells}, sf_period={sf_period:.2f} ms")
    print(f"[Sim] N_CYCLES={N_CYCLES}, total_window={total_window_ms:.2f} ms, N_GATEWAYS={N_GATEWAYS}")
    print(f"[Radio] SLOT_WIDTH_MS={SLOT_WIDTH_MS:.2f}, GW_TX={TX_LEN_MODE_GATEWAY_TX}, GW_RX={RX_LEN_MODE_GATEWAY_RX}")
    print(f"[Delta sweep] {len(deltas)} points in [0, {SLOT_WIDTH_MS:.2f}) ms (gateway #1)")
    print(f"[k] mode={k_desc}")
    print(f"[shape] TX_SAME_SHAPE_FOR_ALL={TX_SAME_SHAPE_FOR_ALL}")
    print(f"[assumption] SAME_FREQUENCY={ASSUME_SAME_FREQUENCY}")
    print("-" * 60)
    print("Min/Max total collision ratio (avg k): "
          f"{ratio_avg_total.min():.6f}% / {ratio_avg_total.max():.6f}%")
    print("Min/Max total collision count  (avg k): "
          f"{count_avg_total.min():.2f} / {count_avg_total.max():.2f}")
    print("-" * 60)
    print(f"TOTAL collisions over all (δ, k): {total_collisions_all:,.0f} events")
    print(f"TOTAL overlap time over all (δ, k): {total_overlap_time_ms_all:,.2f} ms")
    print("Breakdown (overlap time, ms):")
    for k,v in comp_totals.items():
        print(f"  - {k}: {v:.2f}")
    print("Breakdown (collision events):")
    for k,v in comp_counts.items():
        print(f"  - {k}: {v:.0f}")
    print("-" * 60)

    print(f"Total simulation execution time: {time.time() - start_time:.2f} seconds")

    # Figures
    plt.figure(figsize=(9, 4.5))
    plt.plot(deltas, ratio_avg_total, linewidth=2)
    plt.xlabel("δ (ms)   [relative offset of gateway #1 within a slot]")
    plt.ylabel("Collision ratio (%)")
    plt.title(
        f"TOTAL Collision ratio vs δ | schedule={name}, N={N_GATEWAYS}, n_cells={n_cells}, "
        f"slot={SLOT_WIDTH_MS:.2f} ms, cycles={N_CYCLES}, k={k_desc}"
    )
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4.5))
    plt.plot(deltas, count_avg_total, linewidth=2)
    plt.xlabel("δ (ms)   [relative offset of gateway #1 within a slot]")
    plt.ylabel("Collision count (avg over k)")
    plt.title(
        f"TOTAL Collision count vs δ | schedule={name}, N={N_GATEWAYS}, n_cells={n_cells}, "
        f"slot={SLOT_WIDTH_MS:.2f} ms, cycles={N_CYCLES}, k={k_desc}"
    )
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
