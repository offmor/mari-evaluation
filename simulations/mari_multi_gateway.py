import numpy as np
import matplotlib.pyplot as plt

# Configuration
SCHEDULES = {
    1: {
        "name": "huge",
        "slots": "BBB" + ("UUSDUUUUSDUUU" * 11),
        "max_nodes": 102,
        "d_down": 22,
        "sf_duration": 256.88,  # ms
    },
}

SCHEDULE_ID        = 1
SLOT_WIDTH_MS      = 1.70
RADIO_ON_MAX_MS    = 1.02
HALF_FRAME_MS      = 0.51

TX_LEN_MODE_GATEWAY_TX   = "fixed_half"   # for Beacon/Downlink (gateway TX)

# - "fixed_full"   -> 1.02 ms
# - "fixed_half"   -> 0.51 ms
# - "slot_full"    -> full slot
RX_LEN_MODE_GATEWAY_RX   = "fixed_full"

TX_SAME_SHAPE_FOR_ALL = True

N_CYCLES           = 100        # number of slotframe repetitions
N_GATEWAYS         = 2

K_MODE             = "random"
N_K_SAMPLES        = 200
RNG_SEED           = 12345

FIXED_OFFSETS_MS = None

INCLUDE_GW_TX_vs_GW_TX      = True
INCLUDE_GW_RX_vs_GW_TX      = True
INCLUDE_GW_RX_vs_GW_RX      = True
ASSUME_SAME_FREQUENCY       = True

GATEWAY_TX_CELLS = {'B', 'D'}
UPLINK_CELLS     = {'U', 'S'}
# helpers
def _randgen():
    return np.random.default_rng(RNG_SEED) if RNG_SEED is not None else np.random.default_rng()

def sample_len_ms(mode: str, rng: np.random.Generator) -> float:
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

def build_intervals_one_cycle(schedule_str, slot_width_ms, cells_set, len_mode, rng):
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
        if a1 <= b1:
            i += 1
            in_segment = False
        else:
            j += 1
            in_segment = False
    return total, count

def clip_to_window(intervals, w_start, w_end):
    """Return sub-intervals of 'intervals' clipped to [w_start, w_end)."""
    out = []
    for a, b in intervals:
        if b <= w_start or a >= w_end:
            continue
        out.append((max(a, w_start), min(b, w_end)))
    return out

def any_overlap_in_window(A, B, w_start, w_end):
    """True if any overlap between A and B occurs within [w_start, w_end)."""
    Acl = clip_to_window(A, w_start, w_end)
    if not Acl:
        return False
    Bcl = clip_to_window(B, w_start, w_end)
    if not Bcl:
        return False
    Acl.sort(key=lambda x: x[0])
    Bcl.sort(key=lambda x: x[0])
    ov, _ = overlap_time_and_count_two_lists(Acl, Bcl)
    return ov > 0.0

def make_other_gateways_lists(base_shapes_one_cycle, sf_period, n_cycles,
                              n_gateways, slot_width_ms, fixed_offsets, rng,
                              same_shape=True):
    lists_all = {'gw_tx': [], 'gw_rx': []}

    # Gateway 0: base (offset 0.0)
    for key in lists_all.keys():
        base_full = repeat_intervals(base_shapes_one_cycle[key], sf_period, n_cycles)
        lists_all[key].append(shift_intervals(base_full, 0.0))

    for i in range(2, n_gateways):
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
            for key in lists_all.keys():
                base_full = repeat_intervals(base_shapes_one_cycle[key], sf_period, n_cycles)
                lists_all[key].append(shift_intervals(base_full, float(off)))
        else:
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

# sweep X over [0, sf_period)
N_OFFSET_STEPS = 341  # resolution of X across [0, sf_period)

def main():
    rng = _randgen()

    sch = SCHEDULES[SCHEDULE_ID]
    slots_str = sch["slots"]
    name = sch["name"]

    # Build ONE-CYCLE shapes for gateway 0
    gw_tx_1c, n_cells, sf_period = build_intervals_one_cycle(
        schedule_str=slots_str, slot_width_ms=SLOT_WIDTH_MS,
        cells_set=GATEWAY_TX_CELLS, len_mode=TX_LEN_MODE_GATEWAY_TX, rng=rng
    )
    gw_rx_1c, _, _ = build_intervals_one_cycle(
        schedule_str=slots_str, slot_width_ms=SLOT_WIDTH_MS,
        cells_set=UPLINK_CELLS, len_mode=RX_LEN_MODE_GATEWAY_RX, rng=rng
    )

    # Build others across 3 cycles (to handle wrap-around), union them
    base_shapes_1c = {'gw_tx': gw_tx_1c, 'gw_rx': gw_rx_1c}
    lists_others_3cy = make_other_gateways_lists(
        base_shapes_one_cycle=base_shapes_1c,
        sf_period=sf_period,
        n_cycles=3,
        n_gateways=N_GATEWAYS,
        slot_width_ms=SLOT_WIDTH_MS,
        fixed_offsets=FIXED_OFFSETS_MS,
        rng=rng,
        same_shape=TX_SAME_SHAPE_FOR_ALL
    )
    union_others_gw_tx = union_of_many_interval_lists(lists_others_3cy['gw_tx'])
    union_others_gw_rx = union_of_many_interval_lists(lists_others_3cy['gw_rx'])

    # Gateway #1 single-cycle shapes
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

    # Repeat g1 across 3 cycles so wrap-around is visible when shifted by X
    g1_gw_tx_3cy = repeat_intervals(g1_gw_tx_1c, sf_period, 3)
    g1_gw_rx_3cy = repeat_intervals(g1_gw_rx_1c, sf_period, 3)

    # Prepare X grid in [0, sf_period)
    offsets = np.linspace(0.0, sf_period, N_OFFSET_STEPS, endpoint=False)

    # Arrays to store "ratio of slots with any collision" for each X
    ratio_slots_total = np.zeros_like(offsets)
    comp_ratio = {
        'gwtx_gwtx': np.zeros_like(offsets),
        'gwrx_gwtx': np.zeros_like(offsets),
        'gwrx_gwrx': np.zeros_like(offsets),
    }

    central_start = sf_period
    # For slot i, window is [central_start + i*slot, central_start + (i+1)*slot)
    slot_w = SLOT_WIDTH_MS

    # Sweep X
    for idx, X in enumerate(offsets):
        # Shift gateway #1
        g1_tx_shift = shift_intervals(g1_gw_tx_3cy, X)
        g1_rx_shift = shift_intervals(g1_gw_rx_3cy, X)

        # Per-slot collision booleans
        collided_any   = np.zeros(n_cells, dtype=bool)
        collided_tx_tx = np.zeros(n_cells, dtype=bool)
        collided_rx_tx = np.zeros(n_cells, dtype=bool)
        collided_rx_rx = np.zeros(n_cells, dtype=bool)

        # Check each slot window inside the middle frame
        for i in range(n_cells):
            w_start = central_start + i * slot_w
            w_end   = w_start + slot_w

            if INCLUDE_GW_TX_vs_GW_TX:
                c = any_overlap_in_window(g1_tx_shift, union_others_gw_tx, w_start, w_end)
                collided_tx_tx[i] = c
            if INCLUDE_GW_RX_vs_GW_TX:
                c = any_overlap_in_window(g1_rx_shift, union_others_gw_tx, w_start, w_end)
                collided_rx_tx[i] = c
            if INCLUDE_GW_RX_vs_GW_RX:
                c = any_overlap_in_window(g1_rx_shift, union_others_gw_rx, w_start, w_end)
                collided_rx_rx[i] = c

            collided_any[i] = collided_tx_tx[i] or collided_rx_tx[i] or collided_rx_rx[i]

        # Ratios (% of slots in the slotframe that collide)
        ratio_slots_total[idx] = 100.0 * np.mean(collided_any)
        comp_ratio['gwtx_gwtx'][idx] = 100.0 * np.mean(collided_tx_tx) if INCLUDE_GW_TX_vs_GW_TX else 0.0
        comp_ratio['gwrx_gwtx'][idx] = 100.0 * np.mean(collided_rx_tx) if INCLUDE_GW_RX_vs_GW_TX else 0.0
        comp_ratio['gwrx_gwrx'][idx] = 100.0 * np.mean(collided_rx_rx) if INCLUDE_GW_RX_vs_GW_RX else 0.0

    # Summary print
    print(f"[Schedule] {name}, n_cells={n_cells}, sf_period={sf_period:.2f} ms")
    print(f"[Sim] N_GATEWAYS={N_GATEWAYS}")
    print(f"[Radio] SLOT_WIDTH_MS={SLOT_WIDTH_MS:.2f}, GW_TX={TX_LEN_MODE_GATEWAY_TX}, GW_RX={RX_LEN_MODE_GATEWAY_RX}")
    print(f"[Offset X sweep] {len(offsets)} points in [0, {sf_period:.2f}) ms for gateway #1 absolute offset")
    print(f"[shape] TX_SAME_SHAPE_FOR_ALL={TX_SAME_SHAPE_FOR_ALL}")
    print(f"[assumption] SAME_FREQUENCY={ASSUME_SAME_FREQUENCY}")
    print("-" * 60)
    print("Min/Max fraction of slots with any collision vs X: "
          f"{ratio_slots_total.min():.3f}% / {ratio_slots_total.max():.3f}%")
    print("Per-component max over X (slot-collision %):")
    for k, arr in comp_ratio.items():
        print(f"  - {k}: max={arr.max():.3f}%")

    # Figures: ratio of slots collided vs X
    plt.figure(figsize=(9, 4.5))
    plt.plot(offsets, ratio_slots_total, linewidth=2)
    plt.xlabel("Absolute offset X (ms)   [gateway #1]")
    plt.ylabel("Slots with collision (%)")
    plt.title(
        f"Collision slots ratio vs X | schedule={name}, N={N_GATEWAYS}, "
        f"n_cells={n_cells}, slot={SLOT_WIDTH_MS:.2f} ms"
    )
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
