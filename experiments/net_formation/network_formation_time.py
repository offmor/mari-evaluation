import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ===================== Raw data =====================
formation_times_data = {
    '20': [
        ["2025-08-20T17:10:27.997060", "2025-08-20T17:10:29.167477"],
        ["2025-08-20T17:10:45.123005", "2025-08-20T17:10:46.286759"],
        ["2025-08-20T17:11:02.934574", "2025-08-20T17:11:04.348200"],
        ["2025-08-20T17:11:22.125335", "2025-08-20T17:11:23.282726"],
        ["2025-08-20T17:11:42.500051", "2025-08-20T17:11:43.216300"],
        ["2025-08-20T17:11:57.794485", "2025-08-20T17:11:58.959989"],
        ["2025-08-20T17:12:15.076268", "2025-08-20T17:12:16.247260"],
        ["2025-08-20T17:12:49.325709", "2025-08-20T17:12:50.510088"],
        ["2025-08-20T17:13:06.496013", "2025-08-20T17:13:07.440589"],
        ["2025-08-20T17:13:27.727941", "2025-08-20T17:13:28.912174"]
    ],
    '40': [
        ["2025-08-20T17:36:37.460438", "2025-08-20T17:36:39.126636"],
        ["2025-08-20T17:36:55.092739", "2025-08-20T17:36:56.785143"],
        ["2025-08-20T17:37:16.855718", "2025-08-20T17:37:18.569863"],
        ["2025-08-20T17:37:33.775530", "2025-08-20T17:37:35.223233"],
        ["2025-08-20T17:37:52.117386", "2025-08-20T17:37:53.808631"],
        ["2025-08-20T17:38:27.888574", "2025-08-20T17:38:29.645577"],
        ["2025-08-20T17:38:45.289153", "2025-08-20T17:38:46.792183"],
        ["2025-08-20T17:39:06.040298", "2025-08-20T17:39:08.045841"],
        ["2025-08-20T17:39:25.628761", "2025-08-20T17:39:27.665114"],
        ["2025-08-20T17:39:41.658718", "2025-08-20T17:39:42.887393"]
    ],
    '60': [
        ["2025-08-20T18:06:59.810252", "2025-08-20T18:07:02.065380"],
        ["2025-08-20T18:07:20.661893", "2025-08-20T18:07:22.943854"],
        ["2025-08-20T18:07:38.088445", "2025-08-20T18:07:40.619509"],
        ["2025-08-20T18:11:03.379843", "2025-08-20T18:11:05.879121"],
        ["2025-08-20T18:11:23.220535", "2025-08-20T18:11:25.628844"],
        ["2025-08-20T18:11:46.451096", "2025-08-20T18:11:48.613933"],
        ["2025-08-20T18:12:08.676574", "2025-08-20T18:12:10.698052"],
        ["2025-08-20T18:12:29.551181", "2025-08-20T18:12:31.749664"],
        ["2025-08-20T18:12:50.87987", "2025-08-20T18:12:53.109159"],
        ["2025-08-20T18:13:13.591711", "2025-08-20T18:13:16.106868"]
    ],
    '80': [
        ["2025-08-20T18:47:47.571907", "2025-08-20T18:47:50.183764"],
        ["2025-08-20T18:48:09.823060", "2025-08-20T18:48:12.562971"],
        ["2025-08-20T18:48:49.825210", "2025-08-20T18:48:52.758569"],
        ["2025-08-20T18:49:09.226538", "2025-08-20T18:49:11.940658"],
        ["2025-08-20T18:49:29.570239", "2025-08-20T18:49:32.565162"],
        ["2025-08-20T18:49:50.927069", "2025-08-20T18:49:53.445603"],
        ["2025-08-20T18:50:12.504962", "2025-08-20T18:50:15.242730"],
        ["2025-08-20T18:51:45.280364", "2025-08-20T18:51:47.925241"],
        ["2025-08-20T18:52:18.608782", "2025-08-20T18:52:21.626592"],
        ["2025-08-20T18:52:41.110067", "2025-08-20T18:52:43.578102"]
    ],
    '100': [
        ["2025-08-20T19:14:39.751559", "2025-08-20T19:14:43.162334"],
        ["2025-08-20T19:15:00.320734", "2025-08-20T19:15:04.080356"],
        ["2025-08-20T19:15:18.872042", "2025-08-20T19:15:21.890742"],
        ["2025-08-20T19:15:37.619574", "2025-08-20T19:15:41.471992"],
        ["2025-08-20T19:15:58.158890", "2025-08-20T19:16:01.742792"],
        ["2025-08-20T19:16:18.209464", "2025-08-20T19:16:21.739092"],
        ["2025-08-20T19:16:40.027178", "2025-08-20T19:16:43.552969"],
        ["2025-08-20T19:17:02.543253", "2025-08-20T19:17:06.361313"],
        ["2025-08-20T19:17:22.130787", "2025-08-20T19:17:25.487736"],
        ["2025-08-20T19:17:41.083774", "2025-08-20T19:17:44.657159"]
    ]
}

# ===================== Compute durations =====================
durations_data = {}
for node_count, timestamps_list in formation_times_data.items():
    durations = []
    for start_str, end_str in timestamps_list:
        try:
            start_time = datetime.fromisoformat(start_str)
            end_time = datetime.fromisoformat(end_str)
            durations.append((end_time - start_time).total_seconds())
        except Exception:
            continue
    durations_data[node_count] = durations

# ===================== Compute statistics =====================
node_counts_sorted = sorted(int(k) for k in durations_data.keys())
labels_sorted = [str(n) for n in node_counts_sorted]

mean_times_sorted = [
    float(np.mean(durations_data[str(n)])) if durations_data[str(n)] else 0.0
    for n in node_counts_sorted
]
std_devs_sorted = [
    float(np.std(durations_data[str(n)], ddof=0)) if durations_data[str(n)] else 0.0
    for n in node_counts_sorted
]

print("\n--- Network Formation Time: Calculated Statistics ---")
for lab, m, s in zip(labels_sorted, mean_times_sorted, std_devs_sorted):
    print(f"Nodes: {lab:>3}  |  Mean: {m:7.4f} s  |  Std: {s:7.4f} s")
print("-----------------------------------------------------\n")

# ===================== Plot =====================
plt.style.use('seaborn-v0_8-ticks')
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(labels_sorted))
bar_color = plt.cm.Blues(0.6)

# Bars
bars = ax.bar(
    x, mean_times_sorted,
    color=bar_color, zorder=2, width=0.6
)

# Error bars
err = ax.errorbar(
    x, mean_times_sorted, yerr=std_devs_sorted,
    fmt='none', capsize=6, ecolor='dimgray', elinewidth=1.6, zorder=3
)

# Mean labels above error bars
ymax_data = max((m + s) for m, s in zip(mean_times_sorted, std_devs_sorted)) if mean_times_sorted else 1.0
margin = max(0.02 * ymax_data, 0.02)
for xi, m, s in zip(x, mean_times_sorted, std_devs_sorted):
    label_y = m + s + margin
    t = ax.text(
        xi, label_y, f'{m:.2f}',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold'
    )
    t.set_path_effects([
        path_effects.Stroke(linewidth=2.5, foreground='white'),
        path_effects.Normal()
    ])

ax.set_ylim(0, ymax_data * 1.30 if ymax_data > 0 else 10)

ax.set_title('Network Formation Time vs. Number of Nodes', fontsize=18, fontweight='bold', pad=16)
ax.set_xlabel('Number of Nodes (N)', fontsize=14, fontweight='bold')
ax.set_ylabel('Network Formation Time (seconds)', fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels_sorted, fontsize=12)

ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
bar_proxy = Patch(facecolor=bar_color, label='Average')
err_proxy = Line2D([0], [0], color='dimgray', lw=2.2, marker='_', markersize=14, label='Std. Dev.')

leg = ax.legend(
    handles=[bar_proxy, err_proxy],
    loc='upper right',
    frameon=False,
    prop={'weight': 'bold', 'size': 13}
)

plt.tight_layout()

# ===================== Save and show =====================
plt.savefig('evaluation_network_formation_time.pdf', bbox_inches='tight')
plt.show()
