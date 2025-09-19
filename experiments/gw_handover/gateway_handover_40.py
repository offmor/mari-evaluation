import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
TIMEOUT_VALUE_SECONDS = 1.24415

# Raw Data
left_data = {
    'timestamp': [
        '2025-08-21T14:48:24.904007', '2025-08-21T14:48:25.181390', '2025-08-21T14:48:25.460680',
        '2025-08-21T14:48:25.669624', '2025-08-21T14:48:25.703604', '2025-08-21T14:48:26.198363',
        '2025-08-21T14:48:26.676453', '2025-08-21T14:48:27.139810', '2025-08-21T14:48:27.625664',
        '2025-08-21T14:48:27.668476', '2025-08-21T14:48:27.684496', '2025-08-21T14:48:27.885739',
        '2025-08-21T14:48:27.886029', '2025-08-21T14:48:27.967797', '2025-08-21T14:48:28.132502',
        '2025-08-21T14:48:28.132650', '2025-08-21T14:48:28.156566', '2025-08-21T14:48:28.189005',
        '2025-08-21T14:48:28.379886', '2025-08-21T14:48:28.413571', '2025-08-21T14:48:28.420736',
        '2025-08-21T14:48:28.640437', '2025-08-21T14:48:28.642587', '2025-08-21T14:48:28.691065',
        '2025-08-21T14:48:28.691432', '2025-08-21T14:48:28.865892', '2025-08-21T14:48:28.882615',
        '2025-08-21T14:48:28.892570', '2025-08-21T14:48:28.894564', '2025-08-21T14:48:28.946568',
        '2025-08-21T14:48:29.163383', '2025-08-21T14:48:29.371809', '2025-08-21T14:48:29.625568',
        '2025-08-21T14:48:29.647452', '2025-08-21T14:48:29.891620', '2025-08-21T14:48:29.929746',
        '2025-08-21T14:48:30.171668', '2025-08-21T14:48:30.355070', '2025-08-21T14:48:30.369435',
        '2025-08-21T14:48:30.373746'
    ],
    'node_address': [
        '0x44B408F1766E0BF5', '0xFB986FD3DFD36E63', '0xED02BE5529D07C34', '0x613FD651ED0CEF84',
        '0x0AB87C44C20B0149', '0xAD89E8BD0C51208E', '0xBEBAD533DADE0490', '0x0C9A20E2E5E6F54C',
        '0xC82CA0F38D2A15E9', '0xDC878301DB6B0590', '0x151148302FD6F5A2', '0x588D2578218A459B',
        '0x038C701874C52B20', '0xB7B971F84F29BAE7', '0x93B9409637347BA4', '0xDF773BB80AF0C2F9',
        '0x35CE7BC91E0F3535', '0xCD9FC2F10292D689', '0x15D5017AA3800F1E', '0xF32E9A8AD1330DAA',
        '0x554EC897971DCF18', '0x663CFC5276CFCEAB', '0xA8AA71AD0EECCFEA', '0xC48AACFD77448F11',
        '0xC8B80BC5F6082B14', '0x9AB59A717BC7F74D', '0x377F90EF6D2719F0', '0x17D0F99D09B254CD',
        '0x57459F680F660A9F', '0xC5050714D7852A85', '0x7E98AD7EB9DDF59E', '0xBB3EB8601B7E9133',
        '0x4A59DBDDD9A34F45', '0x6CE02E64CFA45053', '0x170A5A45AE7877A0', '0xD7594C7965872E3B',
        '0x0314C3596BF55EB0', '0xDBD14A8C2AB0CE32', '0xDCE0471FD437E359', '0x4A4698749A3DF602'
    ]
}

joined_data = {
    'timestamp': [
        '2025-08-21T14:48:24.081353', '2025-08-21T14:48:24.146380', '2025-08-21T14:48:24.265233',
        '2025-08-21T14:48:24.514026', '2025-08-21T14:48:24.569069', '2025-08-21T14:48:25.088152',
        '2025-08-21T14:48:26.050195', '2025-08-21T14:48:26.559067', '2025-08-21T14:48:26.634163',
        '2025-08-21T14:48:26.646003', '2025-08-21T14:48:26.862109', '2025-08-21T14:48:26.905125',
        '2025-08-21T14:48:27.045040', '2025-08-21T14:48:27.067023', '2025-08-21T14:48:27.100104',
        '2025-08-21T14:48:27.122049', '2025-08-21T14:48:27.144045', '2025-08-21T14:48:27.258548',
        '2025-08-21T14:48:27.510993', '2025-08-21T14:48:27.542953', '2025-08-21T14:48:27.608061',
        '2025-08-21T14:48:27.642135', '2025-08-21T14:48:27.770105', '2025-08-21T14:48:27.782014',
        '2025-08-21T14:48:27.804037', '2025-08-21T14:48:27.890088', '2025-08-21T14:48:27.976554',
        '2025-08-21T14:48:28.018975', '2025-08-21T14:48:28.084041', '2025-08-21T14:48:28.117098',
        '2025-08-21T14:48:28.139041', '2025-08-21T14:48:28.495000', '2025-08-21T14:48:28.560104',
        '2025-08-21T14:48:28.777078', '2025-08-21T14:48:28.960999', '2025-08-21T14:48:29.014049',
        '2025-08-21T14:48:29.069057', '2025-08-21T14:48:29.383042', '2025-08-21T14:48:29.393031',
        '2025-08-21T14:48:29.443893'
    ],
    'node_address': [
        '0x44B408F1766E0BF5', '0xFB986FD3DFD36E63', '0xED02BE5529D07C34', '0x0AB87C44C20B0149',
        '0x613FD651ED0CEF84', '0xAD89E8BD0C51208E', '0x0C9A20E2E5E6F54C', '0xDC878301DB6B0590',
        '0xC82CA0F38D2A15E9', '0x151148302FD6F5A2', '0x588D2578218A459B', '0xB7B971F84F29BAE7',
        '0xCD9FC2F10292D689', '0xBEBAD533DADE0490', '0x93B9409637347BA4', '0x038C701874C52B20',
        '0xDF773BB80AF0C2F9', '0x15D5017AA3800F1E', '0x554EC897971DCF18', '0xC48AACFD77448F11',
        '0x663CFC5276CFCEAB', '0xA8AA71AD0EECCFEA', '0x17D0F99D09B254CD', '0xC5050714D7852A85',
        '0x57459F680F660A9F', '0x35CE7BC91E0F3535', '0xC8B80BC5F6082B14', '0xF32E9A8AD1330DAA',
        '0x9AB59A717BC7F74D', '0x7E98AD7EB9DDF59E', '0x377F90EF6D2719F0', '0xBB3EB8601B7E9133',
        '0x6CE02E64CFA45053', '0x4A59DBDDD9A34F45', '0xD7594C7965872E3B', '0x170A5A45AE7877A0',
        '0x0314C3596BF55EB0', '0x4A4698749A3DF602', '0xDCE0471FD437E359', '0xDBD14A8C2AB0CE32'
    ]
}

# Data Processing
df_left_raw = pd.DataFrame(left_data)
df_join_raw = pd.DataFrame(joined_data)

df_left_raw["timestamp"] = pd.to_datetime(df_left_raw["timestamp"])
df_join_raw["timestamp"] = pd.to_datetime(df_join_raw["timestamp"])

# De-duplicate to keep the first event for each node
df_join = df_join_raw.sort_values("timestamp").drop_duplicates(subset="node_address", keep="first").reset_index(drop=True)
df_left = df_left_raw.sort_values("timestamp").drop_duplicates(subset="node_address", keep="first").reset_index(drop=True)

# Matching and Calculation
# Find common addresses in both datasets
join_set = set(df_join["node_address"])
left_set = set(df_left["node_address"])
common_addrs = sorted(list(join_set & left_set))

# Calculate adjusted left time
TIMEOUT_DELTA = pd.to_timedelta(TIMEOUT_VALUE_SECONDS, unit='s')
df_left["adjusted_left_time"] = df_left["timestamp"] - TIMEOUT_DELTA

# Filter for common addresses before merging
df_join_m = df_join[df_join["node_address"].isin(common_addrs)]
df_left_m = df_left[df_left["node_address"].isin(common_addrs)]

# Merge the two dataframes
merged = pd.merge(
    df_left_m[["node_address", "timestamp", "adjusted_left_time"]],
    df_join_m[["node_address", "timestamp"]],
    on="node_address",
    suffixes=("_left", "_joined"),
)
if merged.empty:
    raise RuntimeError("No matching addresses found after filtering.")

# Calculate the final time difference
merged["final_difference"] = merged["timestamp_joined"] - merged["adjusted_left_time"]

result = merged[["node_address", "timestamp_left", "adjusted_left_time", "timestamp_joined", "final_difference"]].copy()
result = result.rename(columns={
    "timestamp_left": "Original_Left_Time",
    "adjusted_left_time": f"Adjusted_Left_Time (-{TIMEOUT_VALUE_SECONDS:.5f}s)",
    "timestamp_joined": "Rejoined_Time",
    "final_difference": "Final_Time_Difference",
})

# Calculate difference in seconds and clip negative values
result["diff_seconds_raw"] = result["Final_Time_Difference"].dt.total_seconds()
result["diff_seconds"] = result["diff_seconds_raw"].clip(lower=0.0)

# Save rows with negative differences for inspection
neg_rows = result[result["diff_seconds_raw"] < 0].copy()
if not neg_rows.empty:
    neg_rows.to_csv("gateway_handover_40nodes_negative_rows.csv", index=False)
    print(f"Found {len(neg_rows)} negative values; saved to 'gateway_handover_40nodes_negative_rows.csv'.")

# Create ranks for plotting
result = result.sort_values("diff_seconds", kind="mergesort").reset_index(drop=True)
result["rank"] = np.arange(1, len(result) + 1)
result["rank_label"] = result["rank"].apply(lambda i: f"N{i:03d}")

# Save a rank-to-address mapping file
result[["rank_label", "node_address", "diff_seconds", "Original_Left_Time", "Rejoined_Time"]].to_csv(
    "gateway_handover_40nodes_rank_mapping.csv", index=False
)
print(f"Rank mapping saved to 'gateway_handover_40nodes_rank_mapping.csv' ({len(result)} entries)")

# Statistics
avg = result["diff_seconds"].mean()
n = len(result)
print(f"Sample Size N={n} | Average Rejoin Delay: {avg:.3f}s")

# Visualization
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams.update({'font.size': 16})

x = result["rank"].values
y = result["diff_seconds"].values

plt.figure(figsize=(16, 8))
plt.vlines(x, 0, y, linewidth=1.5)
plt.scatter(x, y, s=24)

plt.axhline(avg, linestyle="--", linewidth=2.5, label=f"Average = {avg:.3f}s")

# Configure x-axis ticks
step = 5
ticks = sorted(set([1] + list(range(step, n, step)) + [n]))
plt.xticks(ticks, [f"N{i:02d}" for i in ticks], fontsize=14)

plt.title("Gateway Handover Time — 40 Nodes", fontsize=24, fontweight='bold')
plt.xlabel("Node Rank (N01…N40)", fontsize=18)
plt.ylabel("Rejoin Delay (seconds)", fontsize=18)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(loc="upper left", prop={'size': 16})
plt.tight_layout()

plt.savefig("gateway_handover_40nodes_lollipop.pdf", bbox_inches='tight')
plt.show()
