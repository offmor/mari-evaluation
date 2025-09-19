import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
TIMEOUT_VALUE_SECONDS = 1.24415

# Raw Data
# 1) NODE_LEFT events (60 entries)
left_data = {
    'timestamp': [
        '2025-08-21T14:53:44.654873','2025-08-21T14:53:45.325979','2025-08-21T14:53:45.340823',
        '2025-08-21T14:53:45.537942','2025-08-21T14:53:45.603825','2025-08-21T14:53:45.823399',
        '2025-08-21T14:53:46.292894','2025-08-21T14:53:46.524820','2025-08-21T14:53:46.601026',
        '2025-08-21T14:53:47.011536','2025-08-21T14:53:47.263947','2025-08-21T14:53:47.264230',
        '2025-08-21T14:53:47.543117','2025-08-21T14:53:47.543324','2025-08-21T14:53:47.543450',
        '2025-08-21T14:53:47.850723','2025-08-21T14:53:48.009883','2025-08-21T14:53:48.539005',
        '2025-08-21T14:53:48.553890','2025-08-21T14:53:48.637520','2025-08-21T14:53:48.746105',
        '2025-08-21T14:53:48.814845','2025-08-21T14:53:49.003852','2025-08-21T14:53:49.014889',
        '2025-08-21T14:53:49.260915','2025-08-21T14:53:49.270831','2025-08-21T14:53:49.273893',
        '2025-08-21T14:53:49.552855','2025-08-21T14:53:49.761884','2025-08-21T14:53:49.995058',
        '2025-08-21T14:53:50.018887','2025-08-21T14:53:50.057033','2025-08-21T14:53:50.057284',
        '2025-08-21T14:53:50.070694','2025-08-21T14:53:50.272995','2025-08-21T14:53:50.528864',
        '2025-08-21T14:53:50.529178','2025-08-21T14:53:50.561151','2025-08-21T14:53:50.581902',
        '2025-08-21T14:53:50.582744','2025-08-21T14:53:50.589925','2025-08-21T14:53:50.739651',
        '2025-08-21T14:53:50.739850','2025-08-21T14:53:50.783749','2025-08-21T14:53:51.032282',
        '2025-08-21T14:53:51.033867','2025-08-21T14:53:51.058867','2025-08-21T14:53:51.069008',
        '2025-08-21T14:53:51.088841','2025-08-21T14:53:51.224630','2025-08-21T14:53:51.290246',
        '2025-08-21T14:53:51.544973','2025-08-21T14:53:51.590145','2025-08-21T14:53:51.595964',
        '2025-08-21T14:53:51.596939','2025-08-21T14:53:51.814118','2025-08-21T14:53:51.848884',
        '2025-08-21T14:53:52.020956','2025-08-21T14:53:52.564016','2025-08-21T14:53:52.818025'
    ],
    'node_address': [
        '0x3879F53E18494531','0x4A4698749A3DF602','0x93B9409637347BA4','0xB459710B9A8D5037',
        '0x3EFC6553DEE09DD1','0x151148302FD6F5A2','0x6CE02E64CFA45053','0xED02BE5529D07C34',
        '0xC82CA0F38D2A15E9','0x150E3BE9B7C78815','0x0314C3596BF55EB0','0xC48AACFD77448F11',
        '0x55DA7FF0F238A844','0xA95D8A2DF55E6C44','0xE26E0464A4FE9291','0x7AA3DC93896A316C',
        '0x8C3BDCC588440DCA','0xF32E9A8AD1330DAA','0xBB3EB8601B7E9133','0x42ADEC8748A3212B',
        '0x44E5B4534F8DBC91','0xA8AA71AD0EECCFEA','0xDC878301DB6B0590','0xFB986FD3DFD36E63',
        '0xB7B971F84F29BAE7','0xD7594C7965872E3B','0x7E98AD7EB9DDF59E','0x663CFC5276CFCEAB',
        '0x3EF06E4B72E98CCD','0xCD9FC2F10292D689','0x554EC897971DCF18','0x170A5A45AE7877A0',
        '0x377F90EF6D2719F0','0x15D5017AA3800F1E','0xAD89E8BD0C51208E','0x0AB87C44C20B0149',
        '0xC5050714D7852A85','0x9AB59A717BC7F74D','0x34CC9F023F7278AC','0x70F6661004F7ACD8',
        '0x040877A02041BD1D','0x9579960F203ADE95','0xBEBAD533DADE0490','0x4A59DBDDD9A34F45',
        '0x35CE7BC91E0F3535','0x17D0F99D09B254CD','0x038C701874C52B20','0xDF773BB80AF0C2F9',
        '0x13D398E530317D6F','0xFDB6CC02C854CF1E','0x57459F680F660A9F','0x44B408F1766E0BF5',
        '0xC4FAE708729C9496','0xC063EDB67C4CCA58','0xDBD14A8C2AB0CE32','0xDCE0471FD437E359',
        '0xC8B80BC5F6082B14','0x613FD651ED0CEF84','0x0C9A20E2E5E6F54C','0x588D2578218A459B'
    ]
}

# 2) NODE_JOINED events (60 entries)
joined_data = {
    'timestamp': [
        '2025-08-21T14:53:43.677337','2025-08-21T14:53:44.218255','2025-08-21T14:53:44.240116',
        '2025-08-21T14:53:44.348347','2025-08-21T14:53:44.467235','2025-08-21T14:53:44.684185',
        '2025-08-21T14:53:45.192125','2025-08-21T14:53:45.332040','2025-08-21T14:53:45.733271',
        '2025-08-21T14:53:45.877676','2025-08-21T14:53:46.108132','2025-08-21T14:53:46.177173',
        '2025-08-21T14:53:46.404188','2025-08-21T14:53:46.479177','2025-08-21T14:53:46.631186',
        '2025-08-21T14:53:46.750323','2025-08-21T14:53:47.237297','2025-08-21T14:53:47.355994',
        '2025-08-21T14:53:47.431200','2025-08-21T14:53:47.615023','2025-08-21T14:53:47.670129',
        '2025-08-21T14:53:47.692142','2025-08-21T14:53:47.885208','2025-08-21T14:53:47.907111',
        '2025-08-21T14:53:48.124113','2025-08-21T14:53:48.317986','2025-08-21T14:53:48.395098',
        '2025-08-21T14:53:48.665157','2025-08-21T14:53:48.816005','2025-08-21T14:53:48.836995',
        '2025-08-21T14:53:48.870965','2025-08-21T14:53:48.892041','2025-08-21T14:53:48.958180',
        '2025-08-21T14:53:48.968088','2025-08-21T14:53:49.151063','2025-08-21T14:53:49.346985',
        '2025-08-21T14:53:49.389983','2025-08-21T14:53:49.400111','2025-08-21T14:53:49.412012',
        '2025-08-21T14:53:49.422006','2025-08-21T14:53:49.561969','2025-08-21T14:53:49.616980',
        '2025-08-21T14:53:49.648968','2025-08-21T14:53:49.670992','2025-08-21T14:53:49.810963',
        '2025-08-21T14:53:49.897965','2025-08-21T14:53:50.081965','2025-08-21T14:53:50.190153',
        '2025-08-21T14:53:50.234160','2025-08-21T14:53:50.255181','2025-08-21T14:53:50.272126',
        '2025-08-21T14:53:50.461149','2025-08-21T14:53:50.578956','2025-08-21T14:53:50.644010',
        '2025-08-21T14:53:50.699045','2025-08-21T14:53:50.753281','2025-08-21T14:53:50.845216',
        '2025-08-21T14:53:50.904965','2025-08-21T14:53:51.565036','2025-08-21T14:53:51.927379'
    ],
    'node_address': [
        '0x3879F53E18494531','0x4A4698749A3DF602','0x93B9409637347BA4','0xB459710B9A8D5037',
        '0x3EFC6553DEE09DD1','0x151148302FD6F5A2','0x6CE02E64CFA45053','0xED02BE5529D07C34',
        '0xC82CA0F38D2A15E9','0x150E3BE9B7C78815','0x0314C3596BF55EB0','0xC48AACFD77448F11',
        '0xA95D8A2DF55E6C44','0xE26E0464A4FE9291','0x7AA3DC93896A316C','0x55DA7FF0F238A844',
        '0x8C3BDCC588440DCA','0xBB3EB8601B7E9133','0xF32E9A8AD1330DAA','0x42ADEC8748A3212B',
        '0xA8AA71AD0EECCFEA','0x44E5B4534F8DBC91','0xFB986FD3DFD36E63','0xDC878301DB6B0590',
        '0xB7B971F84F29BAE7','0xD7594C7965872E3B','0x7E98AD7EB9DDF59E','0x663CFC5276CFCEAB',
        '0x15D5017AA3800F1E','0x170A5A45AE7877A0','0x3EF06E4B72E98CCD','0xCD9FC2F10292D689',
        '0x377F90EF6D2719F0','0x554EC897971DCF18','0xAD89E8BD0C51208E','0x70F6661004F7ACD8',
        '0x040877A02041BD1D','0x9AB59A717BC7F74D','0x0AB87C44C20B0149','0xC5050714D7852A85',
        '0xBEBAD533DADE0490','0x9579960F203ADE95','0x4A59DBDDD9A34F45','0x34CC9F023F7278AC',
        '0x17D0F99D09B254CD','0x35CE7BC91E0F3535','0x13D398E530317D6F','0x57459F680F660A9F',
        '0xDF773BB80AF0C2F9','0x038C701874C52B20','0xFDB6CC02C854CF1E','0x44B408F1766E0BF5',
        '0xC063EDB67C4CCA58','0xDBD14A8C2AB0CE32','0xDCE0471FD437E359','0xC4FAE708729C9496',
        '0xC8B80BC5F6082B14','0x613FD651ED0CEF84','0x588D2578218A459B','0x0C9A20E2E5E6F54C'
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
    neg_rows.to_csv("gateway_handover_60nodes_negative_rows.csv", index=False)
    print(f"Found {len(neg_rows)} negative values; saved to 'gateway_handover_60nodes_negative_rows.csv'.")

# Create ranks for plotting
result = result.sort_values("diff_seconds", kind="mergesort").reset_index(drop=True)
result["rank"] = np.arange(1, len(result) + 1)
result["rank_label"] = result["rank"].apply(lambda i: f"N{i:03d}")

# Save a rank-to-address mapping file
result[["rank_label", "node_address", "diff_seconds", "Original_Left_Time", "Rejoined_Time"]].to_csv(
    "gateway_handover_60nodes_rank_mapping.csv", index=False
)
print(f"Rank mapping saved to 'gateway_handover_60nodes_rank_mapping.csv' ({len(result)} entries)")

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

plt.title("Gateway Handover Time — 60 Nodes", fontsize=24, fontweight='bold')
plt.xlabel("Node Rank (N01…N60)", fontsize=18)
plt.ylabel("Rejoin Delay (seconds)", fontsize=18)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(loc="upper left", prop={'size': 16})
plt.tight_layout()

plt.savefig("gateway_handover_60nodes_lollipop.pdf", bbox_inches='tight')
plt.show()
