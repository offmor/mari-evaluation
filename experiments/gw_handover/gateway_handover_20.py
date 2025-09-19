import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TIMEOUT_VALUE_SECONDS = 1.24415

# Raw Data
# NODE_LEFT events
left_data = {
    'timestamp': [
        '2025-08-21T14:44:40.253644', '2025-08-21T14:44:41.757151', '2025-08-21T14:44:42.743532',
        '2025-08-21T14:44:42.761904', '2025-08-21T14:44:42.981801', '2025-08-21T14:44:42.982038',
        '2025-08-21T14:44:42.982194', '2025-08-21T14:44:42.992764', '2025-08-21T14:44:42.998154',
        '2025-08-21T14:44:42.999853', '2025-08-21T14:44:43.007861', '2025-08-21T14:44:43.009973',
        '2025-08-21T14:44:43.014778', '2025-08-21T14:44:43.227167', '2025-08-21T14:44:43.733757',
        '2025-08-21T14:44:44.977051', '2025-08-21T14:44:45.215419', '2025-08-21T14:44:45.478176',
    ],
    'node_address': [
        '0x44B408F1766E0BF5', '0xDF773BB80AF0C2F9', '0x17D0F99D09B254CD', '0x588D2578218A459B',
        '0xC82CA0F38D2A15E9', '0xDBD14A8C2AB0CE32', '0x038C701874C52B20', '0x57459F680F660A9F',
        '0xBB3EB8601B7E9133', '0x0C9A20E2E5E6F54C', '0x9AB59A717BC7F74D', '0x663CFC5276CFCEAB',
        '0x4A4698749A3DF602', '0x170A5A45AE7877A0', '0x4A59DBDDD9A34F45', '0x15D5017AA3800F1E',
        '0x93B9409637347BA4', '0xA8AA71AD0EECCFEA',
    ]
}

# NODE_JOINED events
joined_data = {
    'timestamp': [
        '2025-08-21T14:44:39.219545', '2025-08-21T14:44:40.712362', '2025-08-21T14:44:41.718367',
        '2025-08-21T14:44:41.761358', '2025-08-21T14:44:41.891262', '2025-08-21T14:44:41.901273',
        '2025-08-21T14:44:41.923256', '2025-08-21T14:44:41.935237', '2025-08-21T14:44:41.966329',
        '2025-08-21T14:44:42.010363', '2025-08-21T14:44:42.035059', '2025-08-21T14:44:42.035236',
        '2025-08-21T14:44:42.065291', '2025-08-21T14:44:42.172386', '2025-08-21T14:44:42.227443',
        '2025-08-21T14:44:42.269143', '2025-08-21T14:44:42.778312', '2025-08-21T14:44:43.947337',
        '2025-08-21T14:44:44.239227', '2025-08-21T14:44:44.488340',
    ],
    'node_address': [
        '0x44B408F1766E0BF5', '0xDF773BB80AF0C2F9', '0x588D2578218A459B', '0xDCE0471FD437E359',
        '0x377F90EF6D2719F0', '0x4A4698749A3DF602', '0xC82CA0F38D2A15E9', '0xDBD14A8C2AB0CE32',
        '0x17D0F99D09B254CD', '0x0C9A20E2E5E6F54C', '0x57459F680F660A9F', '0x038C701874C52B20',
        '0xBB3EB8601B7E9133', '0x663CFC5276CFCEAB', '0x9AB59A717BC7F74D', '0x170A5A45AE7877A0',
        '0x4A59DBDDD9A34F45', '0x15D5017AA3800F1E', '0x93B9409637347BA4', '0xA8AA71AD0EECCFEA',
    ]
}

# Data validation
if len(left_data['timestamp']) != len(left_data['node_address']) or \
   len(joined_data['timestamp']) != len(joined_data['node_address']):
    print("Error: Data lists have inconsistent lengths.")
    exit()

# Data Processing and Calculation
df_left = pd.DataFrame(left_data)
df_joined = pd.DataFrame(joined_data)

df_left['timestamp'] = pd.to_datetime(df_left['timestamp'])
df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])

timeout_delta = pd.to_timedelta(TIMEOUT_VALUE_SECONDS, unit='s')
df_left['adjusted_left_time'] = df_left['timestamp'] - timeout_delta

merged_df = pd.merge(
    df_left,
    df_joined,
    on='node_address',
    suffixes=('_left', '_joined')
)

if merged_df.empty:
    print("Error: No matching LEFT and JOINED events found for any node_address.")
    exit()

merged_df['final_difference'] = merged_df['timestamp_joined'] - merged_df['adjusted_left_time']

# Format and Print Results Table
result_table = merged_df[[
    'node_address', 'timestamp_left', 'adjusted_left_time', 'timestamp_joined', 'final_difference'
]].rename(columns={
    'timestamp_left': 'Original_Left_Time',
    'adjusted_left_time': 'Adjusted_Left_Time (-1.24s)',
    'timestamp_joined': 'Rejoined_Time',
    'final_difference': 'Final_Time_Difference'
})

print("=" * 120)
print("Node Rejoin Time Calculation Results (Table)")
print("-" * 120)
print(result_table.to_string())
print("=" * 120)

# Visualization
# Prepare data for plotting
plot_data = result_table.sort_values(by='Final_Time_Difference').copy()
plot_data['final_difference_seconds'] = plot_data['Final_Time_Difference'].dt.total_seconds()

# Calculate average time
mean_time = plot_data['final_difference_seconds'].mean()
print(f"\nAverage rejoin time for all nodes: {mean_time:.4f} seconds")

# Create plot
plt.figure(figsize=(16, 9))
bars = plt.bar(plot_data['node_address'], plot_data['final_difference_seconds'], color='cornflowerblue')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', va='bottom', ha='center', fontsize=9)

plt.title('Rejoin Time Difference for Each Node', fontsize=20)
plt.xlabel('Node Address', fontsize=12)
plt.ylabel('Time Difference (seconds)', fontsize=12)

plt.xticks(rotation=90)
plt.ylim(top=plot_data['final_difference_seconds'].max() * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add average line and legend
plt.axhline(y=mean_time, color='r', linestyle='--', linewidth=2, label=f'Average: {mean_time:.4f}s')
plt.legend()

plt.tight_layout()
plt.show()
