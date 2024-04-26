import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = './Data/3. DailyDelhiClimate.csv'
data = pd.read_csv(file_path)

# Extract the 'wind_speed' column
wind_speed_data = data['wind_speed'].values

# Number of sequences
num_sequences = 5

# Determine the length of each sequence
sequence_length = len(wind_speed_data) // num_sequences

# Create sequences
sequences = [wind_speed_data[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]

# Define bins for the data, creating 10 bins
bin_edges = np.linspace(wind_speed_data.min(), wind_speed_data.max(), num=11)
bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]

# Categorize the sequences using these bins and set 'right' to True to include the right edge in the bin
binned_indices = [np.digitize(seq, bins=bin_edges, right=True) - 1 for seq in sequences]  # subtract 1 to match index

# Flatten sequences and corresponding bin indices
flat_sequences = np.hstack(sequences)
flat_binned_indices = np.hstack(binned_indices)

# Prepare a DataFrame for CSV export with bin labels
df_binned_data = pd.DataFrame({
    'Wind_Speed': flat_sequences,
    'Bin_Category': [bin_labels[idx] if idx < len(bin_labels) else 'Out of range' for idx in flat_binned_indices]
})

# Save to CSV
df_binned_data.to_csv('./Data/Wind_Speed_Binned_Data.csv', index=False)

# Plot the sequences in a continuous format
plt.figure(figsize=(12, 6))
for i, seq in enumerate(sequences):
    plt.plot(range(i * sequence_length, (i + 1) * sequence_length), seq, label=f'Sequence {i+1}')
plt.title('Continuous Plot of Wind Speed Sequences')
plt.xlabel('Index')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df_binned_data['Wind_Speed'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Wind Speeds')
plt.xlabel('Wind Speed')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Count occurrences in each bin
bin_counts = df_binned_data['Bin_Category'].value_counts().sort_index()

# Plot bar plot
plt.figure(figsize=(12, 6))
bin_counts.plot(kind='bar', color='green', alpha=0.6)
plt.title('Wind Speed Counts in Each Bin Category')
plt.xlabel('Bin Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Create sequences of the same length with constant "1"
constant_sequences = [np.ones(len(seq), dtype=int) for seq in sequences]
