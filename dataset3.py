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

# Plot the sequences in a continuous format
plt.figure(figsize=(12, 6))
for i, seq in enumerate(sequences):
    plt.plot(range(i * sequence_length, (i + 1) * sequence_length), seq, label=f'Sequence {i+1}')
plt.title('Continuous Plot of Wind Speed Sequences')
plt.xlabel('Index')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()

# Determine the min and max values in wind_speed_data to help define bins
wind_speed_min, wind_speed_max = wind_speed_data.min(), wind_speed_data.max()

# Define bins for the data
wind_speed_bin_edges = np.linspace(wind_speed_min, wind_speed_max, num=6)  # Create 5 bins across the range

# Categorize the sequences using these bins
wind_speed_binned_sequences = [np.digitize(seq, bins=wind_speed_bin_edges) for seq in sequences]

# Plot the binned sequences
plt.figure(figsize=(12, 6))
for i, seq in enumerate(wind_speed_binned_sequences):
    plt.step(range(i * sequence_length, (i + 1) * sequence_length), seq, where='mid', label=f'Sequence {i+1}')
plt.title('Binned Plot of Wind Speed Sequences')
plt.xlabel('Index')
plt.ylabel('Bin Category')
plt.yticks(range(1, 7), labels=[f'Bin {i}' for i in range(1, 7)])  # Label bins for clarity
plt.legend()
plt.show()

# Create sequences of the same length with constant "1"
constant_sequences = [np.ones(len(seq), dtype=int) for seq in sequences]
