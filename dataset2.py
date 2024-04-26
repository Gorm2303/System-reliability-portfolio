import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file
file_path = './Data/2. PHMdataChallange.xlsx'
data = pd.read_excel(file_path)

# Extract the 'Column7' and save it into an array
column7_data = data['Column7'].values

# Number of sequences
num_sequences = 5

# Determine the length of each sequence
sequence_length = len(column7_data) // num_sequences

# Create sequences
sequences = [column7_data[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]

# Plot the sequences in a continuous format
plt.figure(figsize=(12, 6))
for i, seq in enumerate(sequences):
    plt.plot(range(i * sequence_length, (i + 1) * sequence_length), seq, label=f'Sequence {i+1}')
plt.title('Continuous Plot of Column7 Sequences')
plt.xlabel('Index')
plt.ylabel('Values of Column7')
plt.legend()
plt.show()

# Define bins for the data
column7_min, column7_max = column7_data.min(), column7_data.max()
bin_edges = np.linspace(column7_min, column7_max, num=6)  # Create 5 bins

# Categorize the sequences using these bins
binned_sequences = [np.digitize(seq, bins=bin_edges) for seq in sequences]

# Plot the discretized sequences with bins
plt.figure(figsize=(12, 6))
for i, seq in enumerate(binned_sequences):
    plt.step(range(i * sequence_length, (i + 1) * sequence_length), seq, where='mid', label=f'Sequence {i+1}')
plt.title('Binned Plot of Column7 Sequences')
plt.xlabel('Index')
plt.ylabel('Bin Category')
plt.yticks(range(1, 7), labels=[f'Bin {i}' for i in range(1, 7)])  # Label bins for clarity
plt.legend()
plt.show()

# Create sequences of the same length with constant "1"
constant_sequences = [np.ones(len(seq), dtype=int) for seq in sequences]
