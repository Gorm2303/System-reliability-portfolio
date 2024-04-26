import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = './data/OfficeIndoorClimate.csv'
data = pd.read_csv(file_path)

# Extract the 'ki [C]' column
ki_data = data['ki [C]'].values

# Number of sequences
num_sequences = 5

# Determine the length of each sequence (assuming equal distribution and rounding down if not a perfect fit)
sequence_length = len(ki_data) // num_sequences

# Create sequences
sequences = [ki_data[i*sequence_length:(i+1)*sequence_length] for i in range(num_sequences)]

# Check the sequences and their lengths
sequences_lengths = [len(seq) for seq in sequences]
sequences_lengths

# Plot the sequences in a continuous format
plt.figure(figsize=(12, 6))
for i, seq in enumerate(sequences):
    plt.plot(range(i * sequence_length, (i + 1) * sequence_length), seq, label=f'Sequence {i+1}')

plt.title('Continuous Plot of ki [C] Sequences')
plt.xlabel('Index')
plt.ylabel('Temperature (ki [C])')
plt.legend()
plt.show()

# Convert sequences to discrete format by rounding the values
discrete_sequences = [np.round(seq).astype(int) for seq in sequences]

# Plot the discrete sequences
plt.figure(figsize=(12, 6))
for i, seq in enumerate(discrete_sequences):
    plt.step(range(i * sequence_length, (i + 1) * sequence_length), seq, where='mid', label=f'Sequence {i+1}')

plt.title('Discrete Plot of ki [C] Sequences')
plt.xlabel('Index')
plt.ylabel('Temperature (ki [C], rounded)')
plt.legend()
plt.show()

# Create sequences of the same length with constant "1"
constant_sequences = [np.ones(len(seq), dtype=int) for seq in sequences]

