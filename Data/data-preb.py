import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import joblib


#df = pd.read_csv('1. OfficeIndoorClimate.csv', delimiter=',', header=0, index_col=None, usecols=[1], dtype={'Column1': float})
#df = pd.read_csv('2. PHMdataChallange.csv', delimiter=';', decimal=',', header=0, index_col=None, usecols=[8], dtype={'Column8': float})
#df = pd.read_csv('3. DailyDelhiClimate.csv', delimiter=',', header=0, index_col=None, usecols=[1], dtype={'Column1': float})


data_array = []
arrays = []
universal_quartiles = []
prediction_array = []


def main():
    global data_array, arrays, universal_quartiles, prediction_array

    df = extract_data(6, 'Column7')
    #df = extract_data(9, 'Column10')

    # This will display the first few rows of the DataFrame to check it's loaded correctly
    print(str(df.head()) + "\n -----------------")

    # Convert the entire DataFrame to a NumPy array
    data_array = df.to_numpy()
    print(str(data_array) + "\n -----------------")

    # Split the array into 10 parts
    arrays = np.array_split(data_array, len(data_array)/150)

    sequences = 10
    evaluate_array = []
    number_of_bins = 4

    for i in range(len(arrays), sequences, -1):
        array = arrays.pop()
        if i == sequences + 1:
            prediction_array = array
            evaluate_array = array

    prediction_array = np.array(prediction_array)
    evaluate_array = np.array(evaluate_array)

    # Find the overall min and max values to determine universal quartile ranges
    overall_min = np.min(data_array)
    overall_max = np.max(data_array)

    # Determine the universal quartiles
    universal_quartiles = np.linspace(overall_min, overall_max, number_of_bins + 1)
    print("Universal quartiles: " + str(universal_quartiles) + "\n -----------------") 

    continuous_plot()
    discrete_plot()
    bar_plot()

    model = create_hmm()
    #model = load_model()
    hidden_states = hmm_predict(model, prediction_array)
    save_model(model)

    print_model_parameters(model)

    hidden_to_observations = generate_observations(model, hidden_states)
    print("Predicted observations for the new sequence: " + str(hidden_to_observations) + "\n -----------------")

    evaluate_array = discretize_data([evaluate_array], universal_quartiles)[0][0]
    print("Actual observations for the new sequence: " + str(evaluate_array) + "\n -----------------")

def create_hmm():
    discretized_sequences, labels = discretize_data(arrays, universal_quartiles)

    # Combine sequences into a single sequence for HMM training
    concatenated_sequences = np.concatenate(discretized_sequences).reshape(-1, 1)
    
    # Train HMM
    model = hmm.CategoricalHMM(n_components=3, n_iter=1000)
    model.fit(concatenated_sequences)

    return model

def save_model(model, filename='hmm_model4.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='hmm_model4.pkl'):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def generate_observations(model, hidden_states):
    observations = []
    for state in hidden_states:
        emission_probabilities = model.emissionprob_[state]
        observation = np.argmax(emission_probabilities)
        observations.append(observation)
    return observations

def hmm_predict(model, prediction_array):
    # Discretize the test arrays
    new_discretized_sequence, _ = discretize_data([prediction_array], universal_quartiles)
    new_concatenated_sequence = np.concatenate(new_discretized_sequence).reshape(-1, 1)
    
    # Predict the hidden states for the new sequence
    predicted_states = model.predict(new_concatenated_sequence)

    np.set_printoptions(threshold=np.inf)

    
    # Print detailed information
    print("Discrete test arrays:")
    print(new_discretized_sequence)
    print("\nPredicted states for the new sequence:")
    print(predicted_states)
    return predicted_states


def discretize_data(arrays, universal_quartiles):
    labels = [f'{universal_quartiles[i]:.2f}-{universal_quartiles[i+1]:.2f}' for i in range(len(universal_quartiles) - 1)]
    discretized_sequences = []

    for arr in arrays:
        flattened_array = arr.flatten()
        data_binned = pd.cut(flattened_array, bins=universal_quartiles, labels=range(len(labels)), include_lowest=True)
        discretized_sequences.append(data_binned.to_numpy())

    return discretized_sequences, labels


def print_model_parameters(model):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.4f}'.format})

    initial_state_distribution = model.startprob_
    transition_matrix = model.transmat_
    emission_matrix = model.emissionprob_

    print("Initial state distribution (pi):\n", initial_state_distribution)
    print("Transition matrix (A):\n", transition_matrix)
    print("Emission matrix (B):\n", emission_matrix)
    print("-----------------")

    np.set_printoptions(suppress=False)


def extract_data(column, column_name):
    # Read the first 1000 lines of the CSV file, focusing on columns 2 and 7
    file_path = '2. PHMdataChallange.csv'
    df = pd.read_csv(file_path, delimiter=';', decimal=',', nrows=10000, usecols=[1, column], header=0, names=['Column2', column_name], dtype={'Column2': int, column_name: float})

    # Extract columns of interest
    col2 = df.iloc[:, 0]  # Column 2 (usecols=[1] means it's now the first column in df)
    col7 = df.iloc[:, 1]  # Column 7 (usecols=[6] means it's now the second column in df)

    # Initialize variables to store sequences
    sequences = []
    current_sequence = []
    current_values_col7 = []

    # Process the columns to identify sequences
    for i in range(len(col2)):
        if col2[i] == 1:
            # If we are starting a new sequence, save the previous one if it exists
            if len(current_sequence) == 150:
                sequence_df = pd.DataFrame(current_values_col7, columns=[column_name])
                sequences.append(sequence_df)
            # Reset for the new sequence
            current_sequence = [col2[i]]
            current_values_col7 = [col7[i]]
        elif len(current_sequence) > 0 and col2[i] == current_sequence[-1] + 1:
            # Continue the sequence if it ascends correctly
            current_sequence.append(col2[i])
            current_values_col7.append(col7[i])
            # If we reach 150, finalize this sequence
            if len(current_sequence) == 150:
                sequence_df = pd.DataFrame(current_values_col7, columns=[column_name])
                sequences.append(sequence_df)
                current_sequence = []
                current_values_col7 = []

    # Ensure the last sequence is saved if it reached 150 values
    if len(current_sequence) == 150:
        sequence_df = pd.DataFrame(current_values_col7, columns=[column_name])
        sequences.append(sequence_df)

    # Combine all sequences into a single DataFrame
    combined_df = pd.concat(sequences, ignore_index=True)
    return combined_df


# Plotting the data
def continuous_plot():
    labels = []
    for i in range(len(arrays)):
        labels.append(f'Part {i+1}')

    # Initialize a figure and axis for plotting
    fig, ax = plt.subplots()

    # Assume the data has at least one numeric column; plotting the first numeric column
    for i, arr in enumerate(arrays):
        # Extract the length of the previous arrays to determine the starting index for x
        start_index = sum(len(a) for a in arrays[:i])
        x = range(start_index, start_index + len(arr))
        
        # Assuming column index 0 is what you want to plot
        y = arr[:, 0] if arr.ndim > 1 else arr  # Adjust based on your data structure
        
        ax.plot(x, y, label=labels[i])

    # Add legend to the plot
    ax.legend()

    # Add title and labels
    ax.set_title('Continuous Plot of Split Data')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # Show the plot
    plt.show()

def discretize_data_plot(arrays, universal_quartiles):
    frequency_counts = {}
    labels = [f'{universal_quartiles[i]:.2f}-{universal_quartiles[i+1]:.2f}' for i in range(len(universal_quartiles) - 1)]

    for i, arr in enumerate(arrays):
        # Flatten the array to remove extra dimensions
        flattened_array = arr.flatten()
        # Create bins for the percentile ranges using the universal quartiles
        data_binned = pd.cut(flattened_array, bins=universal_quartiles, labels=labels, include_lowest=True)
        # Calculate the frequency of data points in each bin
        frequency_counts[i] = data_binned.value_counts().sort_index()

    return frequency_counts, labels

def discrete_plot():
    global arrays, universal_quartiles
    frequency_counts, labels = discretize_data_plot(arrays, universal_quartiles)
    
    n_arrays = len(arrays)
    fig, axes = plt.subplots(nrows=n_arrays, ncols=1, figsize=(10, 5 * n_arrays), sharex=True)

    if n_arrays == 1:
        axes = [axes]  # Make axes iterable if there's only one plot

    for i, arr in enumerate(arrays):
        frequency = frequency_counts[i]

        # Plot each sub-array
        axes[i].bar(frequency.index, frequency.values)
        axes[i].set_title(f'Part {i+1}')
        axes[i].set_ylabel('Frequency')
        if i == n_arrays - 1:
            axes[i].set_xlabel('Quartile Range')

    plt.tight_layout()
    plt.show()

def bar_plot():
    global arrays, universal_quartiles
    frequency_counts, labels = discretize_data_plot(arrays, universal_quartiles)
    
    n_groups = len(labels)  # Number of groups (quartiles)
    index = np.arange(n_groups)  # Group positions
    bar_width = 0.15  # Width of the bars for each part

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(arrays)):
        frequencies = frequency_counts[i]  # Retrieve frequency counts for part i
        # Ensure the frequencies are in the correct order
        ordered_frequencies = [frequencies[label] if label in frequencies else 0 for label in labels]
        # Plotting
        ax.bar(index + i * bar_width, ordered_frequencies, bar_width, label=f'Part {i+1}')

    # Set the plot details
    ax.set_xlabel('Quartile Range')
    ax.set_ylabel('Frequency')
    ax.set_title('Bar Chart Quartile Distribution')
    ax.set_xticks(index + bar_width * 2 - bar_width)
    ax.set_xticklabels(labels)
    ax.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
