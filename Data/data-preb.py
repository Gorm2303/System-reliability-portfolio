import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df = pd.read_csv('1. OfficeIndoorClimate.csv', delimiter=',', header=0, index_col=None, usecols=[1], dtype={'Column1': float})
#df = pd.read_csv('2. PHMdataChallange.csv', delimiter=';', decimal=',', header=0, index_col=None, usecols=[8], dtype={'Column8': float})
df = pd.read_csv('3. DailyDelhiClimate.csv', delimiter=',', header=0, index_col=None, usecols=[1], dtype={'Column1': float})


# This will display the first few rows of the DataFrame to check it's loaded correctly
print(str(df.head()) + "\n -----------------")

# Convert the entire DataFrame to a NumPy array
data_array = df.to_numpy()
print(str(data_array) + "\n -----------------")

# Split the array into 5 parts
arrays = np.array_split(data_array, 10)

# Find the overall min and max values to determine universal quartile ranges
overall_min = np.min(data_array)
overall_max = np.max(data_array)

# Determine the universal quartiles
universal_quartiles = np.linspace(overall_min, overall_max, 5)
print ("Universal quartiles: " + str(universal_quartiles) + "\n -----------------")

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


def discrete_plot():
    # Prepare the plot
    n_arrays = len(arrays)
    fig, axes = plt.subplots(nrows=n_arrays, ncols=1, figsize=(10, 5 * n_arrays), sharex=True)
    
    if n_arrays == 1:
        axes = [axes]  # Make axes iterable if there's only one plot

    # Generate labels based on universal quartiles
    labels = [f'{universal_quartiles[i]:.2f}°C-{universal_quartiles[i+1]:.2f}°C' for i in range(len(universal_quartiles) - 1)]

    for i, arr in enumerate(arrays):
        # Flatten the array to remove extra dimensions
        flattened_array = arr.flatten()
        # Create bins for the percentile ranges using the universal quartiles
        data_binned = pd.cut(flattened_array, bins=universal_quartiles, labels=labels, include_lowest=True)
        # Calculate the frequency of data points in each bin
        frequency = data_binned.value_counts().sort_index()

        # Plot each sub-array
        axes[i].bar(frequency.index, frequency.values)
        axes[i].set_title(f'Part {i+1}')
        axes[i].set_ylabel('Frequency')
        if i == n_arrays - 1:
            axes[i].set_xlabel('Quartile Range')

    plt.tight_layout()
    plt.show()


def bar_plot():
    # Initialize the dictionary
    frequency_counts = {}
    # Labels for the quartiles
    #labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    labels = ["6.00°C-14.17°C","14.17°C-22.35°C", "22.35°C-30.53°C", "30.53°C-38.71°C"]
    n_groups = len(labels)  # Number of groups (quartiles)
    index = np.arange(n_groups)  # Group positions
    bar_width = 0.15  # Width of the bars for each part

    # Calculate and store frequency counts for each part
    for i, arr in enumerate(arrays):
        # Flatten the array (if it has more than one dimension) and categorize data into bins
        flattened_array = arr.flatten()
        categories = pd.cut(flattened_array, bins=universal_quartiles, labels=labels, include_lowest=True)
        
        # Count the frequency of each category
        frequency_counts[i] = categories.value_counts().sort_index()

    # Display what we have in the dictionary
    for part, counts in frequency_counts.items():
        print(f"Part {part + 1}:")
        print(counts)
        print()  # This adds an empty line for better readability

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each part's frequency distribution
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



def main():
    continuous_plot()
    discrete_plot()
    bar_plot()

if __name__ == "__main__":
    main()