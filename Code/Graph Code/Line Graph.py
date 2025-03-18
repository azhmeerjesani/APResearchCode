import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_line_chart_accuracy_vs_k():
    """
    This script creates a high-resolution line chart (1200 DPI) plotting the average accuracy
    versus k for each noise type. It loads the CSV files from the specified folder,
    computes the mean and standard deviation of accuracy (over 50 trials) for each k,
    and plots one colored line per noise type with error bands representing ±1 standard deviation.
    A legend is added for clarity.
    """

    # Absolute folder path containing all CSV outputs
    csv_folder = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\Outputs\CSV Outputs"

    # List of noise types to process
    noise_types = [
        "Noiseless",
        "SingleQubit",
        "TwoQubit",
        "ZRotation",
        "TwoQubitXRotation",
        "T1Relaxation",
        "T2Dephasing",
        "MeasurementError",
        "Combined"
    ]

    # Use a Seaborn palette (here, tab10 gives up to 10 distinct colors)
    colors = sns.color_palette("tab10", n_colors=len(noise_types))

    # Create a new figure
    plt.figure(figsize=(12, 8), dpi=1200)

    # Loop through each noise type and plot the line
    for i, noise in enumerate(noise_types):
        if (noise == "SingleQubit"):
            pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_Noisy_*.csv")
        else:
            pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{noise}_*.csv")

        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"No CSV files found for noise type: {noise}")
            continue

        # Load and concatenate CSV files
        df_list = [pd.read_csv(file) for file in matching_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Ensure 'k' is an integer and sort the data by k
        combined_df["k"] = combined_df["k"].astype(int)
        combined_df = combined_df.sort_values(by="k")

        # Group by k and compute mean and std of accuracy
        stats = combined_df.groupby("k")["Accuracy"].agg(["mean", "std"]).reset_index()
        # Extract k, mean, and standard deviation values
        k_values = stats["k"].tolist()
        mean_acc = stats["mean"].tolist()
        std_acc = stats["std"].tolist()

        # Plot the mean accuracy line for this noise type with markers
        plt.plot(k_values, mean_acc, marker="o", label=noise, color=colors[i])
        # Fill the area between (mean - std) and (mean + std)
        plt.fill_between(k_values,
                         [m - s for m, s in zip(mean_acc, std_acc)],
                         [m + s for m, s in zip(mean_acc, std_acc)],
                         color=colors[i], alpha=0.2)

    # Add title, labels, legend, and grid
    plt.title("Average Accuracy vs. k (Grouped by Noise Type)", fontsize=16, fontweight="bold")
    plt.xlabel("k value", fontsize=14)
    plt.ylabel("Average Accuracy", fontsize=14)
    plt.legend(title="Noise Type")
    plt.grid(True)
    plt.xticks(sorted(set(merged for noise in noise_types for merged in combined_df["k"].unique())), rotation=45)

    # Save the figure to the CSV folder with 1200 DPI
    output_filename = os.path.join(csv_folder, "LineChart_Accuracy_vs_k_All_NoiseTypes.png")
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

    print(f"Line chart saved at: {output_filename}")


if __name__ == "__main__":
    generate_line_chart_accuracy_vs_k()
