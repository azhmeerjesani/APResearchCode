import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_line_charts_noiseless_vs_noise():
    r"""
    This script generates high-resolution (1200 DPI) line charts comparing the control
    (Noiseless) versus each other noise type. For each comparison, it plots the average
    accuracy (with ± standard deviation bands) versus k.

    The CSV files are assumed to reside in:
    r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\CSV Outputs"

    Two lines (with error bands) are plotted on each chart:
      - One for the control ("Noiseless")
      - One for the specified noise type

    Each line is assigned a distinct color and a legend is included.
    """

    # Absolute folder path containing all CSV outputs (raw string to prevent unicode escape issues)
    csv_folder = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\CSV Outputs"

    # Define control noise type and other noise types to compare against
    control_noise = "Noiseless"
    noise_types = [
        "Noisy",
        "TwoQubit",
        "ZRotation",
        "TwoQubitXRotation",
        "T1Relaxation",
        "T2Dephasing",
        "MeasurementError",
        "Combined"
    ]

    # Load the control (Noiseless) data
    control_pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{control_noise}_*.csv")
    control_files = glob.glob(control_pattern)
    if not control_files:
        print(f"No CSV files found for control noise type: {control_noise}")
        return
    control_dfs = [pd.read_csv(file) for file in control_files]
    control_df = pd.concat(control_dfs, ignore_index=True)
    control_df["k"] = control_df["k"].astype(int)
    control_df = control_df.sort_values(by="k")

    # Group control data by k and compute stats
    control_stats = control_df.groupby("k")["Accuracy"].agg(["mean", "std"]).reset_index()
    k_control = control_stats["k"].tolist()
    control_mean = control_stats["mean"].tolist()
    control_std = control_stats["std"].tolist()

    # Set a consistent Seaborn style for all plots
    sns.set(style="whitegrid")

    # Define fixed colors for control and comparison lines (using Tab10 palette)
    palette = sns.color_palette("tab10", n_colors=10)
    control_color = palette[0]  # e.g., blue for control

    # Loop through each non-control noise type
    for noise in noise_types:
        pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{noise}_*.csv")
        test_files = glob.glob(pattern)
        if not test_files:
            print(f"No CSV files found for noise type: {noise}")
            continue
        test_dfs = [pd.read_csv(file) for file in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True)
        test_df["k"] = test_df["k"].astype(int)
        test_df = test_df.sort_values(by="k")

        # Group test data by k and compute stats
        test_stats = test_df.groupby("k")["Accuracy"].agg(["mean", "std"]).reset_index()
        k_test = test_stats["k"].tolist()
        test_mean = test_stats["mean"].tolist()
        test_std = test_stats["std"].tolist()

        # Create a figure for the comparison
        plt.figure(figsize=(12, 8), dpi=1200)

        # Plot the control line (Noiseless)
        plt.plot(k_control, control_mean, marker="o", label=f"{control_noise}",
                 color=control_color, linewidth=2)
        plt.fill_between(k_control,
                         [m - s for m, s in zip(control_mean, control_std)],
                         [m + s for m, s in zip(control_mean, control_std)],
                         color=control_color, alpha=0.2)

        # Choose a different color for the noise type (second color from palette)
        noise_color = palette[1]
        plt.plot(k_test, test_mean, marker="o", label=f"{noise}",
                 color=noise_color, linewidth=2)
        plt.fill_between(k_test,
                         [m - s for m, s in zip(test_mean, test_std)],
                         [m + s for m, s in zip(test_mean, test_std)],
                         color=noise_color, alpha=0.2)

        # Title and labels
        plt.title(f"Comparison: {control_noise} vs. {noise} - Accuracy vs. k", fontsize=16, fontweight="bold")
        plt.xlabel("k value", fontsize=14)
        plt.ylabel("Average Accuracy", fontsize=14)
        # Set xticks to union of available k-values from control and test data
        combined_k = sorted(set(control_df["k"].unique()).union(set(test_df["k"].unique())))
        plt.xticks(combined_k, rotation=45)
        plt.legend(title="Noise Type", fontsize=12)
        plt.grid(True)

        # Build the output filename (replace spaces with underscores)
        output_filename = os.path.join(csv_folder,
                                       f"LineChart_{control_noise}_vs_{noise.replace(' ', '_')}_Accuracy_vs_k.png")
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()
        print(f"Line chart (control vs. {noise}) saved at: {output_filename}")


if __name__ == "__main__":
    generate_line_charts_noiseless_vs_noise()
