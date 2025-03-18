import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_difference_plots_1200_dpi():
    r"""
    This script generates high-resolution (1200 DPI) difference plots comparing the control
    (Noiseless) mean accuracy versus each other noise type's mean accuracy, as a function of k.
    For each noise type, the absolute difference between the "Noiseless" (control) and the noise
    type's mean accuracy is computed for each k and plotted as a single line.

    The CSV files are assumed to reside in:
    r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\CSV Outputs"
    """
    # Absolute folder path containing all CSV outputs
    csv_folder = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\CSV Outputs"

    # Define control noise type and noise types to compare
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

    # Load control (Noiseless) data
    control_pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{control_noise}_*.csv")
    control_files = glob.glob(control_pattern)
    if not control_files:
        print(f"No CSV files found for control noise type: {control_noise}")
        return
    control_dfs = [pd.read_csv(file) for file in control_files]
    control_df = pd.concat(control_dfs, ignore_index=True)
    control_df["k"] = control_df["k"].astype(int)
    control_df = control_df.sort_values(by="k")
    control_stats = control_df.groupby("k")["Accuracy"].agg(["mean"]).reset_index()
    k_control = control_stats["k"].tolist()
    control_mean = control_stats["mean"].tolist()

    # Set Seaborn style and define palette for difference plots (one color per noise type)
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(noise_types))

    # Loop over each noise type and create a difference plot comparing to control
    for i, noise in enumerate(noise_types):
        pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{noise}_*.csv")
        test_files = glob.glob(pattern)
        if not test_files:
            print(f"No CSV files found for noise type: {noise}")
            continue
        test_dfs = [pd.read_csv(file) for file in test_files]
        test_df = pd.concat(test_dfs, ignore_index=True)
        test_df["k"] = test_df["k"].astype(int)
        test_df = test_df.sort_values(by="k")
        test_stats = test_df.groupby("k")["Accuracy"].agg(["mean"]).reset_index()
        k_test = test_stats["k"].tolist()
        test_mean = test_stats["mean"].tolist()

        # Compute absolute difference between control and test mean accuracies
        # We assume both control and test have same k-values; if not, take the union and match appropriately.
        # Here, we'll assume k values are identical.
        abs_diff = [abs(c - t) for c, t in zip(control_mean, test_mean)]

        # Create a new figure for this difference plot
        plt.figure(figsize=(12, 8), dpi=1200)

        # Plot the difference line (with markers) using a distinct color
        plt.plot(k_control, abs_diff, marker="o", color=palette[i], linewidth=2, label=f"{control_noise} vs. {noise}")

        # Set title and labels
        plt.title(f"Absolute Difference in Mean Accuracy: {control_noise} vs. {noise}", fontsize=16, fontweight="bold")
        plt.xlabel("k value", fontsize=14)
        plt.ylabel("Absolute Difference in Accuracy", fontsize=14)
        plt.xticks(sorted(set(k_control)), rotation=45)
        plt.legend(title="Comparison", fontsize=12)
        plt.grid(True)

        # Build output filename (replace spaces with underscores)
        output_filename = os.path.join(csv_folder,
                                       f"DifferencePlot_{control_noise}_vs_{noise.replace(' ', '_')}_Accuracy_vs_k.png")
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()
        print(f"Difference plot (control vs. {noise}) saved at: {output_filename}")


if __name__ == "__main__":
    generate_difference_plots_1200_dpi()
