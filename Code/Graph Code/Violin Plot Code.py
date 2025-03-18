import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_violin_plots_1200_dpi():
    """
    This script creates a separate 1200 DPI violin plot for each noise type, showing
    Accuracy vs. k-value, and outputs summary statistics for each plot to a text file.

    It closely mirrors the box-plot code, except we now use sns.violinplot.
    Each violin is color-coded according to k with a dynamic palette that
    adjusts to the number of unique k values.

    The CSV files are assumed to be located in the absolute folder path below.
    """

    # Absolute path to the folder containing all CSV outputs
    csv_folder = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\CSV Outputs"

    # Noise types that appear in the CSV filenames
    noise_types = [
        "SingleQubit",
        "Noisy",
        "TwoQubit",
        "ZRotation",
        "TwoQubitXRotation",
        "T1Relaxation",
        "T2Dephasing",
        "MeasurementError",
        "Combined"
    ]

    # Set Seaborn style for visually appealing plots
    sns.set(style="whitegrid")

    # Loop over each noise type and create a violin plot
    for noise in noise_types:
        # Use a glob pattern to find all CSV files that match this noise type
        if (noise == "SingleQubit"):
            pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_Noisy_*.csv")
        else:
            pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{noise}_*.csv")

        matching_files = glob.glob(pattern)

        # If no files found for a noise type, just skip
        if not matching_files:
            print(f"No CSV files found for noise type: {noise}")
            continue

        # Load and concatenate all runs for this noise type
        df_list = []
        for csv_file in matching_files:
            df_list.append(pd.read_csv(csv_file))
        combined_df = pd.concat(df_list, ignore_index=True)

        # Convert k to integer to ensure proper sorting
        combined_df["k"] = combined_df["k"].astype(int)
        combined_df = combined_df.sort_values(by="k")  # Sort k-values explicitly

        # Create a figure with increased spacing for better readability
        plt.figure(figsize=(12, 6), dpi=1200)

        # Dynamically generate a palette with enough colors for all unique k-values
        unique_ks = sorted(combined_df["k"].unique())
        n_k = len(unique_ks)
        palette = sns.color_palette("viridis", n_colors=n_k)

        # Violin plot: x-axis is 'k', y-axis is 'Accuracy'
        # Each k gets its own color from the custom palette
        ax = sns.violinplot(
            x="k", y="Accuracy",
            data=combined_df,
            palette=palette,
            linewidth=1.5
        )

        # Ensure k-values are in correct order on the x-axis
        ax.set_xticks(unique_ks)
        ax.set_xticklabels(unique_ks, rotation=45)

        # Set title and labels
        plt.title(f"Accuracy vs. k (Violin Plot) for Noise: {noise}", fontsize=14, fontweight="bold")
        plt.xlabel("k value", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)

        # Build the output filename (replace spaces in 'noise' with underscores)
        output_filename = os.path.join(csv_folder, f"ViolinPlot_{noise.replace(' ', '_')}_Accuracy_vs_k.png")

        # Save the plot as a high-resolution PNG
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()

        print(f"Violin plot saved for noise type '{noise}' at: {output_filename}")

        # ============================
        # Generate Summary Statistics
        # ============================
        summary_filename = os.path.join(csv_folder, f"ViolinPlot_{noise.replace(' ', '_')}_Summary_Stats.txt")
        with open(summary_filename, "w") as stats_file:
            stats_file.write(f"Summary Statistics for {noise} Noise Violin Plot\n")
            stats_file.write("="*50 + "\n\n")

            # Group data by k value and compute stats
            for k_value, group in combined_df.groupby("k", sort=True):
                q1 = group["Accuracy"].quantile(0.25)  # 1st quartile
                q2 = group["Accuracy"].median()        # Median (Q2)
                q3 = group["Accuracy"].quantile(0.75)  # 3rd quartile
                mean = group["Accuracy"].mean()        # Mean
                min_value = group["Accuracy"].min()    # Minimum value
                max_value = group["Accuracy"].max()    # Maximum value
                range_value = max_value - min_value    # Range

                # Write to the file
                stats_file.write(f"k = {k_value}\n")
                stats_file.write(f"  - Mean: {mean:.4f}\n")
                stats_file.write(f"  - Median (Q2): {q2:.4f}\n")
                stats_file.write(f"  - Q1 (25%): {q1:.4f}\n")
                stats_file.write(f"  - Q3 (75%): {q3:.4f}\n")
                stats_file.write(f"  - Min: {min_value:.4f}\n")
                stats_file.write(f"  - Max: {max_value:.4f}\n")
                stats_file.write(f"  - Range: {range_value:.4f}\n")
                stats_file.write("-" * 50 + "\n")

        print(f"Summary statistics saved for noise type '{noise}' at: {summary_filename}")


if __name__ == "__main__":
    generate_violin_plots_1200_dpi()
