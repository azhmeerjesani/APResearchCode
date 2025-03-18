import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_box_plots_1200_dpi():
    """
    This script creates a separate 1200 DPI box plot for each noise type, showing
    Accuracy vs. k-value, and outputs summary statistics for each plot to a text file.

    Each box plot will be spaced out for readability, include an appealing color scheme,
    and save summary statistics (mean, median, quartiles, range) to a .txt file.
    """

    # Absolute path to the folder containing all CSV outputs
    csv_folder = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\CSV Outputs"

    # Noise types that appear in the CSV filenames
    noise_types = [
        "Noiseless",
        "Noisy",
        "TwoQubit",
        "ZRotation",
        "TwoQubitXRotation",
        "T1Relaxation",
        "T2Dephasing",
        "MeasurementError",
        "Combined"
    ]

    # Set Seaborn style for more visually appealing plots
    sns.set(style="whitegrid")

    # Loop over each noise type and create a box plot
    for noise in noise_types:
        # Use a glob pattern to find all CSV files that match this noise type
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

        # Box plot with better color handling and explicit hue assignment
        ax = sns.boxplot(
            x="k", y="Accuracy", data=combined_df,
            hue="k",  # Ensures proper differentiation of k-values
            palette=sns.color_palette("viridis", as_cmap=True),  # Continuous colormap
            linewidth=1.5,
            legend=False  # Avoids redundant legend
        )

        # Ensure k-values are in correct order on the x-axis
        ax.set_xticks(sorted(combined_df["k"].unique()))
        ax.set_xticklabels(sorted(combined_df["k"].unique()), rotation=45)

        # Set title and labels
        plt.title(f"Accuracy vs. k for Noise: {noise}", fontsize=14, fontweight="bold")
        plt.xlabel("k value", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)

        # Ensure filenames are valid
        output_filename = os.path.join(csv_folder, f"BoxPlot_{noise.replace(' ', '_')}_Accuracy_vs_k.png")

        # Save the plot as a high-resolution PNG
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()

        print(f"Box plot saved for noise type '{noise}' at: {output_filename}")

        # ============================
        # Generate Summary Statistics
        # ============================
        summary_filename = os.path.join(csv_folder, f"BoxPlot_{noise.replace(' ', '_')}_Summary_Stats.txt")
        with open(summary_filename, "w") as stats_file:
            stats_file.write(f"Summary Statistics for {noise} Noise Box Plot\n")
            stats_file.write("="*50 + "\n\n")

            # Group data by k value and compute stats
            for k_value, group in combined_df.groupby("k", sort=True):  # Ensures sorted k-values
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
    generate_box_plots_1200_dpi()
