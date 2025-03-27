import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_violin_plots_1200_dpi():
    """
    This script creates a separate 1200 DPI violin plot for each noise type, showing
    Accuracy vs. k-value, and outputs summary statistics for each plot to a text file.
    """

    csv_folder = r"C:\Users\azhme\OneDrive - Clear Creek ISD\Files\School\11th Grade\AP Capstone Research\Parts Of Research Paper\APResearchCode\Outputs\CSV Outputs"

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

    sns.set(style="whitegrid")

    for noise in noise_types:
        if noise == "SingleQubit":
            pattern = os.path.join(csv_folder, "Iris_QkNN_Run_*_Noisy_*.csv")
        else:
            pattern = os.path.join(csv_folder, f"Iris_QkNN_Run_*_{noise}_*.csv")

        matching_files = glob.glob(pattern)
        if not matching_files:
            print(f"No CSV files found for noise type: {noise}")
            continue

        # Load and combine all runs for this noise
        df_list = [pd.read_csv(csv_file) for csv_file in matching_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Make sure k is an integer and sorted
        combined_df["k"] = combined_df["k"].astype(int)
        combined_df.sort_values(by="k", inplace=True)

        plt.figure(figsize=(12, 6), dpi=1200)

        # Identify unique k-values in ascending order
        unique_ks = sorted(combined_df["k"].unique())
        n_k = len(unique_ks)
        palette = sns.color_palette("viridis", n_colors=n_k)

        # <-- Fix: tell violinplot the exact order of the k categories
        ax = sns.violinplot(
            x="k",
            y="Accuracy",
            data=combined_df,
            palette=palette,
            linewidth=1.5,
            order=unique_ks        # The key parameter!
        )

        # Optionally set rotation if you want
        plt.xticks(rotation=45)

        plt.title(f"Accuracy vs. k (Violin Plot) for Noise: {noise}", fontsize=14, fontweight="bold")
        plt.xlabel("k value", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)

        out_png = os.path.join(csv_folder, f"ViolinPlot_{noise.replace(' ', '_')}_Accuracy_vs_k.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Violin plot saved for noise type '{noise}' at: {out_png}")

        # (Optional) Write summary statistics code here...
        # ...

if __name__ == "__main__":
    generate_violin_plots_1200_dpi()
