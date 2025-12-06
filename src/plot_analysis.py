import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

RESULT_FILES = [
    "results/results01.csv",
    "results/results02.csv", 
    "results/results03.csv",
    "results/results04.csv",
    "results/results05.csv",
    "results/results06.csv",
    "results/results08.csv",
    "results/results11.csv",
    "results/results14.csv",
    "results/results15.csv",
    "results/results16.csv",
    "results/results18.csv",
]

PLOT_DIR = "plots" # Creating new plot subfolder to save all plots under
os.makedirs(PLOT_DIR, exist_ok=True)

# Loading all data
data = {}
for i, file in enumerate(RESULT_FILES, 1):
    if os.path.exists(file):
        df = pd.read_csv(file)
        
        if "preconditioner" not in df.columns:
            print(f"Skipping {file}: no 'preconditioner' column found.")
            continue
        matrix_name = f"bcsstk0{i}"
        data[matrix_name] = df
        print(f"Loaded {matrix_name} from {file}: {len(df)} rows")

# Plot of mean wall time by preconditioner across matrices
if data:
    all_prec = sorted(set( prec for df in data.values() for prec in df["preconditioner"].unique() ))
    plt.figure(figsize=(10, 6))
    for matrix_idx, (matrix_name, df) in enumerate(sorted(data.items()), start=1):
        grouped = (
            df.groupby("preconditioner")[["wall_time_s", "relative_residual"]]
            .mean()
            .reset_index()
        )
        # aligning the all_prec order
        mean_times = []
        for prec in all_prec:
            row = grouped[grouped["preconditioner"] == prec]
            mean_times.append(row["wall_time_s"].iloc[0] if not row.empty else np.nan)
        x = np.arange(len(all_prec)) + 0.1 * (matrix_idx - 1)
        plt.bar(x, mean_times, width=0.08, label=matrix_name)
    
    plt.xticks(np.arange(len(all_prec)) + 0.1 * (len(data) - 1) / 2, all_prec, rotation=45, ha="right")
    plt.ylabel("Mean Wall Time (s)")
    plt.title("Mean Wall Time by Preconditioner (fixed tolerance)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "preconditioners_time_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")

# Individual plots for each matrix
for matrix_name, df in data.items():
    grouped = (
        df.groupby("preconditioner")[["wall_time_s", "relative_residual"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(grouped["preconditioner"], grouped["wall_time_s"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Wall Time (s)")
    plt.title(f"{matrix_name}: Time by Preconditioner")

    plt.subplot(1, 2, 2)
    plt.bar(grouped["preconditioner"], grouped["relative_residual"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Relative Residual")
    plt.title(f"{matrix_name}: Residual by Preconditioner")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"prec_{matrix_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")

print("Saved all preconditioner comparison plots into plots/.")