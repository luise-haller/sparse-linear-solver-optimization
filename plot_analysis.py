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
        matrix_name = f"bcsstk0{i}"
        grouped = df.groupby("tolerance")[["wall_time_s", "relative_residual"]].mean().reset_index()
        data[matrix_name] = grouped
        print(f"Loaded {matrix_name}: {len(grouped)} tolerance levels")

# Comparison plots with standard tolerance values
tols = np.logspace(-10, -2, 5)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Runtime vs Tolerance for all
ax = axes[0, 0]
for matrix_name, df in data.items():
    ax.loglog(df["tolerance"], df["wall_time_s"], marker="o", label=matrix_name)
ax.invert_xaxis()
ax.set_xlabel("Tolerance")
ax.set_ylabel("Mean Wall Time (s)")
ax.set_title("CG Runtime vs. Tolerance")
ax.legend()
ax.grid(True, which="both")

# Plot 2: Residual vs Tolerance for all
ax = axes[0, 1]
for matrix_name, df in data.items():
    ax.loglog(df["tolerance"], df["relative_residual"], marker="o", label=matrix_name)
ax.invert_xaxis()
ax.set_xlabel("Tolerance")
ax.set_ylabel("Mean Relative Residual")
ax.set_title("Relative Residual vs. Tolerance")
ax.legend()
ax.grid(True, which="both")


sizes = {name: idx for idx, name in enumerate(sorted(data.keys()), start=1)}  # size index for all loaded matrices

# Plot 3: Runtime scaling with matrix size
ax = axes[1, 0]
mean_times = []
for matrix_name in sorted(data.keys()):
    df = data[matrix_name]
    tight_tol_time = (
        df[df["tolerance"] == 1e-10]["wall_time_s"].iloc[0]
        if 1e-10 in df["tolerance"].values
        else df["wall_time_s"].iloc[-1]
    )
    mean_times.append(tight_tol_time)
ax.plot(list(sizes.values()), mean_times, marker="o")
ax.set_xlabel("Matrix Index (ordered bcsstk)")
ax.set_ylabel("Mean Wall Time (s, tightest tol)")
ax.set_title("Runtime Scaling with Matrix Size")
ax.grid(True)

# Plot 4: Iteration count proxy (using 1e-6 tolerance wall time)
ax = axes[1, 1]
times_1e6 = []
for matrix_name in sorted(data.keys()):
    df = data[matrix_name]
    if 1e-6 in df["tolerance"].values:
        times_1e6.append(df[df["tolerance"] == 1e-6]["wall_time_s"].iloc[0])
    else:
        times_1e6.append(np.nan)
ax.plot(list(sizes.values()), times_1e6, marker="o")
ax.set_xlabel("Matrix Index (ordered bcsstk)")
ax.set_ylabel("Mean Wall Time (s, tol=1e-6)")
ax.set_title("Fixed Tolerance Performance")
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "cg_analysis_all_matrices.png"),
            dpi=200, bbox_inches="tight")
print(f"Saved combined analysis plot to {PLOT_DIR}/cg_analysis_all_matrices.png")

# All individual plots for each matrix
for matrix_name, df in data.items():
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.loglog(df["tolerance"], df["wall_time_s"], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Tolerance")
    plt.ylabel("Mean Wall Time (s)")
    plt.title(f"{matrix_name} Runtime vs. Tolerance")
    plt.grid(True, which="both")

    plt.subplot(1, 2, 2)
    plt.loglog(df["tolerance"], df["relative_residual"], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Tolerance")
    plt.ylabel("Mean Relative Residual")
    plt.title(f"{matrix_name} Residual vs. Tolerance")
    plt.grid(True, which="both")

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f"analysis_{matrix_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")

print("Saved all plots into plots/ subfolder.")