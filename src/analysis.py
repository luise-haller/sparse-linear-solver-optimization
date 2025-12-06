import pandas as pd
import os

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

print("---Analysis for ALL BCSSTK Matrices---\n")

all_dfs = {}

for i, file in enumerate(RESULT_FILES, 1):
    if not os.path.exists(file):
        print(f"Warning: {file} not found, skipping.")
        continue

    df = pd.read_csv(file)
    if "preconditioner" not in df.columns:
        print(f"Warning: {file} has no 'preconditioner' column, skipping.")
        continue

    matrix_name = f"bcsstk0{i}"
    all_dfs[matrix_name] = df

    print(f"\n--- {matrix_name} ({file}) ---")
    print("First few rows:")
    print(df.head())
    

    grouped = df.groupby("preconditioner")[["wall_time_s", "residual_norm", "relative_residual"]]
    summary = grouped.agg(["mean", "std", "min", "max"])
    print("\nSummary by preconditioner:")
    print(summary.round(6))


if all_dfs:
    combined = pd.concat(all_dfs, names=["matrix"])
    print("\n=== Combined mean wall time by matrix and preconditioner ===")
    combo_summary = (
        combined.groupby(["matrix", "preconditioner"])["wall_time_s"]
        .agg(["mean", "std", "min", "max"])
        .round(6)
    )
    print(combo_summary)
else:
    print("\nNo valid result files loaded.")