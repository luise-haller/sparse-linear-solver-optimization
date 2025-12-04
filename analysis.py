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
    if os.path.exists(file):
        df = pd.read_csv(file)
        all_dfs[f'bcsstk0{i}'] = df
        print(f"\n---BCSSTK0{i} ({file})---")
        print(f"Shape: {df.shape}")
        print("First few rows:")
        print(df.head())

        # Summary stats by tolerance
        grouped = df.groupby("tolerance")[["wall_time_s", "residual_norm", "relative_residual"]]
        summary = grouped.agg(["mean", "std", "min", "max"])
        print("\nSummary by tolerance:")
        print(summary.round(6))
    else:
        print(f"\nWarning: {file} not found.")

# Big, overall summary of all matrices
if all_dfs:
    combined_df = pd.concat(all_dfs.values(), keys=all_dfs.keys(), names=['matrix'])
    print("\n=== COMBINED SUMMARY ACROSS ALL MATRICES ===")
    print(combined_df.groupby(['matrix', 'tolerance'])["wall_time_s"].agg(['mean', 'std']).round(6))