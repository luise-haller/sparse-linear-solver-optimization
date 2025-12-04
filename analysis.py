import pandas as pd

df = pd.read_csv("results03.csv")

print("First few rows:")
print(df.head())

# Summary stats by tolerance
grouped = df.groupby("tolerance")[["wall_time_s", "residual_norm", "relative_residual"]]
summary = grouped.agg(["mean", "std", "min", "max"])
print("\nSummary by tolerance parameter:")
print(summary)
