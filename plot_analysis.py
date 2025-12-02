import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
grouped = df.groupby("tolerance")[["wall_time_s", "relative_residual"]].mean().reset_index()

tols = grouped["tolerance"].values
times = grouped["wall_time_s"].values
rels = grouped["relative_residual"].values

plt.figure()
plt.loglog(tols, times, marker="o")
plt.gca().invert_xaxis()
plt.xlabel("Tolerance")
plt.ylabel("Mean Wall Sime (s)")
plt.title("CG Runtime vs. Tolerance")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("runtime_vs_tolerance.png", dpi=200)

plt.figure()
plt.loglog(tols, rels, marker="o")
plt.gca().invert_xaxis()
plt.xlabel("Tolerance")
plt.ylabel("Mean Relative Residual")
plt.title("Relative Residual vs. Tolerance")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("residual_vs_tolerance.png", dpi=200)

print("Saved runtime_vs_tolerance.png and residual_vs_tolerance.png")