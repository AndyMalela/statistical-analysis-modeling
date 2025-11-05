import pandas as pd
from pathlib import Path
import numpy as np

# --- Load data ---
csv_path = Path("hiker_randomwalk_runs_seed42_n4000.csv")
df = pd.read_csv(csv_path)

def exit_stats(sub):
    cov = sub[["delta_x", "delta_y"]].cov()
    corr = sub[["delta_x", "delta_y"]].corr().iloc[0, 1]
    mean_x, mean_y = sub["delta_x"].mean(), sub["delta_y"].mean()
    std_x, std_y = sub["delta_x"].std(), sub["delta_y"].std()
    mag = np.sqrt(sub["delta_x"]**2 + sub["delta_y"]**2)
    return pd.Series({
        "mean_delta_x": mean_x,
        "mean_delta_y": mean_y,
        "std_delta_x": std_x,
        "std_delta_y": std_y,
        "cov_x_y": cov.iloc[0, 1],
        "corr_x_y": corr,
        "mean_distance": mag.mean(),
        "std_distance": mag.std()
    })

# Apply per hiker
stats = df.groupby("hiker").apply(exit_stats)
print("Exit distribution statistics for each hiker:\n")
print(stats.round(3))