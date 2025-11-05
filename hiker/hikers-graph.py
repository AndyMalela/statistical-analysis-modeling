import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
CSV_PATH = "hiker_randomwalk_runs_seed42_n4000.csv"
OUT_PNG  = "hiker-running-average-steps.svg"

# --- Load data (expected columns: run, hiker, delta_x, delta_y, steps) ---
df = pd.read_csv(Path(CSV_PATH))

# --- Sort and compute cumulative (expanding) mean of 'steps' per hiker ---
df_sorted = df.sort_values(["hiker", "run"]).copy()
df_sorted["cum_mean_steps"] = (
    df_sorted.groupby("hiker", group_keys=False)["steps"]
    .expanding(min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# --- Plot single figure with all hikers ---
plt.figure(figsize=(10, 6))
legend_labels = []

for hiker_id, g in df_sorted.groupby("hiker"):
    x = range(1, len(g) + 1)               # 1..4000 for each hiker
    plt.plot(x, g["cum_mean_steps"], linewidth=2)
    final_avg = g["cum_mean_steps"].iloc[-1]
    legend_labels.append(f"Hiker {int(hiker_id)} (final avg over 4000: {final_avg:.2f})")

plt.title("Cumulative Running Average of Steps per Hiker (1..4000)")
plt.xlabel("Run index (per hiker)")
plt.ylabel("Cumulative mean of steps")
plt.grid(True, alpha=0.3)
plt.legend(legend_labels, loc="best")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {OUT_PNG}")
