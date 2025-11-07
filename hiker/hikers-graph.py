import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Global font settings ---
plt.rc("font", size=14)          # Base font size
plt.rc("axes", titlesize=16)     # Axes title font
plt.rc("axes", labelsize=15)     # Axes labels
plt.rc("legend", fontsize=12)    # Legend text
plt.rc("xtick", labelsize=13)    # X tick labels
plt.rc("ytick", labelsize=13)    # Y tick labels

# --- Paths ---
CSV_PATH = "hiker_randomwalk_runs_seed42_n4000.csv"
OUT_PNG  = "hiker-running-average-steps.svg"

# --- Load data ---
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
fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
legend_labels = []

for hiker_id, g in df_sorted.groupby("hiker"):
    x = range(1, len(g) + 1)
    ax.plot(x, g["cum_mean_steps"], linewidth=2)
    final_avg = g["cum_mean_steps"].iloc[-1]
    legend_labels.append(f"Hiker {int(hiker_id)} (final avg over 4000: {final_avg:.2f})")

ax.set_title("Cumulative Running Average of Steps per Hiker (1..4000)")
ax.set_xlabel("Run index (per hiker)")
ax.set_ylabel("Cumulative mean of steps")
ax.grid(True, alpha=0.3)
ax.set_box_aspect(1)

# Legend outside
ax.legend(legend_labels, loc="best", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

# --- Save high DPI for clarity ---
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")  # increased dpi
plt.show()
