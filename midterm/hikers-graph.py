import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========= Config =========
CSV_PATH = "hiker_randomwalk_runs_seed42_n4000.csv"   # update if needed
RADIUS_METERS = 1000
OUT_DIR = Path("hiker")   # all svgs saved here
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ==========================

print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} records")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

def save_svg(fig, short_title: str):
    """Save a figure as SVG with standardized file naming."""
    fname = OUT_DIR / f"hiker-{short_title}.svg"
    # SVG is vector; dpi is irrelevant for lines, but we include tight bbox for better layout.
    fig.savefig(fname, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fname}")

def ecdf(vals):
    v = np.sort(np.asarray(vals))
    y = np.arange(1, v.size + 1) / v.size
    return v, y

def running_mean_ci(x, alpha=0.05):
    """Running mean + 95% CI using Welford's algorithm + normal approx."""
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1)
    mean = np.empty_like(x)
    var = np.empty_like(x)

    m, s = 0.0, 0.0
    for i, xi in enumerate(x, start=1):
        delta = xi - m
        m += delta / i
        s += delta * (xi - m)
        mean[i-1] = m
        var[i-1] = s / (i - 1) if i > 1 else 0.0

    z = 1.96
    se = np.sqrt(var / n)
    return n, mean, mean - z * se, mean + z * se


# --- 1) Hexbin (ΔX, ΔY) per hiker ---
for k in [1, 2, 3]:
    sub = df[df["hiker"] == k]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    hb = ax.hexbin(sub["delta_x"], sub["delta_y"], gridsize=55, mincnt=1)
    cb = fig.colorbar(hb, ax=ax); cb.set_label("Count")

    theta = np.linspace(0, 2*np.pi, 500)
    ax.plot(RADIUS_METERS*np.cos(theta), RADIUS_METERS*np.sin(theta),
            linestyle='--', color='gray', linewidth=0.8)

    ax.set_aspect('equal')
    ax.set_xlabel("ΔX (m)")
    ax.set_ylabel("ΔY (m)")
    ax.set_title(f"Exit Locations (ΔX, ΔY) — Hiker {k}")

    save_svg(fig, f"hexbin-hiker{k}")


# --- 2) Rose (polar) histogram of exit angles (overlaid) ---
fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(111, projection="polar")
bins = 32
for k, color in zip([1, 2, 3], colors):
    sub = df[df["hiker"] == k]
    angles = np.arctan2(sub["delta_y"], sub["delta_x"])
    angles = (angles + 2*np.pi) % (2*np.pi)
    counts, edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(edges)
    ax.bar(edges[:-1], counts, width=widths, bottom=0,
           alpha=0.5, edgecolor='none', color=color, label=f"Hiker {k}")
ax.set_title("Exit Angle Distribution (Rose Histogram)")
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.12))
save_svg(fig, "rose-angles")


# --- 3) ECDF of steps (N) ---
fig, ax = plt.subplots(figsize=(7, 5))
for k, color in zip([1, 2, 3], colors):
    xv, yv = ecdf(df[df["hiker"] == k]["steps"].to_numpy())
    ax.plot(xv, yv, label=f"Hiker {k}", color=color)
ax.set_xlabel("Steps to 1000 m (N)")
ax.set_ylabel("ECDF")
ax.set_title("ECDF of Steps to Reach 1000 m")
ax.grid(True, linestyle=":", linewidth=0.7)
ax.legend()
save_svg(fig, "ecdf-steps")


# --- 4) Boxplot of N by hiker ---
fig, ax = plt.subplots(figsize=(6.5, 5.5))
data = [df[df["hiker"] == k]["steps"] for k in [1, 2, 3]]
ax.boxplot(data, labels=["Hiker 1", "Hiker 2", "Hiker 3"],
           showmeans=True, meanline=True)
ax.set_ylabel("Steps to 1000 m (N)")
ax.set_title("Distribution of N by Hiker")
ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
save_svg(fig, "boxplot-steps")


# --- 5) Running mean + 95% CI (per hiker), with CI legend entry ---
for k, color in zip([1, 2, 3], colors):
    vals = df[df["hiker"] == k]["steps"].to_numpy()
    order = np.random.default_rng(123 + k).permutation(len(vals))
    n, mean, lo, hi = running_mean_ci(vals[order])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(n, mean, color=color, label=f"Hiker {k} mean")
    ax.fill_between(n, lo, hi, alpha=0.25, color=color, label="95% CI")
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Running mean of N")
    ax.set_title(f"Running Mean of N with 95% CI — Hiker {k}")
    ax.grid(True, linestyle=":", linewidth=0.7)
    ax.legend(loc="best", frameon=True)

    save_svg(fig, f"running-mean-hiker{k}")
print("All done.")