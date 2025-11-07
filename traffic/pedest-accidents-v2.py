import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -------------------------
# Input data (monthly totals, 10 years)
# -------------------------
april = np.array([100, 83, 97, 71, 95, 63, 72, 65, 67, 70])
sept  = np.array([191, 178, 176, 135, 157, 135, 147, 130, 143, 139])

# -------------------------
# Rates: daily λ = total / (10 * 30)
# -------------------------
A_total, S_total = april.sum(), sept.sum()
lamA_day = A_total / (10 * 30)
lamS_day = S_total / (10 * 30)

# Expected monthly means (for reference)   
lamA_month = lamA_day * 30
lamS_month = lamS_day * 30

# -------------------------
# Poisson-process simulator (continuous time)
# -------------------------
def simulate_month_poisson(lam_day, horizon=30.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    t, arrivals, inter = 0.0, [], []
    if lam_day <= 0:
        return np.array([]), np.array([])
    inv = 1.0 / lam_day
    while True:
        w = rng.exponential(inv)
        t += w
        if t > horizon:
            break
        arrivals.append(t)
        inter.append(w)
    return np.array(arrivals), np.array(inter)

# -------------------------
# Simulations
# -------------------------
rng = np.random.default_rng(2025)
n_sims = 4000
A_counts = np.empty(n_sims, dtype=int)
S_counts = np.empty(n_sims, dtype=int)
A_ia_list, S_ia_list = [], []

for i in range(n_sims):
    aA, iaA = simulate_month_poisson(lamA_day, rng=rng)
    aS, iaS = simulate_month_poisson(lamS_day, rng=rng)
    A_counts[i] = aA.size
    S_counts[i] = aS.size
    if iaA.size: A_ia_list.append(iaA)
    if iaS.size: S_ia_list.append(iaS)

A_ia = np.concatenate(A_ia_list) if A_ia_list else np.array([])
S_ia = np.concatenate(S_ia_list) if S_ia_list else np.array([])

# -------------------------
# 95% CIs for monthly totals (simulated mean)
# -------------------------
def ci95_mean(x):
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    z = 1.959963984540054
    return m, (m - z*se, m + z*se)

mA, ciA = ci95_mean(A_counts)
mS, ciS = ci95_mean(S_counts)

print(f"April:  λ_day={lamA_day:.3f}, E[month]={lamA_month:.2f}, sim mean={mA:.2f}, 95% CI=({ciA[0]:.2f}, {ciA[1]:.2f})")
print(f"Sept :  λ_day={lamS_day:.3f}, E[month]={lamS_month:.2f}, sim mean={mS:.2f}, 95% CI=({ciS[0]:.2f}, {ciS[1]:.2f})")

# -------------------------
# Plots (saved as SVG, not shown)
# -------------------------
os.makedirs("traffic", exist_ok=True)

# 2) Interarrival histograms + exponential overlay
def hist_with_exp(data, lam_day, title, ax, bins=60):
    if data.size == 0:
        ax.set_title(title + " (no data)")
        return

    # 95th percentile cutoff for visualization
    p95 = np.percentile(data, 95)
    x = np.linspace(0, p95, 500)

    # --- Histogram in % of total intervals ---
    weights = np.ones_like(data) / len(data) * 100
    counts, bin_edges, patches = ax.hist(
        data, bins=bins, range=(0, p95),
        weights=weights, alpha=0.6, color="steelblue", edgecolor="gray"
    )

    # --- Scale exponential PDF to % scale ---
    bin_width = bin_edges[1] - bin_edges[0]
    scale_factor = 100 * bin_width  # convert from probability to %
    ax.plot(x, lam_day * np.exp(-lam_day * x) * scale_factor,
            "k--", lw=1.6, label=f"Exp(λ={lam_day:.3f}/day)")

    # --- Labels, grid, and aesthetics ---
    ax.set_title(title)
    ax.set_xlabel("Days between accidents")
    ax.set_ylabel("Frequency (%)")

    # Make grid more detailed and readable
    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.minorticks_on()
    ax.grid(which="minor", axis="x", linestyle=":", alpha=0.3)

    ax.legend()


# -------------------------
# Overlay comparison: April vs September interarrival times
# -------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Define shared range: up to 2 days (not just 95%)
x_max = 2.0
bins = 60
x = np.linspace(0, x_max, 500)
bin_edges = np.linspace(0, x_max, bins + 1)
bin_width = bin_edges[1] - bin_edges[0]
scale_factor = 100 * bin_width  # convert from density to %

# --- September first (behind) ---
ax.hist(
    S_ia, bins=bin_edges, range=(0, x_max),
    weights=np.ones_like(S_ia) / len(S_ia) * 100,
    alpha=0.35, color="tomato", edgecolor="gray", label="September"
)
ax.plot(x, lamS_day * np.exp(-lamS_day * x) * scale_factor,
        color="red", ls="--", lw=1.6, label=f"Exp(λ={lamS_day:.3f}/day)")

# --- April on top ---
ax.hist(
    A_ia, bins=bin_edges, range=(0, x_max),
    weights=np.ones_like(A_ia) / len(A_ia) * 100,
    alpha=0.45, color="steelblue", edgecolor="gray", label="April"
)
ax.plot(x, lamA_day * np.exp(-lamA_day * x) * scale_factor,
        color="navy", ls="--", lw=1.6, label=f"Exp(λ={lamA_day:.3f}/day)")

# --- 95th percentile cutoff line ---
p95_A = np.percentile(A_ia, 95)
p95_S = np.percentile(S_ia, 95)

ax.axvline(p95_S, color="red", ls=":", lw=1.3, alpha=0.8, label=f"Sept 95% = {p95_S:.2f} days")
ax.axvline(p95_A, color="navy", ls=":", lw=1.3, alpha=0.8, label=f"April 95% = {p95_A:.2f} days")


# --- Labels, grid, limits ---
ax.set_xlim(0, x_max)
ax.set_xlabel("Days between accidents")
ax.set_ylabel("Frequency (%)")
ax.set_title("Interarrival Times — April vs September (Overlay with 95% cutoff)")

# Detailed grid
ax.grid(True, which="major", linestyle="--", alpha=0.5)
ax.minorticks_on()
ax.grid(which="minor", linestyle=":", alpha=0.3)

ax.legend()
plt.tight_layout()
plt.savefig("traffic/interarrival_overlay_ci.svg")
plt.close()
# =========================
# Figure 2: Verification — simulated monthly totals (4,000 runs)
# =========================
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

fig, ax = plt.subplots(figsize=(8, 6))

data = [A_counts, S_counts]
labels = ["April", "September"]
colors = ["skyblue", "salmon"]

bp = ax.boxplot(
    data,
    patch_artist=True,
    labels=labels,
    widths=0.5,
    showmeans=True,
    meanline=True,
    notch=True
)

# --- Color styling ---
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for median in bp['medians']:
    median.set(color="black", linewidth=1.2)
for mean in bp['means']:
    mean.set(color="darkred", linewidth=1.5)

# -------------------------
# Expected vs simulated means + 95% CI
# -------------------------
# April
ax.hlines(lamA_month, 0.7, 1.3, colors="blue", ls="--", lw=1.4,
          label=f"April expected mean = {lamA_month:.2f}")
ax.hlines(mA, 0.7, 1.3, colors="navy", ls="-", lw=1.6,
          label=f"April simulated mean = {mA:.2f}, 95% CI = ({ciA[0]:.2f}, {ciA[1]:.2f})")
ax.vlines(1.0, ciA[0], ciA[1], color="navy", lw=3, alpha=0.6)

# September
ax.hlines(lamS_month, 1.7, 2.3, colors="red", ls="--", lw=1.4,
          label=f"Sept expected mean = {lamS_month:.2f}")
ax.hlines(mS, 1.7, 2.3, colors="darkred", ls="-", lw=1.6,
          label=f"Sept simulated mean = {mS:.2f}, 95% CI = ({ciS[0]:.2f}, {ciS[1]:.2f})")
ax.vlines(2.0, ciS[0], ciS[1], color="darkred", lw=3, alpha=0.6)

# -------------------------
# Axes detail & grid improvements
# -------------------------
ax.set_ylabel("Simulated monthly accidents (30 days)")
ax.set_title("Distribution of 4,000 simulated monthly accident totals\n(Poisson process verification)")

# Major gridlines
ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.5)
ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.5)

# Add minor ticks and lighter minor gridlines
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(MultipleLocator(2))  # adjust step for your data scale
ax.grid(which="minor", axis="y", linestyle=":", alpha=0.3)
ax.grid(which="minor", axis="x", linestyle=":", alpha=0.3)

# Tighter tick spacing and formatting
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

ax.legend(loc="upper left", fontsize=9, frameon=True)

plt.tight_layout()
plt.savefig("traffic/verification_boxplot.svg")
plt.close()
