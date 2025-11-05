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

# 1) Monthly totals with CI (minimal, no annotations ⇒ no overlap)
fig, ax = plt.subplots(figsize=(8, 6))
labels = ["April", "September"]
means  = [mA, mS]
yerr   = np.array([[mA - ciA[0], mS - ciS[0]],
                   [ciA[1] - mA, ciS[1] - mS]])

ax.bar(labels, means, alpha=0.75)
ax.errorbar(labels, means, yerr=yerr, fmt="none", capsize=6, color="black")
ax.axhline(lamA_month, ls="--", lw=1, color="tab:blue",  label=f"April E[N(30)]={lamA_month:.1f}")
ax.axhline(lamS_month, ls="--", lw=1, color="tab:red",   label=f"Sept E[N(30)]={lamS_month:.1f}")
ax.set_ylabel("Accidents per 30 days")
ax.set_title("Simulated monthly totals (4,000 runs) with 95% CIs")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("traffic/poisson_monthly_totals.svg")
plt.close()

# 2) Interarrival histograms + exponential overlay
def hist_with_exp(data, lam_day, title, ax, bins=60):
    if data.size == 0:
        ax.set_title(title + " (no data)"); return
    p95 = np.percentile(data, 95)
    x = np.linspace(0, p95, 500)
    ax.hist(data, bins=bins, range=(0, p95), density=True, alpha=0.6)
    ax.plot(x, lam_day*np.exp(-lam_day*x), "k--", lw=1.5, label=f"Exp(λ={lam_day:.3f}/day)")
    ax.set_title(title)
    ax.set_xlabel("Days between accidents"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
hist_with_exp(A_ia, lamA_day, "Interarrival times — April", axes[0])
hist_with_exp(S_ia, lamS_day, "Interarrival times — September", axes[1])
plt.tight_layout()
plt.savefig("traffic/interarrival_histograms.svg")
plt.close()

# 3) ECDF comparison
def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, x.size+1)/x.size
    return x, y

if A_ia.size and S_ia.size:
    xA, yA = ecdf(A_ia)
    xS, yS = ecdf(S_ia)
    p95 = max(np.percentile(A_ia, 95), np.percentile(S_ia, 95))
    plt.figure(figsize=(8, 6))
    plt.step(xA, yA, where="post", label="April")
    plt.step(xS, yS, where="post", label="September")
    plt.xlim(0, p95); plt.ylim(0, 1)
    plt.xlabel("Days between accidents"); plt.ylabel("Empirical CDF")
    plt.title("Interarrival time ECDFs — April vs September")
    plt.grid(alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig("traffic/ecdf_interarrival.svg")
    plt.close()
    
# =========================
# EXTRA VISUAL: Candlestick (boxplot) of the 4,000 simulated monthly totals
# =========================
fig, ax = plt.subplots(figsize=(8, 6))

# Combine both months’ samples
data = [A_counts, S_counts]
labels = ["April", "September"]
colors = ["skyblue", "salmon"]

# Boxplot
bp = ax.boxplot(
    data,
    patch_artist=True,
    labels=labels,
    widths=0.5,
    showmeans=True,
    meanline=True,
    notch=True
)

# Color styling
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
for median in bp['medians']:
    median.set(color="black", linewidth=1.2)
for mean in bp['means']:
    mean.set(color="darkred", linewidth=1.5)

# Expected means (from real data)
ax.hlines(lamA_month, 0.7, 1.3, colors="blue", ls="--", lw=1.2, label=f"April expected mean = {lamA_month:.1f}")
ax.hlines(lamS_month, 1.7, 2.3, colors="red",  ls="--", lw=1.2, label=f"September expected mean = {lamS_month:.1f}")

# Cosmetics
ax.set_ylabel("Simulated monthly accidents (30 days)")
ax.set_title("Distribution of 4,000 simulated monthly accident totals\n(Poisson process models for April & September)")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("traffic/simulation_boxplot.svg")
plt.close()
