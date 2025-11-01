import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Setup: create output folder
# =========================
os.makedirs("traffic", exist_ok=True)

# =========================
# Data and fitted rates
# =========================
april_obs = np.array([100, 83, 97, 71, 95, 63, 72, 65, 67, 70])
sept_obs  = np.array([191, 178, 176, 135, 157, 135, 147, 130, 143, 139])

λA_month, λS_month = april_obs.mean(), sept_obs.mean()
λA_day, λS_day = λA_month / 30, λS_month / 30
print(f"April λ_month={λA_month:.2f}, λ_day={λA_day:.3f}")
print(f"Sept  λ_month={λS_month:.2f}, λ_day={λS_day:.3f}")

# =========================
# Poisson-process simulator
# =========================
def simulate_month(rate_per_day, horizon_days=30, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    arrivals, interarrivals = [], []
    t = 0.0
    while True:
        w = rng.exponential(1.0 / rate_per_day)
        t += w
        if t > horizon_days:
            break
        arrivals.append(t)
        interarrivals.append(w)
    return np.array(arrivals), np.array(interarrivals)

# =========================
# Simulate 4000 months
# =========================
rng = np.random.default_rng(2025)
n_sims = 4000
april_counts, sept_counts = [], []
april_ia, sept_ia = [], []

for _ in range(n_sims):
    aA, iaA = simulate_month(λA_day, rng=rng)
    aS, iaS = simulate_month(λS_day, rng=rng)
    april_counts.append(len(aA))
    sept_counts.append(len(aS))
    if len(iaA): april_ia.append(iaA)
    if len(iaS): sept_ia.append(iaS)

april_counts, sept_counts = np.array(april_counts), np.array(sept_counts)
april_ia = np.concatenate(april_ia)
sept_ia = np.concatenate(sept_ia)

# =========================
# 95% Confidence Intervals
# =========================
def ci_mean(x):
    m, se = np.mean(x), np.std(x, ddof=1)/np.sqrt(len(x))
    z = 1.96
    return m, (m - z*se, m + z*se)

mA, ciA = ci_mean(april_counts)
mS, ciS = ci_mean(sept_counts)

# =========================
# Plot 1: Monthly totals with 95% CI + expected λ
# =========================
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["April", "September"]
means = [mA, mS]
cis = np.array([[mA - ciA[0], mS - ciS[0]], [ciA[1] - mA, ciS[1] - mS]])

bars = ax.bar(labels, means, color=["skyblue", "salmon"], alpha=0.7)
ax.errorbar(labels, means, yerr=cis, fmt="none", capsize=6, color="black")
ax.axhline(λA_month, color="blue", ls="--", lw=1, label=f"April fitted λ={λA_month:.1f}")
ax.axhline(λS_month, color="red",  ls="--", lw=1, label=f"Sept fitted λ={λS_month:.1f}")

for i, (mean, ci) in enumerate(zip(means, [ciA, ciS])):
    ax.text(i, mean + 2, f"Mean={mean:.1f}\n95% CI={ci[0]:.1f}–{ci[1]:.1f}",
            ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Accidents per 30 days")
ax.set_title("Simulated monthly totals (4,000 runs) with 95% CIs")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("traffic/poisson_monthly_totals.svg", format="svg")
plt.show()

# =========================
# Plot 2: Interarrival histograms + exponential overlays
# =========================
p95 = max(np.percentile(april_ia, 95), np.percentile(sept_ia, 95))
x = np.linspace(0, p95, 500)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, data, λ_day, color, month in [
    (axes[0], april_ia, λA_day, "skyblue", "April"),
    (axes[1], sept_ia,  λS_day, "salmon", "September")
]:
    ax.hist(data, bins=60, range=(0, p95), density=True, alpha=0.6, color=color, label="Simulated")
    ax.plot(x, λ_day * np.exp(-λ_day * x), "k--", lw=1.5, label=f"Exp(λ={λ_day:.3f}/day)")
    mean_ia = np.mean(data)
    ax.axvline(mean_ia, color="black", lw=1, alpha=0.6)
    ax.text(mean_ia + 0.1, 0.9*ax.get_ylim()[1],
            f"mean ≈ {mean_ia:.2f} days", rotation=90, va="top", fontsize=9)
    ax.set_title(f"Interarrival times — {month}\n(λ_day={λ_day:.3f}/day)")
    ax.set_xlabel("Days between accidents")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

fig.suptitle("Interarrival time distributions (empirical vs theoretical exponential)", fontsize=12)
plt.tight_layout()
plt.savefig("traffic/interarrival_histograms.svg", format="svg")
plt.show()

# =========================
# Plot 3: ECDF comparison + medians
# =========================
def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x)+1)/len(x)
    return x, y

xA, yA = ecdf(april_ia)
xS, yS = ecdf(sept_ia)
medA, medS = np.median(april_ia), np.median(sept_ia)

plt.figure(figsize=(7,5))
plt.step(xA, yA, where="post", label=f"April (median={medA:.2f} d)")
plt.step(xS, yS, where="post", label=f"September (median={medS:.2f} d)")
plt.axvline(medA, color="blue", ls="--", alpha=0.5)
plt.axvline(medS, color="red",  ls="--", alpha=0.5)
plt.xlim(0, p95)
plt.ylim(0, 1)
plt.xlabel("Days between accidents")
plt.ylabel("Empirical CDF")
plt.title("ECDF of interarrival times — April vs September")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("traffic/ecdf_interarrival.svg", format="svg")
plt.show()

# =========================
# Discussion summary
# =========================
print("\n--- Interpretation ---")
print(f"• λ_day (April) = {λA_day:.3f}/day → mean interarrival ≈ {1/λA_day:.2f} days")
print(f"• λ_day (September) = {λS_day:.3f}/day → mean interarrival ≈ {1/λS_day:.2f} days")
print("• Monthly totals from 4,000 simulated 30-day Poisson processes closely match fitted λ_month values.")
print("• Interarrival histograms follow exponential shapes; September shows shorter intervals (higher λ).")
print("• ECDF confirms September’s faster accumulation of events — consistent with higher accident frequency.")
print("\nSVG plots saved to folder: 'traffic/'")
