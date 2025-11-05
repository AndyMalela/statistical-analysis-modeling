import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')  # force non-interactive backend

# Parameters
lam = 2.0          # rate λ
h = 0.001          # step size
r = 5              # number of events waited for → shape k=r
S = 101            # sample size per dataset
M = 2000           # number of datasets

# ---------- Discrete-time waiting for r-th event ----------

def time_to_rth_event(lam, r):
    # Sum of r iid exponentials ⇒ Gamma(shape=r, scale=1/lam)
    return np.random.gamma(shape=r, scale=1/lam)

# ---------- Generate and fit ----------
shape_hats, scale_hats = [], []

for i in range(1, M + 1):
    # Progress info
    if i % 100 == 0 or i == 1:
        print(f"Generating dataset {i}/{M} (waiting for r={r} events)...")

    # S samples of Gamma(k=r, θ=1/λ)
    sample = np.random.gamma(shape=r, scale=1/lam, size=S)  # vectorized, faster

    # Fit gamma distribution
    a, loc, scale = gamma.fit(sample, floc=0)
    shape_hats.append(a)
    scale_hats.append(scale)

print("All datasets generated and fitted successfully.")

# ---------- Plot ----------
plt.figure(figsize=(12, 5))

# --- Shape (k) estimates ---
plt.subplot(1, 2, 1)
plt.hist(shape_hats, bins=40, color='skyblue', edgecolor='black')
plt.axvline(r, color='red', ls='--', label=f"True k={r}")
mean_shape = np.mean(shape_hats)
plt.axvline(mean_shape, color='orange', ls='-', label=f"Mean k̂={mean_shape:.3f}")
plt.title(f"Gamma Shape Parameter (Waiting for r={r} Events)\nλ={lam}, S={S}, Samples={M}")
plt.xlabel("Shape (k̂)")
plt.ylabel("Frequency")
plt.legend()

# --- Scale (θ) estimates ---
plt.subplot(1, 2, 2)
plt.hist(scale_hats, bins=40, color='lightgreen', edgecolor='black')
plt.axvline(1 / lam, color='red', ls='--', label=f"True θ=1/λ={1/lam:.3f}")
mean_scale = np.mean(scale_hats)
plt.axvline(mean_scale, color='orange', ls='-', label=f"Mean θ̂={mean_scale:.3f}")
plt.title(f"Gamma Scale Parameter (Waiting for r={r} Events)\nλ={lam}, S={S}, Samples={M}")
plt.xlabel("Scale (θ̂)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig("w3-parameter-estimate-2.svg", format="svg", bbox_inches="tight")
plt.close()