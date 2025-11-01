import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')  # force non-interactive backend

# Parameters
lam = 2.0          # Poisson rate λ
h = 0.0001         # step size, keep λh ≪ 1
S = 101            # sample size per dataset
M = 2000           # number of datasets

# ---------- Discrete-time Poisson process generator ----------
def interarrival_time_discrete(lam, h):
    """Return one exponential waiting time via Bernoulli steps."""
    t = 0.0
    while True:
        U = np.random.rand()
        t += h
        if U < lam * h:     # an event occurs
            return t

# ---------- Generate and fit ----------
shape_hats, scale_hats = [], []

for i in range(1, M + 1):
    # Display progress
    if i % 100 == 0 or i == 1:
        print(f"Generating dataset {i}/{M}...")

    # S interarrival times (i.i.d. exponential)
    sample = [interarrival_time_discrete(lam, h) for _ in range(S)]

    # Fit gamma (Exponential = Gamma(k=1))
    a, loc, scale = gamma.fit(sample, floc=0)
    shape_hats.append(a)
    scale_hats.append(scale)

print("All datasets generated and fitted.")

# ---------- Plot ----------
plt.figure(figsize=(12, 5))

# --- Shape (k) estimates ---
plt.subplot(1, 2, 1)
plt.hist(shape_hats, bins=40, color='skyblue', edgecolor='black')
plt.axvline(1, color='red', ls='--', label='True k=1')
mean_shape = np.mean(shape_hats)
plt.axvline(mean_shape, color='orange', ls='-', label=f"Mean k̂={mean_shape:.3f}")
plt.title(f"Case: Shape Estimates\nλ={lam}, S={S}, Samples={M}")
plt.xlabel("Shape (k̂)")
plt.ylabel("Frequency")
plt.legend()

# --- Scale (θ) estimates ---
plt.subplot(1, 2, 2)
plt.hist(scale_hats, bins=40, color='lightgreen', edgecolor='black')
plt.axvline(1 / lam, color='red', ls='--', label=f"True θ=1/λ={1/lam:.3f}")
mean_scale = np.mean(scale_hats)
plt.axvline(mean_scale, color='orange', ls='-', label=f"Mean θ̂={mean_scale:.3f}")
plt.title(f"Case: Scale Estimates\nλ={lam}, S={S}, Samples={M}")
plt.xlabel("Scale (θ̂)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig("week3-2-gamma1-1.svg", format="svg", bbox_inches="tight")
plt.close()