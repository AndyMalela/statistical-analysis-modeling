import numpy as np
import matplotlib.pyplot as plt

#if 99.7% of points of a standard Wiener process W(t) lie within the cone defined by y=3√t and y=-3√t for t in [0,T], it follows a normal distribution with mean 0 and variance t at each time t.

# ----- parameters -----
np.random.seed(42)
n_paths = 2000
T = 10.0
n_steps = 1000
dt = T / n_steps

# ----- simulate standard Wiener process (Brownian motion) -----
# W_0 = 0, W_{t+dt} - W_t ~ Normal(0, dt)
incr = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
W = np.zeros((n_paths, n_steps + 1))
W[:, 1:] = np.cumsum(incr, axis=1)

t = np.linspace(0.0, T, n_steps + 1)
cone = 3.0 * np.sqrt(t)
upper, lower = cone, -cone

# ----- fraction of all simulated points inside the cone -----
inside = (W <= upper) & (W >= lower)         # shape: (n_paths, n_steps+1)
percent_all_points = inside.mean() * 100.0

# Per-time fraction (useful to see it stays ~99.7% across t>0)
per_t_percent = inside.mean(axis=0) * 100.0
avg_per_t_percent_excl0 = per_t_percent[1:].mean()   # exclude t=0 to avoid 0/0 quirks

# Optional: standardize and check |W_t|/sqrt(t) <= 3 directly (exclude t=0)
Z = W[:, 1:] / np.sqrt(t[1:])
percent_3sigma_check = (np.abs(Z) <= 3.0).mean() * 100.0

print(f"Percentage of ALL simulated points inside the cone: {percent_all_points:.3f}%")
print(f"Average per-time percentage (t>0): {avg_per_t_percent_excl0:.3f}%")
print(f'3-sigma check via Z = W_t/√t (t>0): {percent_3sigma_check:.3f}% (theory ≈ 99.7%)')

# ----- visualization (thicker paths) -----
plt.figure(figsize=(10, 6))

n_show = 200  # still plot a subset for clarity
plt.plot(t, W[:n_show, :].T, color='tab:blue', alpha=0.15, lw=1.5)  # thicker + slightly more opaque

# boundary lines
plt.plot(t, upper, 'r--', lw=2.5, label=r'$y_1(t)=3\sqrt{t}$')
plt.plot(t, lower, 'r--', lw=2.5, label=r'$y_2(t)=-3\sqrt{t}$')

plt.title(f'{n_paths} Wiener paths on [0, {T}] with 3√t cone')
plt.xlabel('t')
plt.ylabel('W(t)')
plt.legend(loc='upper left')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
