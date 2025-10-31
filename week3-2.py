import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist

# ============================================================
# Parameters
# ============================================================
lambda_val = 8.0     # true rate λ
S = 101              # wait until S events (true shape k)
num_samples = 2000   # number of Gamma samples
h = 1e-4             # Bernoulli time step
save_file = f"gamma_samples_S{S}_lam{int(lambda_val)}.npz"

# ============================================================
# Poisson-process simulator (Bernoulli step)
# ============================================================

def generate_one_gamma_sample(lambda_val, S, h):
    """Simulate a Poisson process via Bernoulli(λh) trials until S events occur.
       Returns the total time to the S-th event (Gamma(S, λ))."""
    t = 0.0
    N = 0
    p = lambda_val * h
    while N < S:
        if np.random.rand() < p:
            N += 1
        t += h
    return t

# ============================================================
# Generate or load 2000 Gamma samples
# ============================================================

if os.path.exists(save_file):
    print(f"Found saved samples '{save_file}'. Loading...")
    data = np.load(save_file)
    gamma_samples = data["gamma_samples"]
else:
    print(f"No saved samples found. Generating {num_samples} Gamma samples...")
    gamma_samples = np.array([generate_one_gamma_sample(lambda_val, S, h)
                              for _ in range(num_samples)], dtype=float)
    np.savez(save_file, gamma_samples=gamma_samples,
             lambda_val=lambda_val, S=S, h=h)
    print(f"Saved {num_samples} samples to '{save_file}'")

# ============================================================
# Fit one Gamma distribution via MLE
# ============================================================

a_hat, loc_hat, scale_hat = gamma_dist.fit(gamma_samples, floc=0)
k_hat = a_hat
theta_hat = scale_hat
lambda_hat = 1.0 / theta_hat

# ============================================================
# Print results
# ============================================================

print("\n=== MLE Results from 2000 Gamma samples ===")
print(f"Estimated shape (k̂): {k_hat:.3f}   | True k = {S}")
print(f"Estimated rate  (λ̂): {lambda_hat:.4f} | True λ = {lambda_val}")
print(f"Estimated scale (θ̂): {theta_hat:.4f} | True θ = {1/lambda_val:.4f}")

# Sanity check: sample mean & variance vs theoretical
sample_mean = np.mean(gamma_samples)
sample_var = np.var(gamma_samples)
theory_mean = S / lambda_val
theory_var = S / (lambda_val ** 2)

print(f"\nSample mean: {sample_mean:.4f} (theoretical {theory_mean:.4f})")
print(f"Sample var : {sample_var:.4f} (theoretical {theory_var:.4f})")

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(gamma_samples, bins=60, density=True, alpha=0.7,
        edgecolor='black', color='skyblue', label='Simulated data')

# Fitted Gamma PDF
x = np.linspace(min(gamma_samples), max(gamma_samples), 500)
ax.plot(x, gamma_dist.pdf(x, k_hat, loc=0, scale=theta_hat),
        'r-', lw=2, label='Fitted Gamma PDF')
ax.axvline(sample_mean, color='green', linestyle='--', lw=2,
           label=f"Sample mean = {sample_mean:.2f}")
ax.axvline(theory_mean, color='orange', linestyle='--', lw=2,
           label=f"Theoretical mean = {theory_mean:.2f}")

ax.set_title(f"Gamma MLE from 2000 samples (S={S}, λ={lambda_val})")
ax.set_xlabel("t (time to S-th event)")
ax.set_ylabel("Density")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
