import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.optimize import fminbound

# Parameters
lambda_val = 8
S = 101  # Shape parameter (wait until S events)
total_samples = 1000
h = 0.0001

# ============================================================
# Generate Gamma samples using Poisson process
# ============================================================

def generate_one_gamma_sample(lambda_val, S, h):
    """
    Run Poisson process until S events occur.
    Return the total time elapsed.
    """
    t = 0
    N = 0
    
    while N < S:
        if np.random.rand() < lambda_val * h:
            N += 1
        t += h
    
    return t

# Generate 2000 samples
print("Generating 2000 Gamma samples...")
gamma_samples = []
for i in range(total_samples):
    sample = generate_one_gamma_sample(lambda_val, S, h)
    gamma_samples.append(sample)

gamma_samples = np.array(gamma_samples)

print(f"Sample mean: {np.mean(gamma_samples):.4f} (expected: {S/lambda_val:.4f})")
print(f"Sample variance: {np.var(gamma_samples):.4f} (expected: {S/lambda_val**2:.4f})")

# ============================================================
# MLE Estimation
# ============================================================

def estimate_gamma_mle(data):
    """
    Estimate Gamma(k, λ) parameters using MLE.
    Solve: digamma(k) - ln(k) = ln(mean) - mean(ln(x))
    Then: λ = k / mean
    """
    mean_x = np.mean(data)
    mean_ln_x = np.mean(np.log(data))
    
    # Objective function to minimize
    def objective(k):
        return abs(digamma(k) - np.log(k) - (np.log(mean_x) - mean_ln_x))
    
    # Find k
    k_hat = fminbound(objective, 0.1, 1000)
    lambda_hat = k_hat / mean_x
    
    return k_hat, lambda_hat

# Compute MLE for each of the 2000 samples
print("\nComputing MLE for each sample...")
mle_k_estimates = []
mle_lambda_estimates = []

for i in range(total_samples):
    # Each gamma_samples[i] is treated as a single observation
    # We need multiple observations to estimate parameters
    # So we'll draw a small sample around each point
    # OR: we pool all data and estimate once, then resample
    pass

# Better approach: for each sample, generate a small subsample and estimate
# Actually, simplest: treat each sample as one draw, pool all 2000 to get one MLE
# But the problem asks for MLE for EACH of 2000 samples...

# Let's generate 2000 small datasets instead
print("Generating 2000 small datasets and computing MLE for each...")
mle_k_estimates = []
mle_lambda_estimates = []

for i in range(total_samples):
    # Generate a small sample of size n from Gamma(S, lambda_val)
    small_sample_size = 50
    small_sample = np.array([generate_one_gamma_sample(lambda_val, S, h) 
                             for _ in range(small_sample_size)])
    
    k_hat, lambda_hat = estimate_gamma_mle(small_sample)
    mle_k_estimates.append(k_hat)
    mle_lambda_estimates.append(lambda_hat)
    
    print(f"  Sample {i + 1}/{total_samples} completed")
    
    if (i + 1) % 500 == 0:
        print(f"    --> {i + 1}/{total_samples} ✓")

mle_k_estimates = np.array(mle_k_estimates)
mle_lambda_estimates = np.array(mle_lambda_estimates)

print(f"\nMLE Results:")
print(f"Mean k estimate: {np.mean(mle_k_estimates):.2f} (true k = {S})")
print(f"Std k estimate: {np.std(mle_k_estimates):.2f}")
print(f"Mean λ estimate: {np.mean(mle_lambda_estimates):.4f} (true λ = {lambda_val})")
print(f"Std λ estimate: {np.std(mle_lambda_estimates):.4f}")

# ============================================================
# Plotting
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of k estimates
axes[0].hist(mle_k_estimates, bins=50, density=True, alpha=0.7, edgecolor='black', color='blue')
axes[0].axvline(S, color='red', linestyle='--', linewidth=2.5, label=f'True k = {S}')
axes[0].axvline(np.mean(mle_k_estimates), color='green', linestyle='--', linewidth=2.5, label=f'Mean estimate = {np.mean(mle_k_estimates):.2f}')
axes[0].set_xlabel('k estimate', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'Distribution of MLE k Estimates (n=2000)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Histogram of λ estimates
axes[1].hist(mle_lambda_estimates, bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
axes[1].axvline(lambda_val, color='red', linestyle='--', linewidth=2.5, label=f'True λ = {lambda_val}')
axes[1].axvline(np.mean(mle_lambda_estimates), color='green', linestyle='--', linewidth=2.5, label=f'Mean estimate = {np.mean(mle_lambda_estimates):.4f}')
axes[1].set_xlabel('λ estimate', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title(f'Distribution of MLE λ Estimates (n=2000)', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()