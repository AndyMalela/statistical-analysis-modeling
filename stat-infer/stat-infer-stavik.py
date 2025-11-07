# ============================================
# Classical-once, Bootstrap-many (no cross-tests)
# N = 37, seed = 1337
# Normal: mu=12.5, sigma=2  -> c=12.5
# Exponential: lambda=0.08  -> mean=12.5 -> c=12.5
# Bootstrap: SIMS=1000, B=4000
# ============================================

import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
SEED = 1337
N = 37
ALPHA = 0.05

# Monte Carlo + bootstrap
SIMS = 1000
B = 4000

# Models
MU_NORMAL = 12.5
SIGMA_NORMAL = 2.0
LAMBDA_EXP = 0.08
MEAN_EXP = 1.0 / LAMBDA_EXP  # 12.5

# ---------------------------
# Sampling helpers
# ---------------------------
def draw_sample_normal(rng, n=N):
    return rng.normal(loc=MU_NORMAL, scale=SIGMA_NORMAL, size=n)

def draw_sample_exponential(rng, n=N):
    # numpy exponential uses scale = 1/lambda
    return rng.exponential(scale=1.0 / LAMBDA_EXP, size=n)

# ---------------------------
# Test statistics / p-values
# ---------------------------
def t_statistic_one_sample(x, c):
    """
    t = (xbar - c) / (s / sqrt(n)), with s as sample std (ddof=1)
    Returns t, df
    """
    x = np.asarray(x)
    n = x.size
    xbar = x.mean()
    s = x.std(ddof=1)
    t = (xbar - c) / (s / math.sqrt(n))
    df = n - 1
    return t, df

def pvalue_from_t(t, df):
    """Two-sided p-value from Student's t."""
    return 2.0 * (1.0 - stats.t.cdf(abs(t), df))

def bootstrap_algorithm_pvalue(x, c, B, rng):
    """
    8-step algorithm (nonparametric, studentized):
      1) xbar
      2) t_X = (xbar - c) / (sX / sqrt(n))
      3) For b in 1..B: Y = sample with replacement from X (size n)
      4) t_Y = (ybar - xbar) / (sY / sqrt(n))
      5) A += 1 if |t_Y| > |t_X|
      6) Repeat
      7) p = (A+1)/(B+1)
    """
    x = np.asarray(x)
    n = x.size
    xbar = x.mean()
    sX = x.std(ddof=1)
    tX = (xbar - c) / (sX / math.sqrt(n))

    A = 0
    for _ in range(B):
        y = rng.choice(x, size=n, replace=True)
        ybar = y.mean()
        sY = y.std(ddof=1)
        if sY == 0:
            # extremely rare with N=37; skip degenerate resample
            continue
        tY = (ybar - xbar) / (sY / math.sqrt(n))
        if abs(tY) > abs(tX):
            A += 1

    p = (A + 1.0) / (B + 1.0)
    return p

# ---------------------------
# Classical tests ONCE (fixed samples from seed)
# ---------------------------
def classical_once():
    rng = np.random.default_rng(SEED)

    x_norm = draw_sample_normal(rng)
    x_exp  = draw_sample_exponential(rng)

    # Normal vs its own mean (c = 12.5)
    t_n, df_n = t_statistic_one_sample(x_norm, MU_NORMAL)
    p_n = pvalue_from_t(t_n, df_n)

    # Exponential vs its own mean (c = 12.5)
    t_e, df_e = t_statistic_one_sample(x_exp, MEAN_EXP)
    p_e = pvalue_from_t(t_e, df_e)

    print("=== Classical one-sample t-tests (computed ONCE) ===")
    print(f"Normal sample vs c=12.5:   t={t_n:.3f}, df={df_n}, p={p_n:.4f}")
    print(f"Exponential sample vs c=12.5: t={t_e:.3f}, df={df_e}, p={p_e:.4f}")

# ---------------------------
# Bootstrap Monte Carlo (SIMS times, each with B resamples)
# ---------------------------
def bootstrap_monte_carlo(sims=SIMS, B=B, alpha=ALPHA, make_plots=True):
    rng = np.random.default_rng(SEED)

    pvals_norm = np.empty(sims, dtype=float)
    pvals_exp  = np.empty(sims, dtype=float)

    for i in range(sims):
        # Fresh original samples each simulation
        x_norm = draw_sample_normal(rng)
        x_exp  = draw_sample_exponential(rng)

        # 8-step bootstrap p-values (no cross-tests)
        pvals_norm[i] = bootstrap_algorithm_pvalue(x_norm, MU_NORMAL, B=B, rng=rng)
        pvals_exp[i]  = bootstrap_algorithm_pvalue(x_exp,  MEAN_EXP,  B=B, rng=rng)

    # Summaries
    ks_norm = stats.kstest(pvals_norm, 'uniform')
    ks_exp  = stats.kstest(pvals_exp,  'uniform')

    print("\n=== KS test for p-value uniformity ===")
    print(f"Normal:      D={ks_norm.statistic:.3f}, p={ks_norm.pvalue:.3f}")
    print(f"Exponential: D={ks_exp.statistic:.3f}, p={ks_exp.pvalue:.3f}")
    
    summ = {
        "normal": {
            "mean_p": float(pvals_norm.mean()),
            "std_p": float(pvals_norm.std(ddof=1)),
            "sig_rate_%": 100.0 * float(np.mean(pvals_norm < alpha)),  # proportion p<0.05
            "sims": sims,
            "alpha": alpha,
        },
        "exponential": {
            "mean_p": float(pvals_exp.mean()),
            "std_p": float(pvals_exp.std(ddof=1)),
            "sig_rate_%": 100.0 * float(np.mean(pvals_exp < alpha)),  # proportion p<0.05
            "sims": sims,
            "alpha": alpha,
        }
    }

    print("\n=== Bootstrap Monte Carlo (8-step) over simulations ===")
    print(f"Normal:      mean p={summ['normal']['mean_p']:.3f}, sd p={summ['normal']['std_p']:.3f}, "
          f"% p<0.05={summ['normal']['sig_rate_%']:.2f}%, sims={sims}")
    print(f"Exponential: mean p={summ['exponential']['mean_p']:.3f}, sd p={summ['exponential']['std_p']:.3f}, "
          f"% p<0.05={summ['exponential']['sig_rate_%']:.2f}%, sims={sims}")


    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].hist(pvals_norm, bins=30)
        axes[0].axvline(ALPHA, linestyle="--")
        axes[0].set_title("Bootstrap p-values (Normal)")
        axes[0].set_xlabel("p-value"); axes[0].set_ylabel("count")

        axes[1].hist(pvals_exp, bins=30)
        axes[1].axvline(ALPHA, linestyle="--")
        axes[1].set_title("Bootstrap p-values (Exponential)")
        axes[1].set_xlabel("p-value"); axes[1].set_ylabel("count")

        plt.suptitle(f"Bootstrap p-value distributions across {sims} simulations (B={B}, N={N})")
        plt.show()
        
    return {"normal": pvals_norm, "exponential": pvals_exp}, summ

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    classical_once()
    _, summary = bootstrap_monte_carlo(sims=SIMS, B=B, alpha=ALPHA, make_plots=True)
    
)