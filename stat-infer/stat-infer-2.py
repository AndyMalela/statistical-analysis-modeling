# ============================================
# One-sample t-test vs. bootstrap algorithm
# N=37, seed=1337, B=4000, sims=1000
# Models:
#   M1: Normal(mu=12.5, sigma=2)
#   M2: Exponential(lam=0.08)  # mean = 1/lam = 12.5
# Tests: each sample mean vs each population mean (4 tests total)
# ============================================

import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt

# ---------------------------
# Configuration
# ---------------------------
SEED = 1337
rng = np.random.default_rng(SEED)

N = 37
B = 4000         # bootstrap iterations per test
SIMS = 1000      # number of repeated simulations for significance estimation
ALPHA = 0.05

# Probability models and their population means
MU_NORMAL = 12.5
SIGMA_NORMAL = 2.0

LAMBDA_EXP = 0.08
MEAN_EXP = 1.0 / LAMBDA_EXP  # 12.5

# We will test against c equal to each model's population mean:
C_VALUES = {
    "normal_mean": MU_NORMAL,
    "exp_mean": MEAN_EXP
}

# ---------------------------
# Helpers
# ---------------------------
def t_statistic_one_sample(x, c):
    """
    Slide 14 statistic:
    t_X = (xbar - c) / (s / sqrt(n)), where s is sample std with ddof=1
    Returns (t, df)
    """
    x = np.asarray(x)
    n = x.size
    xbar = x.mean()
    s = x.std(ddof=1)
    t = (xbar - c) / (s / math.sqrt(n))
    df = n - 1
    return t, df

def ttest_pvalue_from_stat(t, df):
    """Two-sided p-value from t-statistic and df."""
    # p = 2 * (1 - F(|t|)), where F is CDF of Student-t(df)
    return 2.0 * (1.0 - stats.t.cdf(abs(t), df))

def bootstrap_algorithm_pvalue(x, c, B=4000, rng=None):
    """
    Implements Steps 1–7:
      1) compute xbar
      2) compute t_X = (xbar - c) / (s_X / sqrt(N))
      3) for b=1..B:
           - draw Y by sampling with replacement from X, size N
      4) compute t_Y = (ybar - xbar) / (s_Y / sqrt(N))
      5) A += 1 if |t_Y| > |t_X|
      6) repeat
      7) p = (A + 1) / (B + 1)
    Returns p-value.
    """
    if rng is None:
        rng = np.random.default_rng()

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
        # Protect against zero std in degenerate bootstrap samples (rare with N=37)
        if sY == 0:
            continue
        tY = (ybar - xbar) / (sY / math.sqrt(n))
        if abs(tY) > abs(tX):
            A += 1

    p = (A + 1.0) / (B + 1.0)
    return p

def draw_sample(model, rng):
    """
    model in {"normal", "exp"}.
    Returns a sample of length N from the specified model.
    """
    if model == "normal":
        return rng.normal(loc=MU_NORMAL, scale=SIGMA_NORMAL, size=N)
    elif model == "exp":
        # numpy exponential uses scale=1/lambda
        return rng.exponential(scale=1.0 / LAMBDA_EXP, size=N)
    else:
        raise ValueError("Unknown model")

# ---------------------------
# Single-run (one set of two samples) utilities
# ---------------------------
def run_four_tests_once(x_norm, x_exp, B, rng):
    """
    Given one sample from each model, run the 4 tests:
      (x_norm vs normal_mean), (x_norm vs exp_mean),
      (x_exp  vs normal_mean), (x_exp  vs exp_mean)
    Each tested by:
      a) slide-14 t-test (analytic)
      b) bootstrap algorithm
    Returns dictionary of p-values.
    """
    results = {}

    # Define all combinations
    combos = [
        ("x_norm_vs_normal", x_norm, C_VALUES["normal_mean"]),
        ("x_norm_vs_exp",    x_norm, C_VALUES["exp_mean"]),
        ("x_exp_vs_normal",  x_exp,  C_VALUES["normal_mean"]),
        ("x_exp_vs_exp",     x_exp,  C_VALUES["exp_mean"]),
    ]

    for name, sample, c in combos:
        # Theoretical t-test (slide 14)
        t, df = t_statistic_one_sample(sample, c)
        p_theory = ttest_pvalue_from_stat(t, df)

        # Bootstrap algorithm
        p_boot = bootstrap_algorithm_pvalue(sample, c, B=B, rng=rng)

        results[name] = {
            "t_stat": float(t),
            "df": int(df),
            "p_theory": float(p_theory),
            "p_boot": float(p_boot),
            "n": len(sample),
            "c": float(c)
        }

    return results

# ---------------------------
# Repeated simulations to estimate significance (Type I error)
# ---------------------------
def estimate_significance(sims=SIMS, B=B, rng=None, make_plots=True):
    """
    Repeats Steps 1–7 'sims' times for each of the 4 tests and
    estimates the test significance (percentage of p-values < alpha).

    Returns a dict with p-value arrays and summary stats.
    Optionally shows plots of p-value distributions.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Store p-values across simulations
    acc = {
        "x_norm_vs_normal": {"theory": [], "boot": []},
        "x_norm_vs_exp":    {"theory": [], "boot": []},
        "x_exp_vs_normal":  {"theory": [], "boot": []},
        "x_exp_vs_exp":     {"theory": [], "boot": []},
    }

    for _ in range(sims):
        x_norm = draw_sample("normal", rng)
        x_exp  = draw_sample("exp", rng)
        res = run_four_tests_once(x_norm, x_exp, B=B, rng=rng)
        for key in acc:
            acc[key]["theory"].append(res[key]["p_theory"])
            acc[key]["boot"].append(res[key]["p_boot"])

    # Convert to arrays
    for key in acc:
        acc[key]["theory"] = np.array(acc[key]["theory"])
        acc[key]["boot"]   = np.array(acc[key]["boot"])

    # Summaries (empirical significance at alpha)
    summary = {}
    for key in acc:
        p_theory = acc[key]["theory"]
        p_boot   = acc[key]["boot"]
        summary[key] = {
            "alpha": ALPHA,
            "sims": sims,
            "sig_rate_theory_%": 100.0 * np.mean(p_theory < ALPHA),
            "sig_rate_boot_%":   100.0 * np.mean(p_boot   < ALPHA),
            "p_theory_mean": float(p_theory.mean()),
            "p_boot_mean":   float(p_boot.mean())
        }

    # Optional plots
    if make_plots:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        titles = [
            ("x_norm_vs_normal", "Normal sample vs Normal mean (c=12.5)"),
            ("x_norm_vs_exp",    "Normal sample vs Exp mean (c=12.5)"),
            ("x_exp_vs_normal",  "Exponential sample vs Normal mean (c=12.5)"),
            ("x_exp_vs_exp",     "Exponential sample vs Exp mean (c=12.5)")
        ]

        for ax, (key, ttl) in zip(axes.flat, titles):
            ax.hist(acc[key]["theory"], bins=30, alpha=0.6, label="slide14 t-test")
            ax.hist(acc[key]["boot"],   bins=30, alpha=0.6, label="bootstrap algo")
            ax.axvline(ALPHA, linestyle="--", label=f"alpha={ALPHA}")
            ax.set_title(ttl)
            ax.set_xlabel("p-value")
            ax.set_ylabel("count")
            ax.legend()

        plt.suptitle("P-value distributions across simulations (N=37, B=4000, sims=1000)")
        plt.show()

    return {"pvals": acc, "summary": summary}

# ---------------------------
# Run once (one instance) + repeated simulations
# ---------------------------
if __name__ == "__main__":
    # One concrete realization (so you can see the four test results once)
    x_norm_once = draw_sample("normal", rng)
    x_exp_once  = draw_sample("exp", rng)
    one_run = run_four_tests_once(x_norm_once, x_exp_once, B=B, rng=rng)

    print("=== Single-run results (one set of samples) ===")
    for k, v in one_run.items():
        print(f"{k}: t={v['t_stat']:.3f}, df={v['df']}, p_theory={v['p_theory']:.4f}, p_boot={v['p_boot']:.4f}")

    # Repeated simulations to estimate significance under H0 (since c matches true means)
    out = estimate_significance(sims=SIMS, B=B, rng=rng, make_plots=True)

    print("\n=== Empirical significance (Type I error %) at alpha=0.05 over sims ===")
    for k, v in out["summary"].items():
        print(f"{k}: theory={v['sig_rate_theory_%']:.2f}%, boot={v['sig_rate_boot_%']:.2f}% "
              f"(mean p: theory={v['p_theory_mean']:.3f}, boot={v['p_boot_mean']:.3f})")
