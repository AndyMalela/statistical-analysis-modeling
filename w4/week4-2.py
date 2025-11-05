# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
from pathlib import Path
import re

def display_dataframe_to_user(title, df):
    print(title)
    print(df)



# Reproducibility
rng = np.random.default_rng(42)

N = 1000

# ---- Parameter choices ----
params = {
    "Gaussian (Normal)": {"mu": 0.0, "sigma": 2.0},
    "Exponential": {"lam": 1.0},            # rate λ; scale = 1/λ
    "Gamma": {"k": 3.0, "theta": 2.0},      # shape k, scale θ
}

# ---- Utility: output folder + safe file names ----
OUTDIR = Path("figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "figure"

# ---- Utility functions ----
def freedman_diaconis_bins(x):
    """Return a sensible number of bins via the Freedman–Diaconis rule."""
    x = np.asarray(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        iqr = np.std(x)  # fallback
    h = 2 * iqr * (len(x) ** (-1/3))
    if h <= 0:
        return 30  # fallback
    bins = int(np.ceil((x.max() - x.min()) / h))
    return max(bins, 10)  # keep a minimum for smoother mode estimate

def sample_mode_hist(x):
    """Histogram-based mode estimate."""
    bins = freedman_diaconis_bins(x)
    counts, edges = np.histogram(x, bins=bins)
    idx = np.argmax(counts)
    mode_est = (edges[idx] + edges[idx+1]) / 2
    return mode_est

def central_moments(x):
    x = np.asarray(x)
    mu = x.mean()
    c = x - mu
    m2 = np.mean(c**2)
    m3 = np.mean(c**3)
    m4 = np.mean(c**4)
    return mu, m2, m3, m4

def sample_skewness_kurtosis(x):
    """Return bias-corrected sample skewness and excess kurtosis (Fisher)."""
    x = np.asarray(x)
    n = len(x)
    mu, m2, m3, m4 = central_moments(x)
    if m2 == 0:
        return 0.0, -3.0  # degenerate edge case
    g1 = m3 / (m2 ** 1.5)  # uncorrected skewness
    g2 = m4 / (m2 ** 2) - 3  # uncorrected excess kurtosis
    # Bias corrections (Joanes & Gill, 1998; Fisher-Pearson)
    if n > 2:
        G1 = np.sqrt(n*(n-1)) / (n-2) * g1
    else:
        G1 = g1
    if n > 3:
        G2 = ((n-1)/((n-2)*(n-3))) * ((n+1)*g2 + 6)
    else:
        G2 = g2
    return G1, G2

def mean_confint(x, alpha=0.05):
    """95% CI for the mean using normal approximation (n=1000)."""
    x = np.asarray(x)
    n = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    z = 1.959963984540054  # ~N(0,1) 97.5th percentile
    half_width = z * s / np.sqrt(n)
    return m - half_width, m + half_width

# Theoretical characteristics
def theoretical_characteristics(name, p):
    if name.startswith("Gaussian"):
        mu, sigma = p["mu"], p["sigma"]
        return {
            "true_mean": mu,
            "true_mode": mu,
            "true_skewness": 0.0,
            "true_excess_kurtosis": 0.0,
            "pdf": lambda x: (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
        }
    if name == "Exponential":
        lam = p["lam"]
        return {
            "true_mean": 1/lam,
            "true_mode": 0.0,
            "true_skewness": 2.0,
            "true_excess_kurtosis": 6.0,
            "pdf": lambda x: lam*np.exp(-lam*np.maximum(x, 0)) * (x>=0)
        }
    if name == "Gamma":
        k, theta = p["k"], p["theta"]
        from math import gamma as gamma_fn
        def pdf(x):
            x = np.asarray(x)
            coeff = (x ** (k-1)) * np.exp(-x/theta)
            denom = (theta ** k) * gamma_fn(k)
            return np.where(x>=0, coeff/denom, 0.0)
        mode = (k-1)*theta if k >= 1 else np.nan
        return {
            "true_mean": k*theta,
            "true_mode": mode,
            "true_skewness": 2/np.sqrt(k),
            "true_excess_kurtosis": 6/k,
            "pdf": pdf
        }
    raise ValueError("Unknown distribution")

# ---- Generate samples ----
samples = {}
samples["Gaussian (Normal)"] = rng.normal(loc=params["Gaussian (Normal)"]["mu"],
                                          scale=params["Gaussian (Normal)"]["sigma"],
                                          size=N)

samples["Exponential"] = rng.exponential(scale=1/params["Exponential"]["lam"], size=N)

samples["Gamma"] = rng.gamma(shape=params["Gamma"]["k"],
                             scale=params["Gamma"]["theta"],
                             size=N)

# ---- Compute statistics & assemble comparison table ----
rows = []
for name, x in samples.items():
    p = params[name] if name in params else {}
    theo = theoretical_characteristics(name, p)
    mean = np.mean(x)
    mode_est = sample_mode_hist(x)
    skew, ex_kurt = sample_skewness_kurtosis(x)
    ci_low, ci_high = mean_confint(x)
    rows.append({
        "Distribution": name,
        "N": len(x),
        "Sample mean": mean,
        "Mean 95% CI (low)": ci_low,
        "Mean 95% CI (high)": ci_high,
        "Sample mode (hist)": mode_est,
        "Sample skewness": skew,
        "Sample excess kurtosis": ex_kurt,
        "True mean": theo["true_mean"],
        "True mode": theo["true_mode"],
        "True skewness": theo["true_skewness"],
        "True excess kurtosis": theo["true_excess_kurtosis"],
    })

results_df = pd.DataFrame(rows)

# Display results as a spreadsheet-style table
display_dataframe_to_user("Sample vs. true statistics (N=1000)", results_df.round(4))

# ---- Visualization ----
def plot_dist(name, x, theo):
    plt.figure()
    # Histogram with density
    plt.hist(x, bins=freedman_diaconis_bins(x), density=True, alpha=0.5, label="Sample (hist)")
    # Theoretical pdf curve
    xs = np.linspace(min(x.min(), -5), max(x.max(), 5), 500)
    try:
        ys = theo["pdf"](xs)
        plt.plot(xs, ys, label="Theoretical PDF")
    except Exception:
        pass
    # Vertical lines for means/modes
    plt.axvline(x.mean(), linestyle="--", label="Sample mean")
    plt.axvline(theo["true_mean"], linestyle=":", label="True mean")
    if np.isfinite(theo["true_mode"]):
        plt.axvline(theo["true_mode"], linestyle="-.", label="True mode")
    plt.title(f"{name}: N={len(x)}")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()

    # --- NEW: save as SVG instead of (or in addition to) showing ---
    fname = OUTDIR / f"{slugify(name)}.svg"
    plt.savefig(fname, format="svg", bbox_inches="tight")
    plt.close()  # free memory

for name, x in samples.items():
    theo = theoretical_characteristics(name, params[name])
    plot_dist(name, x, theo)

results_df
