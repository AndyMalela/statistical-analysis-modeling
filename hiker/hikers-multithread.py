# random_walk_hikers_stats_parallel.py
# - One example plot of 3 hikers' trajectories
# - 4000 simulations in parallel with progress bar
# - CSV export of all runs (ΔX, ΔY, N) for each hiker
# - Distribution plots (hexbin, rose histogram, ECDF, boxplot, running mean CI)

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Tunable parameters
# -----------------------------
RADIUS_METERS = 1000          # stopping radius (meters)
CHUNK = 200_000               # steps per batch (speed/memory tradeoff)
SEED = 99                     # seed for reproducibility
NUM_RUNS = 4000               # total independent simulations to collect stats
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)  # parallel workers
# -----------------------------


def simulate_walk(R=RADIUS_METERS, chunk=CHUNK, rng=None):
    """Return (x_path, y_path, steps_to_hit). 1 m step length, angle ~ U[0, 2π)."""
    if rng is None:
        rng = np.random.default_rng()
    x, y = 0.0, 0.0
    xs, ys = [x], [y]
    steps = 0
    R2 = R * R
    while True:
        angles = rng.random(chunk) * (2 * np.pi)
        dx = np.cos(angles)
        dy = np.sin(angles)
        cumx = np.cumsum(dx) + x
        cumy = np.cumsum(dy) + y
        radii2 = cumx * cumx + cumy * cumy
        hit_mask = radii2 >= R2
        if not np.any(hit_mask):
            xs.extend(cumx.tolist()); ys.extend(cumy.tolist())
            x, y = cumx[-1], cumy[-1]
            steps += chunk
        else:
            first_idx = int(np.argmax(hit_mask))
            xs.extend(cumx[:first_idx+1].tolist())
            ys.extend(cumy[:first_idx+1].tolist())
            steps += first_idx + 1
            break
    return np.array(xs), np.array(ys), steps


def simulate_hiker_group(seed_tuple):
    """Run one simulation (3 hikers) and return rows: (run, hiker, ΔX, ΔY, N)."""
    run_idx, base_seed = seed_tuple
    rng = np.random.default_rng(base_seed)
    rows = []
    for k in range(3):
        hiker_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        x_path, y_path, n = simulate_walk(rng=hiker_rng)
        rows.append((run_idx, k + 1, float(x_path[-1]), float(y_path[-1]), int(n)))
    return rows


def ecdf(vals):
    """x, y for empirical CDF."""
    v = np.sort(np.asarray(vals))
    y = np.arange(1, v.size + 1) / v.size
    return v, y


def running_mean_ci(x, alpha=0.05):
    """
    Running mean with a (1-alpha) CI using normal approx and running variance.
    Returns n, mean, lo, hi (each length len(x)).
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(1, len(x) + 1)
    mean = np.empty_like(x, dtype=float)
    var = np.empty_like(x, dtype=float)

    # Welford’s algorithm
    m = 0.0
    s = 0.0
    for i, xi in enumerate(x, start=1):
        delta = xi - m
        m += delta / i
        s += delta * (xi - m)
        mean[i-1] = m
        var[i-1] = s / (i - 1) if i > 1 else 0.0

    z = 1.96  # ~95% normal quantile
    se = np.sqrt(var / n)
    lo = mean - z * se
    hi = mean + z * se
    return n, mean, lo, hi


def example_plot(rng):
    """Make a single example trajectory plot for three hikers."""
    example_results = []
    for _ in range(3):
        hiker_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        x_path, y_path, steps = simulate_walk(rng=hiker_rng)
        example_results.append((x_path, y_path, steps))

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (x_path, y_path, steps) in enumerate(example_results, start=1):
        ax.plot(x_path, y_path, color=colors[i-1], linewidth=0.8, alpha=0.6,
                label=f"Hiker {i} ({steps:,} steps)")
        ax.scatter(x_path[-1], y_path[-1], color=colors[i-1], edgecolor='black',
                   s=25, zorder=5)
    ax.scatter(0, 0, color='black', s=30, label='Start', zorder=6)
    theta = np.linspace(0, 2*np.pi, 600)
    ax.plot(RADIUS_METERS*np.cos(theta), RADIUS_METERS*np.sin(theta),
            linestyle='--', linewidth=0.8, color='gray',
            label='Help boundary (1000 m)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('2D Random Walks — Example Single Simulation')
    ax.legend(loc='best')
    plt.show()


def run_parallel_simulations(num_runs=NUM_RUNS):
    """Run num_runs groups of 3 hikers in parallel and return a DataFrame."""
    base_rng = np.random.default_rng(SEED)
    seeds = [(i + 1, int(base_rng.integers(0, 2**63 - 1))) for i in range(num_runs)]
    all_rows = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(simulate_hiker_group, s) for s in seeds]
        with tqdm(total=num_runs, desc="Simulations", unit="sim") as pbar:
            for f in as_completed(futures):
                all_rows.extend(f.result())
                pbar.update(1)

    df = pd.DataFrame(all_rows, columns=["run", "hiker", "delta_x", "delta_y", "steps"])
    return df


def plots_from_dataframe(df: pd.DataFrame):
    """Make distribution plots (hexbin, rose, ECDF, boxplot, running mean CI)."""
    # (a) Hexbin per hiker
    for k in [1, 2, 3]:
        sub = df[df["hiker"] == k]
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        hb = ax.hexbin(sub["delta_x"], sub["delta_y"], gridsize=55, mincnt=1)
        cb = fig.colorbar(hb, ax=ax); cb.set_label('Count')
        theta = np.linspace(0, 2*np.pi, 600)
        ax.plot(RADIUS_METERS*np.cos(theta), RADIUS_METERS*np.sin(theta),
                linestyle='--', linewidth=1.0, color='gray')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Exit Locations (ΔX, ΔY) — Hiker {k}')
        ax.set_xlabel('ΔX (m)'); ax.set_ylabel('ΔY (m)')
        plt.show()

    # (b) Polar rose histogram of exit angles overlaid
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection='polar')
    bins = 32
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for k, color in zip([1, 2, 3], colors):
        sub = df[df["hiker"] == k]
        angles = np.arctan2(sub["delta_y"].to_numpy(), sub["delta_x"].to_numpy())
        angles = (angles + 2*np.pi) % (2*np.pi)
        counts, edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
        widths = np.diff(edges)
        ax.bar(edges[:-1], counts, width=widths, bottom=0.0,
               alpha=0.45, edgecolor='none', label=f'Hiker {k}', color=color)
    ax.set_title('Exit Angle Distribution (Rose Histogram)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.12))
    plt.show()

    # (c) ECDF of steps
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in [1, 2, 3]:
        sub = df[df["hiker"] == k]["steps"].to_numpy()
        xv, yv = ecdf(sub)
        ax.plot(xv, yv, label=f'Hiker {k}')
    ax.set_xlabel('Steps to 1000 m (N)')
    ax.set_ylabel('ECDF')
    ax.set_title('ECDF of Steps to Reach 1000 m')
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.legend()
    plt.show()

    # (d) Boxplot of steps by hiker
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    data = [df[df["hiker"] == k]["steps"].to_numpy() for k in [1, 2, 3]]
    ax.boxplot(data, labels=['Hiker 1', 'Hiker 2', 'Hiker 3'],
               showmeans=True, meanline=True)
    ax.set_ylabel('Steps to 1000 m (N)')
    ax.set_title('Distribution of N by Hiker')
    ax.grid(True, axis='y', linestyle=':', linewidth=0.7)
    plt.show()

    # (e) Running mean with 95% CI
    for k in [1, 2, 3]:
        vals = df[df["hiker"] == k]["steps"].to_numpy()
        # Shuffle order for a representative running mean curve
        order = np.random.default_rng(SEED + 100 + k).permutation(len(vals))
        n, mean, lo, hi = running_mean_ci(vals[order], alpha=0.05)
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(n, mean, label=f'Hiker {k} mean')
        ax.fill_between(n, lo, hi, alpha=0.25, label='95% CI')
        ax.set_xlabel('Number of simulations')
        ax.set_ylabel('Running mean of N')
        ax.set_title(f'Running Mean of N with 95% CI — Hiker {k}')
        ax.grid(True, linestyle=':', linewidth=0.7)
        ax.legend()
        plt.show()


def print_summary(df: pd.DataFrame):
    print(f"\nSeed = {SEED} | Runs = {NUM_RUNS} | Workers = {NUM_WORKERS}")
    for k in [1, 2, 3]:
        sub = df[df["hiker"] == k]
        mean_dx = sub["delta_x"].mean()
        mean_dy = sub["delta_y"].mean()
        mean_n  = sub["steps"].mean()
        std_n   = sub["steps"].std(ddof=1)
        print(f"Hiker {k}: "
              f"E[ΔX]={mean_dx: .2f}, E[ΔY]={mean_dy: .2f}, "
              f"E[N]={mean_n:,.1f} (SD={std_n:,.1f})")


def main():
    rng = np.random.default_rng(SEED)

    # 1) Example single-simulation plot
    example_plot(rng)

    # 2) Parallel simulations with progress bar
    df = run_parallel_simulations(NUM_RUNS)

    # 3) Save CSV
    out_path = Path(f"hiker_randomwalk_runs_seed{SEED}_n{NUM_RUNS}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved all {NUM_RUNS*3:,} records to {out_path.resolve()}")

    # 4) Summary + plots
    print_summary(df)
    plots_from_dataframe(df)


if __name__ == "__main__":
    # Important for Windows multiprocessing
    main()
