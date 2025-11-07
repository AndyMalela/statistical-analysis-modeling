import numpy as np
import matplotlib
matplotlib.use('Agg')  # force non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import math
from scipy.stats import gamma

# ============================================================================
# POISSON PROCESS
# ============================================================================

def generate_poisson_process(lambda_val, max_time):
    """Generate arrival times and inter-arrival times for a Poisson process."""
    arrivals, inter_arrivals = [], []
    current_time = 0.0

    while True:
        y_i = -np.log(np.random.rand()) / lambda_val
        current_time += y_i
        if current_time > max_time:
            break
        inter_arrivals.append(y_i)
        arrivals.append(current_time)

    return np.array(arrivals), np.array(inter_arrivals)


def count_arrivals_per_interval(arrivals, interval_length, max_time):
    """Bin arrivals into fixed time intervals."""
    bins = np.arange(0, max_time + interval_length, interval_length)
    counts, _ = np.histogram(arrivals, bins=bins)
    return counts


def simulate_poisson_for_lambdas(lambdas, max_time, interval_length):
    """Run Poisson simulation for multiple lambda values."""
    results = {}
    for lam in lambdas:
        arrivals, inter_arrivals = generate_poisson_process(lam, max_time)
        counts = count_arrivals_per_interval(arrivals, interval_length, max_time)
        results[lam] = {
            'arrivals': arrivals,
            'inter_arrivals': inter_arrivals,
            'counts': counts
        }
    return results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_inter_arrivals(ax, inter_arrivals, lambda_val):
    """Plot histogram of inter-arrival times (no theoretical curve)."""
    if len(inter_arrivals) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    ax.hist(inter_arrivals, bins=50, density=True, alpha=0.7, label='Simulated')
    ax.set_title(f'Inter-arrival Times (λ={lambda_val})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Density')
    ax.legend()


def plot_arrivals(ax, arrivals, lambda_val):
    """Plot histogram of arrival times."""
    if len(arrivals) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    ax.hist(arrivals, bins=100, alpha=0.7)
    ax.set_title(f'Arrival Times (λ={lambda_val})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')


def plot_counting_process(ax, arrivals, lambda_val, max_time):
    """Plot right-continuous step realization N(t) of the Poisson process."""
    times = np.concatenate(([0.0], arrivals, [max_time]))
    counts = np.arange(0, len(times))
    counts = counts[:len(times)]
    if len(times) >= 2:
        ax.step(times, counts, where='post')
    else:
        ax.hlines(0, 0, max_time)
    ax.set_xlim(0, max_time)
    ax.set_ylim(bottom=0)
    ax.set_title(f'Counting Process N(t) (λ={lambda_val})')
    ax.set_xlabel('t')
    ax.set_ylabel('N(t)')


def plot_poisson_overview(results_poisson, lambdas, max_time, interval_length):
    """Create comprehensive Poisson process visualization."""
    fig, axes = plt.subplots(len(lambdas), 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f'Poisson Process Simulation with T={max_time}', fontsize=20)

    if len(lambdas) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, lam in enumerate(lambdas):
        data = results_poisson[lam]
        plot_inter_arrivals(axes[i, 0], data['inter_arrivals'], lam)
        plot_arrivals(axes[i, 1], data['arrivals'], lam)
        plot_counting_process(axes[i, 2], data['arrivals'], lam, max_time)

    plt.savefig("poisson-overview.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# GAMMA DISTRIBUTION (Waiting Times)
# ============================================================================

def compute_waiting_times(inter_arrivals, k):
    """Calculate waiting times for k arrivals by summing inter-arrival times."""
    num_samples = len(inter_arrivals) // k
    if num_samples < 2:
        return None
    n_to_use = num_samples * k
    chunks = inter_arrivals[:n_to_use].reshape((num_samples, k))
    waiting_times = chunks.sum(axis=1)
    return waiting_times


def plot_gamma_distribution(ax, waiting_times, k, lambda_val):
    """Plot histogram of waiting times (no theoretical gamma curve)."""
    if waiting_times is None or len(waiting_times) < 2:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        return
    num_samples = len(waiting_times)
    ax.hist(waiting_times, bins=max(10, num_samples // 5), density=True, alpha=0.7,
            label=f'Simulated (n={num_samples})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Density')
    ax.legend()


def plot_gamma_overview(results_poisson, lambdas, ks):
    """Create comprehensive Gamma waiting time visualization."""
    fig, axes = plt.subplots(len(lambdas), len(ks),
                             figsize=(len(ks) * 5, len(lambdas) * 2.5),
                             constrained_layout=True)
    fig.suptitle('Waiting Time for k Arrivals', fontsize=20, y=1.03)

    if len(lambdas) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(ks) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, lam in enumerate(lambdas):
        inter_arrivals = results_poisson[lam]['inter_arrivals']
        for j, k in enumerate(ks):
            waiting_times = compute_waiting_times(inter_arrivals, k)
            plot_gamma_distribution(axes[i, j], waiting_times, k, lam)
            if i == 0:
                axes[i, j].set_title(f'Wait for k = {k} Arrivals')
            if j == 0:
                axes[i, j].set_ylabel(f'λ = {lam}\n\nDensity')

    plt.savefig("gamma-overview.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    LAMBDAS = [2, 3, 5, 10]
    MAX_TIME = 10000.0
    INTERVAL_LENGTH = 1.0
    KS = [4]

    print("Generating Poisson process data...")
    results = simulate_poisson_for_lambdas(LAMBDAS, MAX_TIME, INTERVAL_LENGTH)

    print("Creating Poisson process plots...")
    plot_poisson_overview(results, LAMBDAS, MAX_TIME, INTERVAL_LENGTH)

    print("Creating Gamma waiting time plots...")
    plot_gamma_overview(results, LAMBDAS, KS)
