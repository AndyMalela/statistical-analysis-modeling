import numpy as np
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
        # Generate exponentially distributed inter-arrival time
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
    """Plot histogram of inter-arrival times vs exponential distribution."""
    if len(inter_arrivals) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    ax.hist(inter_arrivals, bins=50, density=True, alpha=0.7, label='Simulation')
    
    # Overlay theoretical exponential distribution
    x_fit = np.linspace(0, np.max(inter_arrivals), 100)
    pdf = lambda_val * np.exp(-lambda_val * x_fit)
    ax.plot(x_fit, pdf, 'r--', lw=2, label=f'Theoretical Exp(λ={lambda_val})')
    
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


def plot_event_counts(ax, counts, lambda_val, interval_length):
    """Plot histogram of event counts per interval vs Poisson distribution."""
    if len(counts) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return
    
    mu = lambda_val * interval_length
    max_count = int(np.max(counts))
    
    # Plot histogram
    ax.hist(counts, bins=np.arange(0, max_count + 2), density=True, alpha=0.7, 
            align='left', label='Simulation')
    
    # Overlay theoretical Poisson distribution
    k_vals = np.arange(0, max_count + 1)
    pmf_vals = []
    
    for k in k_vals:
        try:
            pmf = np.exp(-mu) * (mu ** k) / math.factorial(k)
            pmf_vals.append(pmf)
        except (OverflowError, ValueError):
            pmf_vals.append(0)
    
    ax.plot(k_vals, pmf_vals, 'ro', ms=5, label=f'Theoretical Poisson(μ={mu:.1f})')
    ax.set_xticks(k_vals[::max(1, len(k_vals) // 10)])
    
    ax.set_title(f'Event Counts per {interval_length}s (λ={lambda_val})')
    ax.set_xlabel('Count (k)')
    ax.set_ylabel('Probability P(N(t)=k)')
    ax.legend()


def plot_poisson_overview(results_poisson, lambdas, max_time, interval_length):
    """Create comprehensive Poisson process visualization."""
    fig, axes = plt.subplots(len(lambdas), 3, figsize=(18, 20), constrained_layout=True)
    fig.suptitle(f'Poisson Process Simulation (T={max_time})', fontsize=20)
    
    for i, lam in enumerate(lambdas):
        data = results_poisson[lam]
        
        plot_inter_arrivals(axes[i, 0], data['inter_arrivals'], lam)
        plot_arrivals(axes[i, 1], data['arrivals'], lam)
        plot_event_counts(axes[i, 2], data['counts'], lam, interval_length)
    
    plt.show()


# ============================================================================
# GAMMA DISTRIBUTION (Waiting Times)
# ============================================================================

def compute_waiting_times(inter_arrivals, alpha):
    """Calculate waiting times for alpha arrivals by summing inter-arrival times."""
    num_samples = len(inter_arrivals) // alpha
    
    if num_samples < 2:
        return None
    
    # Reshape and sum
    n_to_use = num_samples * alpha
    chunks = inter_arrivals[:n_to_use].reshape((num_samples, alpha))
    waiting_times = chunks.sum(axis=1)
    
    return waiting_times


def plot_gamma_distribution(ax, waiting_times, alpha, lambda_val):
    """Plot histogram of waiting times vs theoretical Gamma distribution."""
    if waiting_times is None or len(waiting_times) < 2:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        return
    
    num_samples = len(waiting_times)
    
    # Plot histogram
    ax.hist(waiting_times, bins=max(10, num_samples // 5), density=True, alpha=0.7,
            label=f'Simulated (n={num_samples})')
    
    # Overlay theoretical Gamma distribution
    x_fit = np.linspace(0, np.max(waiting_times), 100)
    pdf = gamma.pdf(x_fit, a=alpha, scale=1 / lambda_val)
    ax.plot(x_fit, pdf, 'r--', lw=2, label=f'Theoretical Gamma(α={alpha}, λ={lambda_val})')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Density')
    ax.legend()


def plot_gamma_overview(results_poisson, lambdas, alphas):
    """Create comprehensive Gamma waiting time visualization."""
    fig, axes = plt.subplots(len(lambdas), len(alphas), 
                              figsize=(len(alphas) * 5, len(lambdas) * 4), 
                              constrained_layout=True)
    fig.suptitle('Gamma Distribution: Waiting Time for α Arrivals', fontsize=20, y=1.03)
    
    for i, lam in enumerate(lambdas):
        inter_arrivals = results_poisson[lam]['inter_arrivals']
        
        for j, alpha in enumerate(alphas):
            waiting_times = compute_waiting_times(inter_arrivals, alpha)
            plot_gamma_distribution(axes[i, j], waiting_times, alpha, lam)
            
            # Set labels
            if i == 0:
                axes[i, j].set_title(f'Wait for α = {alpha} Arrivals')
            if j == 0:
                axes[i, j].set_ylabel(f'λ = {lam}\n\nDensity')
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    LAMBDAS = [2, 3, 5, 8]
    MAX_TIME = 1000.0
    INTERVAL_LENGTH = 1.0
    ALPHAS = [2, 5, 10]
    
    # Run simulations
    print("Generating Poisson process data...")
    results = simulate_poisson_for_lambdas(LAMBDAS, MAX_TIME, INTERVAL_LENGTH)
    
    # Create visualizations
    print("Creating Poisson process plots...")
    plot_poisson_overview(results, LAMBDAS, MAX_TIME, INTERVAL_LENGTH)
    
    print("Creating Gamma waiting time plots...")
    plot_gamma_overview(results, LAMBDAS, ALPHAS)