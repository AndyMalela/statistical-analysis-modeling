import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gamma

def generate_poisson_process(lambda_val, max_time):
    """Generate Poisson process arrival and inter-arrival times."""
    arrival_times, inter_arrival_times = [], []
    current_time = 0.0
    
    while True:
        y_i = -np.log(np.random.rand()) / lambda_val
        current_time += y_i
        
        if current_time > max_time:
            break
            
        inter_arrival_times.append(y_i)
        arrival_times.append(current_time)
    
    return np.array(arrival_times), np.array(inter_arrival_times)

# Configuration for Poisson Process
lambdas_poisson = [2, 3, 5, 8]
MAX_TIME_POISSON = 500.0
INTERVAL_LENGTH = 1.0

# Generate Poisson process data
results_poisson = {}
for lam in lambdas_poisson:
    arrivals, inter_arrivals = generate_poisson_process(lam, MAX_TIME_POISSON)
    bins = np.arange(0, MAX_TIME_POISSON + INTERVAL_LENGTH, INTERVAL_LENGTH)
    counts, _ = np.histogram(arrivals, bins=bins)
    results_poisson[lam] = {'arrivals': arrivals, 'inter_arrivals': inter_arrivals, 'counts': counts}

# --- Plot 1: Original Poisson Process Simulation ---
fig_poisson, axes_poisson = plt.subplots(len(lambdas_poisson), 3, figsize=(18, 20), constrained_layout=True)
fig_poisson.suptitle(f'Poisson Process Simulation (T={MAX_TIME_POISSON}) - Auto-scaled Y-Axes', fontsize=20)

for i, lam in enumerate(lambdas_poisson):
    data = results_poisson[lam]
    mu = lam * INTERVAL_LENGTH
    
    # Plot 1: Inter-arrival times
    ax = axes_poisson[i, 0]
    if len(data['inter_arrivals']) > 0:
        ax.hist(data['inter_arrivals'], bins=50, density=True, alpha=0.7, label='Simulation')
        x_fit = np.linspace(0, np.max(data['inter_arrivals']), 100)
        ax.plot(x_fit, lam * np.exp(-lam * x_fit), 'r--', lw=2, label=f'Theoretical Exp(λ={lam})')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'Inter-arrival Times (λ={lam})')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Plot 2: Arrival times
    ax = axes_poisson[i, 1]
    if len(data['arrivals']) > 0:
        ax.hist(data['arrivals'], bins=50, alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'Arrival Times (λ={lam})')
    ax.set_ylabel('Frequency')
    
    # Plot 3: Event counts
    ax = axes_poisson[i, 2]
    if len(data['counts']) > 0:
        max_count = np.max(data['counts'])
        hist_bins = np.arange(0, max_count + 2)
        ax.hist(data['counts'], bins=hist_bins, density=True, alpha=0.7, align='left', label='Simulation')
        
        k_fit = np.arange(0, max_count + 1)
        pmf_fit = []
        for k in k_fit:
            try:
                val = np.exp(-mu) * (mu**k) / math.factorial(k)
                pmf_fit.append(val)
            except (OverflowError, ValueError):
                pmf_fit.append(0)
        
        ax.plot(k_fit, pmf_fit, 'ro', ms=5, label=f'Theoretical Poisson(μ={mu:.1f})')
        ax.set_xticks(k_fit[::max(1, len(k_fit)//10)])
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(f'Event Counts per {INTERVAL_LENGTH}s (λ={lam})')
    ax.set_ylabel('Probability P(N(t)=k)')
    ax.legend()
    
    # Set x-axis labels
    axes_poisson[i, 0].set_xlabel('Time')
    axes_poisson[i, 1].set_xlabel('Time')
    axes_poisson[i, 2].set_xlabel('Count (k)')

plt.show()

# --- Plot 2: Gamma Waiting Times DERIVED From Process Realizations ---

# Alphas (number of arrivals to wait for) to plot
alphas = [2, 5, 10] 
num_alphas = len(alphas)
num_lambdas = len(lambdas_poisson)

# Create a grid: one row per lambda, one column per alpha
fig_gamma, axes_gamma = plt.subplots(num_lambdas, num_alphas, figsize=(num_alphas * 5, num_lambdas * 4), constrained_layout=True)
fig_gamma.suptitle('Gamma Distribution: Waiting Time for α Arrivals', fontsize=20, y=1.03)

for i, lam in enumerate(lambdas_poisson):
    # Get the inter-arrival times from the simulation we already ran
    inter_arrivals = results_poisson[lam]['inter_arrivals']
    
    for j, alpha in enumerate(alphas):
        ax = axes_gamma[i, j]
        
        # Calculate how many full "waiting time" samples we can get
        num_samples = len(inter_arrivals) // alpha
        
        if num_samples > 1:
            # We have enough data to make a histogram
            
            # 1. Get the number of inter-arrivals to use (must be multiple of alpha)
            n_to_use = num_samples * alpha
            
            # 2. Reshape into 'num_samples' chunks, each of size 'alpha'
            chunks = inter_arrivals[:n_to_use].reshape((num_samples, alpha))
            
            # 3. Sum each chunk to get the waiting time for 'alpha' arrivals
            waiting_times = chunks.sum(axis=1)
            
            # 4. Plot the histogram of these waiting times
            ax.hist(waiting_times, bins=max(10, num_samples // 5), density=True, alpha=0.7, 
                    label=f'Simulated Data\n(n={num_samples} samples)')
            
            # 5. Plot the theoretical Gamma PDF
            # In scipy.stats.gamma, a=alpha (shape), scale=1/lambda (rate)
            x_fit = np.linspace(0, np.max(waiting_times), 100)
            pdf = gamma.pdf(x_fit, a=alpha, scale=1/lam)
            ax.plot(x_fit, pdf, 'r--', lw=2, label=f'Theoretical\nGamma(α={alpha}, λ={lam})')
            ax.legend()
            
        else:
            # Not enough data for this alpha
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)

        # Set titles and labels
        if i == 0:
            ax.set_title(f'Wait for α = {alpha} Arrivals')
        if j == 0:
            ax.set_ylabel(f'λ = {lam}\n\nDensity')
        if i == num_lambdas - 1:
            ax.set_xlabel('Time')

plt.show()
