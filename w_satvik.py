import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

lmbda = 3
total_t = 100
h = 0.0001
samples = 2000

estimate_shapes = []
estimate_scales = []

for sample_idx in range(samples):
    print(f"Running sample {sample_idx+1}/{samples}") 

    N = 0
    t = 0
    time_point = []
    event_count = []
    event_times = []

    for i in range(int(total_t / h)):
        U = np.random.rand()
        if U < lmbda * h:
            N += 1
            event_times.append(t)
        t += h
        time_point.append(t)
        event_count.append(N)

    # Compute interarrival times
    if len(event_times) > 1:
        inter_arrival_times = np.diff(event_times)
        # Fit gamma using MLE
        shape, loc, scale = gamma.fit(inter_arrival_times, floc=0)
        estimate_shapes.append(shape)
        estimate_scales.append(scale)

# Plot histograms of estimated parameters
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(estimate_shapes, bins=30, color='blue', edgecolor='black')
plt.title("Histogram of Shape Estimates")
plt.xlabel("Shape Estimate k")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(estimate_scales, bins=30, color='green', edgecolor='black')
plt.title("Histogram of Scale Estimates")
plt.xlabel("Scale Estimate θ")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
