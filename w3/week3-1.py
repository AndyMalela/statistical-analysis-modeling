import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_val = 2
total_t = 100
h = 0.0001

# Initialization
t = 0
N = 0
time_points = []
event_counts = []

# Simulation
while t < total_t:
    if np.random.rand() < lambda_val * h:
        N += 1
    t += h
    time_points.append(t)
    event_counts.append(N)

# Plot results
plt.plot(time_points, event_counts, drawstyle='steps-post')
plt.xlabel(f'Time')
plt.ylabel(f'Event Count')
plt.title(f'Poisson Process Simulation (λ={lambda_val} h={h})')
plt.show()
