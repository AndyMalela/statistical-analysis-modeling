import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_val = 3
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
plt.xlabel('Time')
plt.ylabel('Event Count')
plt.title('Poisson Process Simulation (λ=3 h=0.0001)')
plt.show()
