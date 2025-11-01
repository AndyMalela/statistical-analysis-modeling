import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')  # force non-interactive backend
from scipy.special import erfinv

def generate_wiener_process_3d(sigma, mu, num_steps, dt):
    """Generate a 3D Wiener process with drift mu and volatility sigma."""
    path = np.zeros((num_steps + 1, 3))
    
    for i in range(1, num_steps + 1):
        # Generate standard normal increments for 3D
        Z = np.sqrt(2) * erfinv(2 * np.random.rand(3) - 1)
        
        # Wiener increment: dW = mu*dt + sigma*sqrt(dt)*Z
        path[i] = path[i-1] + mu * dt + sigma * np.sqrt(dt) * Z
    
    return path

# Parameters
mu = 0.1
num_steps = 1000
dt = 0.01
sigma_values = [1.0, 2.0]

# Generate processes
paths = [generate_wiener_process_3d(sigma=s, mu=mu, num_steps=num_steps, dt=dt) 
         for s in sigma_values]

# Plot
total_time = num_steps * dt
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i, (path, sigma) in enumerate(zip(paths, sigma_values)):
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 
            label=f'Process {i+1} (σ={sigma}, μ={mu})', 
            color=['blue', 'red'][i], alpha=0.8)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title(f'3D Wiener Processes (T={total_time:.2f}, μ={mu})') 
# The total time is 10 seconds, not 1000. 1000 steps, but each step is only 0.01 time units, so the total duration is 10 time units.
ax.legend()
ax.grid(True)
plt.savefig("week2-2-1.svg", format="svg", bbox_inches="tight")
plt.close()