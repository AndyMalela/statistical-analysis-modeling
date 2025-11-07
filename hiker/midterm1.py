import pandas as pd

# Load the CSV file
df = pd.read_csv("hiker_randomwalk_runs_seed42_n4000.csv")

# Group by hiker and compute averages
avg_stats = df.groupby("hiker")[["delta_x", "delta_y", "steps"]].mean().reset_index()

# Print the result
print(avg_stats)

# Optionally, save the results to a new CSV
avg_stats.to_csv("hiker_average_stats.csv", index=False)
