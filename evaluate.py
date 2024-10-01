import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('distribution_differences.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Set up plots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot KL Divergence by n
for n in df['n'].unique():
    subset = df[df['n'] == n]
    axes[0].plot(subset['p'], subset['KL Divergence'], label=f'n = {n}')

axes[0].set_title('KL Divergence by p for different n')
axes[0].set_ylabel('KL Divergence')
axes[0].legend()
axes[0].grid()

# Plot Jensen-Shannon Divergence by n
for n in df['n'].unique():
    subset = df[df['n'] == n]
    axes[1].plot(subset['p'], subset['Jensen-Shannon Divergence'], label=f'n = {n}')

axes[1].set_title('Jensen-Shannon Divergence by p for different n')
axes[1].set_ylabel('Jensen-Shannon Divergence')
axes[1].legend()
axes[1].grid()

# Plot Wasserstein Distance by n
for n in df['n'].unique():
    subset = df[df['n'] == n]
    axes[2].plot(subset['p'], subset['Wasserstein Distance'], label=f'n = {n}')

axes[2].set_title('Wasserstein Distance by p for different n')
axes[2].set_xlabel('p')
axes[2].set_ylabel('Wasserstein Distance')
axes[2].legend()
axes[2].grid()

# Show plots
plt.tight_layout()
plt.show()
