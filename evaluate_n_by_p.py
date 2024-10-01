
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('distribution_differences.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Set up plots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot KL Divergence by p
for p in df['p'].unique():
    subset = df[df['p'] == p]
    axes[0].plot(subset['n'], subset['KL Divergence'], label=f'p = {p:.2f}')

axes[0].set_title('KL Divergence by n for different p')
axes[0].set_ylabel('KL Divergence')
#axes[0].legend()
axes[0].grid()

# Plot Jensen-Shannon Divergence by p
for p in df['p'].unique():
    subset = df[df['p'] == p]
    axes[1].plot(subset['n'], subset['Jensen-Shannon Divergence'], label=f'p = {p:.2f}')

axes[1].set_title('Jensen-Shannon Divergence by p for different n')
axes[1].set_ylabel('Jensen-Shannon Divergence')
#axes[1].legend()
axes[1].grid()

# Plot Wasserstein Distance by n
for p in df['p'].unique():
    subset = df[df['p'] == p]
    axes[2].plot(subset['n'], subset['Wasserstein Distance'], label=f'p = {p:.2f}')

axes[2].set_title('Wasserstein Distance by p for different n')
axes[2].set_xlabel('p')
axes[2].set_ylabel('Wasserstein Distance')
#axes[2].legend()
axes[2].grid()

# Show plots
plt.tight_layout()
plt.show()
