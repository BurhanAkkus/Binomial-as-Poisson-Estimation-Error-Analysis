import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

# Lists to store results
distributions = []
differences = []

# Parameters
n_array = [10, 100, 1000, 10000, 100000, 1000000]
p_array = np.arange(0, 1, 0.01)

# Iterate over n and p
for n in n_array:
    for p in p_array:
        # Skip edge cases for p = 0 and p = 1
        if p == 0 or p == 1:
            continue

        lambda_poisson = n * p  # Lambda for Poisson

        # Generate integer k values (since PMF is discrete)
        k_values = np.arange(0,  n)  # Integer range of k values

        # Compute PMF values (not the frozen distribution object)
        binom_pmf = binom.pmf(k_values, n, p)
        poisson_pmf = poisson.pmf(k_values, lambda_poisson)

        # Check if PMFs sum to non-zero values before normalizing
        binom_sum = np.sum(binom_pmf)
        poisson_sum = np.sum(poisson_pmf)

        if binom_sum > 0 and poisson_sum > 0:
            # Normalize PMFs
            binom_pmf /= binom_sum
            poisson_pmf /= poisson_sum

            # Store distributions
            distributions.append((binom_pmf, poisson_pmf))

            # KL-Divergence (Entropy)
            kl_div = entropy(binom_pmf, poisson_pmf)

            # Jensen-Shannon Divergence
            js_div = jensenshannon(binom_pmf, poisson_pmf)

            # Wasserstein Distance
            wasser_dist = wasserstein_distance(k_values, k_values, binom_pmf, poisson_pmf)

            # Store differences
            differences.append((kl_div, js_div, wasser_dist))

            # Optionally, print the results for every (n, p)
            print(f"n = {n}, p = {p:.2f} | KL Divergence: {kl_div:.4f} | Jensen-Shannon: {js_div:.4f} | Wasserstein: {wasser_dist:.4f}")
        else:
            # Handle case where the PMF sum is zero (skip calculation or log the issue)
            print(f"Skipping (n = {n}, p = {p:.2f}) due to zero PMF sum.")

# Optional: Plot the last distribution
plt.plot(k_values, binom_pmf, 'b-', label="Binomial PMF")
plt.plot(k_values, poisson_pmf, 'r-', label="Poisson PMF")
plt.xlabel("k")
plt.ylabel("Probability")
plt.legend()
plt.title(f"Binomial vs Poisson PMF (n = {n}, p = {p:.2f})")
plt.show()
