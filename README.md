# Binomial as Poisson Estimation Error Analysis

This project analyzes the approximation of the binomial distribution by the Poisson distribution. Using several distance measures, such as Kullback-Leibler (KL) divergence, Jensen-Shannon divergence, and Wasserstein distance, the project quantifies the error between these distributions for varying parameters `n` (number of trials) and `p` (probability of success).

### Hypothesis

In the approximation of the binomial distribution by the Poisson distribution, the underlying assumption is that no two events occur simultaneously within a given time unit. This assumption is valid when the success probability \( p \) is small, as the likelihood of event collisions (i.e., multiple events occurring in the same time unit) remains negligible. However, as \( p \) increases, the probability of collisions grows, and this approximation becomes increasingly inaccurate. Notably, this increase in event collisions is primarily influenced by \( p \) and remains independent of the number of trials \( n \). Consequently, the Poisson approximation is effective only for small values of \( p \), and its accuracy diminishes as \( p \) approaches larger values, regardless of the size of \( n \).


## Project Structure

- **`calculate_differences.py`**: This script calculates the differences between binomial and Poisson distributions using various statistical metrics. It stores the results in a CSV file (`distribution_differences.csv`).
- **`evaluate_n_by_p.py`**: This script evaluates the results stored in the CSV file and generates plots of the divergence and distance metrics as a function of `n` for fixed values of `p`.
- **`evaluate_p_by_n.py`**: This script generates plots of divergence metrics as a function of `p` for fixed values of `n`.

## Key Features

- **Distance Metrics**:
  - **KL Divergence**: A measure of how one probability distribution diverges from a second, expected probability distribution.
  - **Jensen-Shannon Divergence**: A symmetrized and smoothed version of the KL divergence.
  - **Wasserstein Distance**: Measures the distance between two probability distributions in terms of the "mass" that must be moved to transform one distribution into the other.

- **CSV Output**: The script writes the comparison metrics for various `n` and `p` values to a CSV file, which can be further analyzed and plotted.
  
- **Plotting**: The project includes scripts to visualize the results of the analysis, showing how the approximation error behaves with changes in the parameters `n` and `p`.

## Usage

1. Run the `calculate_differences.py` script to compute the statistical divergences and distances between binomial and Poisson distributions:
    ```bash
    python calculate_differences.py
    ```

2. Use `evaluate_n_by_p.py` to generate plots of the results for different values of `n`:
    ```bash
    python evaluate_n_by_p.py
    ```

3. Use `evaluate_p_by_n.py` to generate plots of the results for different values of `p`:
    ```bash
    python evaluate_p_by_n.py
    ```

## Output

The primary output of the project includes:
- A CSV file (`distribution_differences.csv`) containing the results of the distribution comparison.
- Several visualizations showing the behavior of the divergence metrics across different values of `n` and `p`.

### Conclusion

The results of this analysis support the hypothesis that the Poisson approximation of the binomial distribution becomes increasingly inaccurate as the probability of success \( p \) increases. While the Poisson assumption that no two events occur simultaneously holds for small values of \( p \), the likelihood of event collisions grows as \( p \) increases. This leads to a greater divergence between the binomial and Poisson distributions, as shown by the rising values of KL divergence, Jensen-Shannon divergence, and Wasserstein distance.

Importantly, the number of collisions—and thus the inaccuracy of the Poisson approximation—depends largely on \( p \) and remains independent of the number of trials \( n \). This confirms that for large \( p \) values, the Poisson approximation fails, regardless of \( n \), whereas for small \( p \), the approximation remains effective. These findings highlight the limitations of the Poisson approximation and emphasize the need to consider \( p \) carefully when using this method to approximate binomial behavior.


## Dependencies

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Pandas

You can install the dependencies with:
```bash
pip install numpy scipy matplotlib pandas
