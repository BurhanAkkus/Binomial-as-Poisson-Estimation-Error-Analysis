# Binomial as Poisson Estimation Error Analysis

This project analyzes the approximation of the binomial distribution by the Poisson distribution. Using several distance measures, such as Kullback-Leibler (KL) divergence, Jensen-Shannon divergence, and Wasserstein distance, the project quantifies the error between these distributions for varying parameters `n` (number of trials) and `p` (probability of success).

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

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Pandas

You can install the dependencies with:
```bash
pip install numpy scipy matplotlib pandas
