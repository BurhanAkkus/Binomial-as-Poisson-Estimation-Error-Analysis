# When Binomial Approximates Poisson: Error Analysis

This repository studies when `Binomial(N, p)` can approximate `Poisson(lambda)` with `lambda = Np`, and when that approximation breaks.

Target and approximator:
- Target distribution: `Poisson(lambda = N * p)`
- Approximating distribution: `Binomial(N, p)`

using:
- Total variation (TV) distance
- Jensen-Shannon (JS) distance
- Collision probability per slot under `Poisson(p)`: `P(X >= 2) = 1 - e^{-p}(1+p)`
- Le Cam upper bound for TV: `2 * N * p^2`

## Hypothesis Being Tested

If time is split into `N` slots and each slot is treated as Bernoulli (0 or 1 event), approximation quality is expected to worsen as `p` gets larger because within-slot multi-event collisions become more likely.

This project tests both:
- fixed `lambda`, varying `N` (so `p = lambda/N`)
- fixed `p`, varying `N` (so `lambda = N*p`)

## Project Structure

- `poisson_binomial_analysis.py`: unified script for all experiments and plots.

## Setup

```bash
python -m pip install numpy scipy pandas matplotlib
```

## Usage

### 1) Full hypothesis sweep (recommended for submission)

Runs both experiments (fixed-lambda and fixed-p), writes CSV tables and TV/JS plots.

```bash
python poisson_binomial_analysis.py hypothesis
```

Outputs go to `outputs/hypothesis/` by default.

### 2) Single fixed-p experiment

```bash
python poisson_binomial_analysis.py fixed-p --p 0.2
```

Outputs go to `outputs/fixed_p_single/` by default.

### 3) Multi-p fixed-p experiment with one combined JS plot

```bash
python poisson_binomial_analysis.py fixed-p-grid --p-values 0.01 0.03 0.05 0.1 0.2 0.3 0.5 0.8 0.9 0.99
```

Outputs go to `outputs/fixed_p_grid/` by default, including:
- `fixed_p_grid_data.csv`
- `fixed_p_grid_summary.csv`
- `fixed_p_grid_js_combined.png`

## Customization

- Set custom `N` values:
  ```bash
  python poisson_binomial_analysis.py hypothesis --n-values 5 10 20 50 100 200 500 1000
  ```
- Set custom output directory:
  ```bash
  python poisson_binomial_analysis.py hypothesis --output-dir my_results
  ```

## Notes for Interpretation

- Large `p` strongly increases approximation error and collision probability.
- At fixed `p`, error can still vary with `N` (not perfectly `N`-independent).
- At fixed `lambda`, increasing `N` decreases `p=lambda/N`, which usually improves the approximation.
