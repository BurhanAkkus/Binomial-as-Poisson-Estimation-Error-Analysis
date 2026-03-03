"""
Submission-ready analysis for when Binomial(N, p) approximates Poisson(lambda).

This single script replaces the previous multi-script workflow:
  1) Hypothesis sweep:
     - fixed lambda, vary N (p = lambda / N)
     - fixed p, vary N (lambda = N * p)
  2) Single fixed-p study
  3) Multi-p fixed-p study with one combined JS-only plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binom, poisson


TAIL_MASS = 1e-12
EPS = 1e-15

DEFAULT_N_VALUES = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
DEFAULT_LAMBDA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_P_VALUES = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 0.90, 0.99]


def normalize(arr: np.ndarray) -> np.ndarray:
    total = float(np.sum(arr))
    if total == 0.0:
        return np.zeros_like(arr, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64) / total


def pmf_with_tail(n: int, p: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    k_poisson = int(max(1, poisson.ppf(1.0 - TAIL_MASS, lam)))
    k_max = max(k_poisson, n)
    x = np.arange(k_max + 1, dtype=np.int64)

    pmf_bin = binom.pmf(x, n, p)
    pmf_poi = poisson.pmf(x, lam)

    pmf_bin = np.append(pmf_bin, max(0.0, 1.0 - float(np.sum(pmf_bin))))
    pmf_poi = np.append(pmf_poi, max(0.0, 1.0 - float(np.sum(pmf_poi))))

    return normalize(pmf_bin), normalize(pmf_poi)


def metrics(n: int, p: float) -> dict[str, float]:
    lam = float(n * p)
    pmf_bin, pmf_poi = pmf_with_tail(n=n, p=p, lam=lam)

    tv = float(0.5 * np.sum(np.abs(pmf_bin - pmf_poi)))
    pmf_bin_s = normalize(np.clip(pmf_bin, EPS, None))
    pmf_poi_s = normalize(np.clip(pmf_poi, EPS, None))
    js = float(jensenshannon(pmf_bin_s, pmf_poi_s))

    collision_prob = float(1.0 - np.exp(-p) * (1.0 + p))
    le_cam_tv_bound = float(min(1.0, 2.0 * n * (p**2)))

    return {
        "n": int(n),
        "p": float(p),
        "lambda": lam,
        "tv_distance": tv,
        "js_distance": js,
        "collision_prob_per_slot": collision_prob,
        "le_cam_tv_bound": le_cam_tv_bound,
    }


def fixed_lambda_dataframe(lambda_values: list[float], n_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for lam in lambda_values:
        for n in n_values:
            p = lam / n
            if 0.0 < p < 1.0:
                row = metrics(n=n, p=p)
                row["experiment"] = "fixed_lambda"
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["lambda", "n"]).reset_index(drop=True)


def fixed_p_dataframe(p_values: list[float], n_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for p in p_values:
        if not (0.0 < p < 1.0):
            raise ValueError(f"All p values must be in (0, 1). Invalid p={p}")
        for n in n_values:
            row = metrics(n=n, p=p)
            row["experiment"] = "fixed_p"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["p", "n"]).reset_index(drop=True)


def plot_fixed_lambda_tv(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for lam in sorted(df["lambda"].unique()):
        sub = df[np.isclose(df["lambda"], lam)].sort_values("n")
        ax.plot(sub["n"], sub["tv_distance"], marker="o", label=f"lambda={lam:g}")
    ax.set_xscale("log")
    ax.set_xlabel("N (number of zones)")
    ax.set_ylabel("TV distance")
    ax.set_title("Fixed lambda: TV distance vs N (p=lambda/N)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fixed_p_metric(df: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in sorted(df["p"].unique()):
        sub = df[np.isclose(df["p"], p)].sort_values("n")
        ax.plot(sub["n"], sub[metric], marker="o", label=f"p={p:g}")
    ax.set_xscale("log")
    ax.set_xlabel("N (number of zones)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Fixed p: {ylabel} vs N (lambda=N*p)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_fixed_p_summary(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for p in sorted(df["p"].unique()):
        sub = df[df["p"] == p].sort_values("n")
        rows.append(
            {
                "p": float(p),
                "collision_prob_per_slot": float(sub["collision_prob_per_slot"].iloc[0]),
                "tv_min": float(sub["tv_distance"].min()),
                "tv_max": float(sub["tv_distance"].max()),
                "tv_spread": float(sub["tv_distance"].max() - sub["tv_distance"].min()),
                "js_min": float(sub["js_distance"].min()),
                "js_max": float(sub["js_distance"].max()),
                "js_spread": float(sub["js_distance"].max() - sub["js_distance"].min()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    summary.to_csv(out_path, index=False)
    return summary


def cmd_hypothesis(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_lambda = fixed_lambda_dataframe(args.lambda_values, args.n_values)
    df_p = fixed_p_dataframe(args.p_values, args.n_values)

    lambda_csv = out_dir / "hypothesis_fixed_lambda.csv"
    p_csv = out_dir / "hypothesis_fixed_p.csv"
    lambda_tv_png = out_dir / "hypothesis_fixed_lambda_tv.png"
    p_tv_png = out_dir / "hypothesis_fixed_p_tv.png"
    p_js_png = out_dir / "hypothesis_fixed_p_js.png"
    p_summary_csv = out_dir / "hypothesis_fixed_p_summary.csv"

    df_lambda.to_csv(lambda_csv, index=False)
    df_p.to_csv(p_csv, index=False)
    plot_fixed_lambda_tv(df_lambda, lambda_tv_png)
    plot_fixed_p_metric(df_p, "tv_distance", "TV distance", p_tv_png)
    plot_fixed_p_metric(df_p, "js_distance", "JS distance", p_js_png)
    summary = write_fixed_p_summary(df_p, p_summary_csv)

    print("\n=== Fixed lambda summary (TV at smallest vs largest N) ===")
    for lam in sorted(df_lambda["lambda"].unique()):
        sub = df_lambda[df_lambda["lambda"] == lam].sort_values("n")
        n_small = int(sub.iloc[0]["n"])
        n_large = int(sub.iloc[-1]["n"])
        tv_small = float(sub.iloc[0]["tv_distance"])
        tv_large = float(sub.iloc[-1]["tv_distance"])
        ratio = tv_small / tv_large if tv_large > 0 else np.nan
        print(
            f"lambda={lam:g}: TV(N={n_small})={tv_small:.6f}, "
            f"TV(N={n_large})={tv_large:.6f}, ratio={ratio:.2f}x"
        )

    print("\n=== Fixed p summary ===")
    print(summary.to_string(index=False))

    print(f"\nWrote: {lambda_csv}")
    print(f"Wrote: {p_csv}")
    print(f"Wrote: {lambda_tv_png}")
    print(f"Wrote: {p_tv_png}")
    print(f"Wrote: {p_js_png}")
    print(f"Wrote: {p_summary_csv}")


def cmd_fixed_p(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fixed_p_dataframe([args.p], args.n_values)
    csv_path = out_dir / "fixed_p_vary_n_results.csv"
    tv_plot = out_dir / "fixed_p_vary_n_tv.png"
    js_plot = out_dir / "fixed_p_vary_n_js.png"

    df.to_csv(csv_path, index=False)
    plot_fixed_p_metric(df, "tv_distance", "TV distance", tv_plot)
    plot_fixed_p_metric(df, "js_distance", "JS distance", js_plot)

    print(df.to_string(index=False))
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {tv_plot}")
    print(f"Wrote: {js_plot}")


def cmd_fixed_p_grid(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fixed_p_dataframe(args.p_values, args.n_values)
    all_csv = out_dir / "fixed_p_grid_data.csv"
    js_plot = out_dir / "fixed_p_grid_js_combined.png"
    summary_csv = out_dir / "fixed_p_grid_summary.csv"

    df.to_csv(all_csv, index=False)
    plot_fixed_p_metric(df, "js_distance", "JS distance", js_plot)
    summary = write_fixed_p_summary(df, summary_csv)

    print(summary.to_string(index=False))
    print(f"\nWrote: {all_csv}")
    print(f"Wrote: {js_plot}")
    print(f"Wrote: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified analysis of Binomial approximating Poisson."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    hypothesis = sub.add_parser(
        "hypothesis",
        help="Run fixed-lambda and fixed-p sweeps and generate CSV/plots.",
    )
    hypothesis.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help="N values to evaluate.",
    )
    hypothesis.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDA_VALUES,
        help="Lambda values for the fixed-lambda sweep.",
    )
    hypothesis.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=DEFAULT_P_VALUES,
        help="p values for the fixed-p sweep.",
    )
    hypothesis.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hypothesis"),
        help="Output directory.",
    )
    hypothesis.set_defaults(func=cmd_hypothesis)

    fixed_p = sub.add_parser(
        "fixed-p",
        help="Run one fixed-p sweep over N and generate CSV + TV/JS plots.",
    )
    fixed_p.add_argument("--p", type=float, required=True, help="Fixed p in (0,1).")
    fixed_p.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help="N values to evaluate.",
    )
    fixed_p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fixed_p_single"),
        help="Output directory.",
    )
    fixed_p.set_defaults(func=cmd_fixed_p)

    fixed_p_grid = sub.add_parser(
        "fixed-p-grid",
        help="Run multiple fixed-p sweeps and generate one combined JS plot.",
    )
    fixed_p_grid.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=DEFAULT_P_VALUES,
        help="List of fixed p values in (0,1).",
    )
    fixed_p_grid.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=DEFAULT_N_VALUES,
        help="N values to evaluate.",
    )
    fixed_p_grid.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fixed_p_grid"),
        help="Output directory.",
    )
    fixed_p_grid.set_defaults(func=cmd_fixed_p_grid)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
