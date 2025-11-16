# src/plot_results.py

"""
Generate plots for the Finsight project:
- Equity curves of all strategies
- Bar chart of annualized returns
- Bar chart of Sharpe ratios

The script reads the CSV files created by save_results.py
from data/results/.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src import config


def load_results():
    """
    Load equity curves and performance summary from data/results/.
    """
    results_dir = config.DATA_DIR / "results"

    equity_path = results_dir / "equity_curves.csv"
    stats_path = results_dir / "performance_summary.csv"

    equity_df = pd.read_csv(equity_path, parse_dates=["date"], index_col="date")
    stats_df = pd.read_csv(stats_path, index_col="strategy")

    return equity_df, stats_df, results_dir


def plot_equity_curves(equity_df: pd.DataFrame, results_dir: Path) -> None:
    """
    Plot equity curves (BH, Momentum, Mean Reversion, ML Adaptive, Oracle)
    and save the figure to data/results/.
    """
    plt.figure(figsize=(10, 6))

    for col in equity_df.columns:
        plt.plot(equity_df.index, equity_df[col], label=col)

    plt.title("Equity curves of all strategies")
    plt.xlabel("Date")
    plt.ylabel("Equity (starting at 1.0)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    out_path = results_dir / "equity_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved equity curves plot to: {out_path}")


def plot_bar_metric(stats_df: pd.DataFrame, metric: str, ylabel: str, filename: str, results_dir: Path) -> None:
    """
    Generic function to plot a bar chart for a given performance metric
    (e.g., annualized_return, sharpe_ratio) across strategies.
    """
    if metric not in stats_df.columns:
        raise ValueError(f"Metric '{metric}' not found in performance_summary.csv columns: {stats_df.columns}")

    plt.figure(figsize=(8, 5))

    # Sort strategies by metric value (optional, easier to read)
    stats_sorted = stats_df.sort_values(metric, ascending=False)

    plt.bar(stats_sorted.index, stats_sorted[metric])
    plt.title(metric.replace("_", " ").capitalize())
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    out_path = results_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved bar plot for {metric} to: {out_path}")


def main():
    print(">>> plot_results.py started")

    equity_df, stats_df, results_dir = load_results()

    print("Equity curves shape:", equity_df.shape)
    print("Performance summary:")
    print(stats_df)

    # 1) Equity curves
    plot_equity_curves(equity_df, results_dir)

    # 2) Annualized return bar chart
    plot_bar_metric(
        stats_df,
        metric="annualized_return",
        ylabel="Annualized return",
        filename="annualized_return_bar.png",
        results_dir=results_dir,
    )

    # 3) Sharpe ratio bar chart
    plot_bar_metric(
        stats_df,
        metric="sharpe_ratio",
        ylabel="Sharpe ratio (rf = 0)",
        filename="sharpe_ratio_bar.png",
        results_dir=results_dir,
    )

    print(">>> plot_results.py finished")


if __name__ == "__main__":
    main()
