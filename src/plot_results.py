"""
plot_results.py

Load backtest result tables from data/results and produce:
  - equity_curves.png
  - annualized_return_bar.png
  - sharpe_ratio_bar.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results():
    """
    Load equity curves and performance summary from data/results.

    Expected files:
      - data/results/equity_curves.csv
      - data/results/performance_summary.csv

    The equity CSV is expected to have a datetime index column named 'month'.
    """
    results_dir = Path("data/results")
    equity_path = results_dir / "equity_curves.csv"
    perf_path = results_dir / "performance_summary.csv"

    # IMPORTANT: the index column is 'month', not 'date'
    equity_df = pd.read_csv(equity_path, parse_dates=["month"], index_col="month")
    stats_df = pd.read_csv(perf_path, index_col=0)

    return equity_df, stats_df, results_dir


def find_metric_column(stats_df: pd.DataFrame, keywords):
    """
    Find a column in stats_df whose lowercase name contains all keywords.
    This makes the script robust to names like 'annualized_return' or 'AnnualizedReturn'.
    """
    lower_map = {col.lower(): col for col in stats_df.columns}

    for lower_name, original_name in lower_map.items():
        if all(kw in lower_name for kw in keywords):
            return original_name

    raise ValueError(
        f"Could not find column containing keywords {keywords}. "
        f"Available columns are: {list(stats_df.columns)}"
    )


def plot_equity_curves(equity_df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot cumulative equity curves for all strategies.
    """
    plt.figure(figsize=(10, 6))
    for col in equity_df.columns:
        plt.plot(equity_df.index, equity_df[col], label=col)

    plt.xlabel("Time (month)")
    plt.ylabel("Equity (starting at 1)")
    plt.title("Equity curves – Buy & Hold vs Active strategies vs ML")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar_from_stats(
    stats_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Generic bar plot for a given metric in the performance summary.

    Parameters
    ----------
    stats_df : DataFrame
        Performance summary with one row per strategy.
    metric_col : str
        Column name to plot (ex: 'annualized_return', 'sharpe_ratio').
    title : str
        Title of the figure.
    ylabel : str
        Label of y-axis.
    out_path : Path
        Where to save the figure.
    """
    if metric_col not in stats_df.columns:
        raise ValueError(
            f"Column '{metric_col}' not found in performance summary "
            f"(available: {list(stats_df.columns)})"
        )

    plt.figure(figsize=(8, 5))
    values = stats_df[metric_col]
    plt.bar(values.index, values.values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    print(">>> plot_results.py started")

    equity_df, stats_df, results_dir = load_results()

    # 1) Equity curves
    equity_fig_path = results_dir / "equity_curves.png"
    plot_equity_curves(equity_df, equity_fig_path)

    # 2) Find correct column names in a robust way
    ann_ret_col = find_metric_column(stats_df, ["annualized", "return"])
    sharpe_col = find_metric_column(stats_df, ["sharpe"])

    # 3) Annualized returns bar chart
    ann_ret_fig_path = results_dir / "annualized_return_bar.png"
    plot_bar_from_stats(
        stats_df,
        metric_col=ann_ret_col,
        title="Annualized return by strategy",
        ylabel="Annualized return",
        out_path=ann_ret_fig_path,
    )

    # 4) Sharpe ratio bar chart
    sharpe_fig_path = results_dir / "sharpe_ratio_bar.png"
    plot_bar_from_stats(
        stats_df,
        metric_col=sharpe_col,
        title="Sharpe ratio by strategy",
        ylabel="Sharpe ratio",
        out_path=sharpe_fig_path,
    )

    print(f"Saved equity curves figure to: {equity_fig_path}")
    print(f"Saved annualized return bar to: {ann_ret_fig_path}")
    print(f"Saved Sharpe ratio bar to: {sharpe_fig_path}")
    print(">>> plot_results.py finished")


if __name__ == "__main__":
    main()
