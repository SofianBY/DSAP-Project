# src/save_results.py

from pathlib import Path
from typing import Dict

import pandas as pd

from src import config
from src.backtesting import (
    load_daily_labeled_data,
    build_monthly_frame,
    build_features_and_target,
    rolling_multiclass_backtest,
    compute_performance_stats,
)


def ensure_results_dir() -> Path:
    """
    Ensure that data/results/ exists and return its Path.
    """
    results_dir = config.DATA_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def build_oracle_returns(
    bh_rets: pd.Series,
    mom_rets: pd.Series,
    mr_rets: pd.Series,
) -> pd.Series:
    """
    Build the ex-post "Oracle" return series:
    each month, pick the best realized return among the 3 strategies.
    """
    df = pd.concat(
        [bh_rets.rename("bh"), mom_rets.rename("mom"), mr_rets.rename("mr")],
        axis=1,
    )
    best = df.max(axis=1)
    best.name = "oracle_return"
    return best


def save_equity_curves(
    results: Dict[str, pd.Series],
    oracle_equity: pd.Series,
    results_dir: Path,
) -> None:
    """
    Save equity curves (BH, Momentum, Mean Reversion, ML Adaptive, Oracle)
    to a CSV file in data/results/.
    """
    equity_df = pd.concat(
        [
            results["bh_equity"].rename("BH"),
            results["mom_equity"].rename("Momentum"),
            results["mr_equity"].rename("Mean Reversion"),
            results["ml_equity"].rename("ML Adaptive"),
            oracle_equity.rename("Oracle"),
        ],
        axis=1,
    )

    out_path = results_dir / "equity_curves.csv"
    equity_df.to_csv(out_path, index_label="date")
    print(f"Saved equity curves to: {out_path}")


def save_monthly_returns(
    bh_rets: pd.Series,
    mom_rets: pd.Series,
    mr_rets: pd.Series,
    ml_rets: pd.Series,
    oracle_rets: pd.Series,
    results_dir: Path,
) -> None:
    """
    Save monthly returns of the 4 strategies + Oracle
    to a CSV file in data/results/.
    """
    returns_df = pd.concat(
        [
            bh_rets.rename("BH"),
            mom_rets.rename("Momentum"),
            mr_rets.rename("Mean Reversion"),
            ml_rets.rename("ML Adaptive"),
            oracle_rets.rename("Oracle"),
        ],
        axis=1,
    )

    out_path = results_dir / "monthly_returns.csv"
    returns_df.to_csv(out_path, index_label="date")
    print(f"Saved monthly returns to: {out_path}")


def save_performance_summary(
    bh_rets: pd.Series,
    mom_rets: pd.Series,
    mr_rets: pd.Series,
    ml_rets: pd.Series,
    oracle_rets: pd.Series,
    results_dir: Path,
) -> None:
    """
    Compute and save performance statistics for all strategies.
    """
    stats = {
        "Buy & Hold": compute_performance_stats(bh_rets),
        "Momentum": compute_performance_stats(mom_rets),
        "Mean Reversion": compute_performance_stats(mr_rets),
        "ML Adaptive": compute_performance_stats(ml_rets),
        "Oracle": compute_performance_stats(oracle_rets),
    }

    stats_df = pd.DataFrame(stats).T
    stats_df.index.name = "strategy"

    out_path = results_dir / "performance_summary.csv"
    stats_df.to_csv(out_path)
    print(f"Saved performance summary to: {out_path}")


def main():
    print(">>> save_results.py started")

    # 1) Ensure results directory exists
    results_dir = ensure_results_dir()

    # 2) Reload the daily labeled data
    df_daily = load_daily_labeled_data()
    print("Daily labeled data shape:", df_daily.shape)

    # 3) Aggregate to monthly frame and build strategy returns
    monthly = build_monthly_frame(df_daily)
    print("Monthly frame shape:", monthly.shape)

    # 4) Build features X and multi-class target y (3 strategies)
    X, y = build_features_and_target(monthly)
    print("Monthly features shape:", X.shape)
    print("Multi-class target distribution (0=BH, 1=Mom, 2=MR):")
    print(y.value_counts())

    # 5) Run the rolling backtest
    results = rolling_multiclass_backtest(monthly, X, y, min_train_size=60)

    bh_rets = results["bh_rets"]
    mom_rets = results["mom_rets"]
    mr_rets = results["mr_rets"]
    ml_rets = results["ml_rets"]

    # 6) Build Oracle returns and equity
    oracle_rets = build_oracle_returns(bh_rets, mom_rets, mr_rets)
    oracle_equity = (1 + oracle_rets).cumprod()

    # 7) Save equity curves
    save_equity_curves(results, oracle_equity, results_dir)

    # 8) Save monthly returns
    save_monthly_returns(
        bh_rets, mom_rets, mr_rets, ml_rets, oracle_rets, results_dir
    )

    # 9) Save performance summary
    save_performance_summary(
        bh_rets, mom_rets, mr_rets, ml_rets, oracle_rets, results_dir
    )

    print(">>> save_results.py finished")


if __name__ == "__main__":
    main()
