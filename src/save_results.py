"""
save_results.py

Run the monthly backtest and export:
  - equity_curves.csv
  - monthly_returns.csv
  - performance_summary.csv

This script relies on the new backtesting API:
  - run_backtest(return_details=True) returns a dict with:
      * 'performance_summary' : DataFrame
      * 'equity_curves'       : DataFrame
      * 'monthly_returns'     : DataFrame
      * 'ml_decisions'        : Series
      * 'monthly'             : DataFrame (full monthly frame)
"""

from pathlib import Path

from src.backtesting import run_backtest


def main() -> None:
    """Run the backtest and save all relevant result tables to CSV."""
    print(">>> save_results.py started")

    # 1) Ensure output directory exists
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Run backtest with detailed outputs
    results = run_backtest(return_details=True)

    performance_summary = results["performance_summary"]
    equity_curves = results["equity_curves"]
    monthly_returns = results["monthly_returns"]

    # 3) Build file paths
    equity_path = out_dir / "equity_curves.csv"
    monthly_returns_path = out_dir / "monthly_returns.csv"
    perf_path = out_dir / "performance_summary.csv"

    # 4) Save to CSV
    equity_curves.to_csv(equity_path)
    monthly_returns.to_csv(monthly_returns_path)
    performance_summary.to_csv(perf_path)

    print(f"Saved equity curves to: {equity_path}")
    print(f"Saved monthly returns to: {monthly_returns_path}")
    print(f"Saved performance summary to: {perf_path}")
    print(">>> save_results.py finished")


if __name__ == "__main__":
    main()
