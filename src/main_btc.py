# src/main_btc.py
"""
main_btc.py — entry point for the BTC extension of the Finsight project.

It:
1. Backs up current SP500 raw / processed / results files
2. Swaps SPY_prices.csv with BTC_prices.csv for this run only
3. Runs the standard pipeline (features, labeling, enrich, backtest, save, plot)
4. Copies the BTC outputs into data/processed/btc and data/results/btc
5. Restores the original SP500 files

Run with:
    python -m src.data_download_btc   # first time or to refresh BTC prices
    python -m src.main_btc
"""

from datetime import datetime
from pathlib import Path
import shutil

from src import (
    config,
    config_btc,
    features,
    labeling,
    enrich_features,
    backtesting,
    save_results,
    plot_results,
)


def _backup_file(path: Path, backup_dir: Path) -> None:
    if path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)


def _restore_file(path: Path, backup_dir: Path) -> None:
    backup_path = backup_dir / path.name
    if backup_path.exists():
        shutil.copy2(backup_path, path)


def run_btc_pipeline() -> None:
    print("=" * 80)
    print("Finsight – BTC Market Regime Adaptive Strategies")
    print(f"Run started at {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)

    raw_dir = config.RAW_DATA_DIR
    processed_dir = config.PROCESSED_DATA_DIR
    results_dir = config.RESULTS_DIR

    # Paths for SP500 files
    spy_price_file = raw_dir / "SPY_prices.csv"

    # BTC raw prices file
    btc_price_file = raw_dir / "BTC_prices.csv"

    # BTC archive folders
    btc_processed_dir = processed_dir / "btc"
    btc_results_dir = results_dir / "btc"
    btc_processed_dir.mkdir(parents=True, exist_ok=True)
    btc_results_dir.mkdir(parents=True, exist_ok=True)

    # Temporary backup dir for SP500 files
    backup_dir = results_dir / "_tmp_backup_spy_for_btc"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 0. Safety checks
    # ---------------------------------------------------------------------
    if not btc_price_file.exists():
        raise FileNotFoundError(
            f"{btc_price_file} not found. "
            "Run `python -m src.data_download_btc` first to download BTC data."
        )

    # ---------------------------------------------------------------------
    # 1. Backup current SP500 files
    # ---------------------------------------------------------------------
    print("\n[0] Backing up current SP500 files...")

    # Raw prices
    _backup_file(spy_price_file, backup_dir)

    # Processed data (SPY)
    for name in ["features.csv", "labeled_data.csv", "monthly_regimes.csv"]:
        _backup_file(processed_dir / name, backup_dir)

    # Results (SPY)
    for name in [
        "equity_curves.csv",
        "monthly_returns.csv",
        "performance_summary.csv",
        "equity_curves.png",
        "annualized_return_bar.png",
        "sharpe_ratio_bar.png",
    ]:
        _backup_file(results_dir / name, backup_dir)

    # ---------------------------------------------------------------------
    # 2. Swap SPY raw prices with BTC prices for this run only
    # ---------------------------------------------------------------------
    print("\n[1] Swapping SP500 raw prices with BTC prices for this run...")
    shutil.copy2(btc_price_file, spy_price_file)

    # Temporarily change the ticker in config (for logs only)
    original_ticker = getattr(config, "TICKER", "SPY")
    config.TICKER = config_btc.TICKER

    # ---------------------------------------------------------------------
    # 3. Run the standard pipeline (now working on BTC prices)
    # ---------------------------------------------------------------------
    print("\n[2/7] Computing BTC features...")
    features.main()

    print("\n[3/7] Labeling BTC data...")
    labeling.main()

    print("\n[4/7] Enriching BTC features...")
    enrich_features.main()

    print("\n[5/7] Running BTC backtest...")
    backtesting.run_backtest(return_details=False)

    print("\n[6/7] Saving BTC results...")
    save_results.main()

    print("\n[7/7] Generating BTC plots...")
    plot_results.main()

    # ---------------------------------------------------------------------
    # 4. Archive BTC outputs in dedicated folders
    # ---------------------------------------------------------------------
    print("\nArchiving BTC outputs into data/processed/btc and data/results/btc ...")

    # Processed data
    for src_name, dst_name in [
        ("features.csv", "features_btc.csv"),
        ("labeled_data.csv", "labeled_data_btc.csv"),
        ("monthly_regimes.csv", "monthly_regimes_btc.csv"),
    ]:
        src_path = processed_dir / src_name
        if src_path.exists():
            shutil.copy2(src_path, btc_processed_dir / dst_name)

    # Results
    for src_name, dst_name in [
        ("equity_curves.csv", "equity_curves_btc.csv"),
        ("monthly_returns.csv", "monthly_returns_btc.csv"),
        ("performance_summary.csv", "performance_summary_btc.csv"),
        ("equity_curves.png", "equity_curves_btc.png"),
        ("annualized_return_bar.png", "annualized_return_bar_btc.png"),
        ("sharpe_ratio_bar.png", "sharpe_ratio_bar_btc.png"),
    ]:
        src_path = results_dir / src_name
        if src_path.exists():
            shutil.copy2(src_path, btc_results_dir / dst_name)

    # ---------------------------------------------------------------------
    # 5. Restore original SP500 files
    # ---------------------------------------------------------------------
    print("\nRestoring original SP500 files...")

    # Restore ticker
    config.TICKER = original_ticker

    # Restore raw prices
    _restore_file(spy_price_file, backup_dir)

    # Restore processed data
    for name in ["features.csv", "labeled_data.csv", "monthly_regimes.csv"]:
        _restore_file(processed_dir / name, backup_dir)

    # Restore results
    for name in [
        "equity_curves.csv",
        "monthly_returns.csv",
        "performance_summary.csv",
        "equity_curves.png",
        "annualized_return_bar.png",
        "sharpe_ratio_bar.png",
    ]:
        _restore_file(results_dir / name, backup_dir)

    # Clean temporary backup
    shutil.rmtree(backup_dir, ignore_errors=True)

    print("\nBTC pipeline finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_btc_pipeline()
