"""
main.py — Entry point for the Finsight project.

Run with:
    python -m src.main
"""

from datetime import datetime

from src import (
    data_download,
    features,
    labeling,
    enrich_features,
    backtesting,
    save_results,
    plot_results,
)


def run_finsight_pipeline():
    print("=" * 80)
    print("Finsight – Market Regime Adaptive Strategies")
    print(f"Run started at {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)

    # 1. Optional: download raw data
    if hasattr(data_download, "main"):
        print("\n[1/7] Downloading raw data...")
        data_download.main()
    else:
        print("\n[1/7] Download step skipped.")

    # 2. Compute daily features
    print("\n[2/7] Computing daily features...")
    features.main()

    # 3. Label future best strategy
    print("\n[3/7] Labeling data...")
    labeling.main()

    # 4. Enrich with macro variables
    print("\n[4/7] Enriching features...")
    enrich_features.main()

    # 5. Backtest
    print("\n[5/7] Running backtest...")
    backtesting.run_backtest(return_details=False)

    # 6. Save results
    print("\n[6/7] Saving results...")
    save_results.main()

    # 7. Plot results
    print("\n[7/7] Generating plots...")
    plot_results.main()

    print("\nPipeline finished successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_finsight_pipeline()
