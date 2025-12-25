"""
Main pipeline orchestration for the Finsight project.

- SP500 pipeline (monthly, raw SPY prices)
- BTC monthly pipeline (ONLY backtesting_btc.py, using already prepared files)
- BTC weekly pipeline (prepare_btc_weekly.py + modeling_btc_weekly.py + backtesting_btc_weekly.py)
- Final plotting pipeline (plot_results.py)
"""

import warnings

# On coupe les warnings moches (sklearn, pandas, etc.)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
#  SP500 PIPELINE
# -------------------------------------------------------------------
def run_sp500_pipeline() -> None:
    """
    Run the SP500 (SPY) backtesting pipeline.

    Uses src/backtesting.py::main()
    """
    print("\n==================== SP500 PIPELINE ====================\n")
    print("[SP500] Running backtesting.py ...\n")

    from .backtesting import main as backtesting_sp500_main

    backtesting_sp500_main()

    print("\n[SP500] Pipeline completed.\n")


# -------------------------------------------------------------------
#  BTC MONTHLY PIPELINE (BACKTEST SEULEMENT)
# -------------------------------------------------------------------
def run_btc_monthly_pipeline() -> None:
    """
    Run the BTC *monthly* pipeline.

    We only use: src/backtesting_btc.py::main()
    Assumes:
        data/processed/labeled_data_btc.csv
        data/raw/BTC_prices.csv
    already exist.
    """
    print("\n==================== BTC MONTHLY PIPELINE ====================\n")

    from .backtesting_btc import main as backtesting_btc_main

    print("[BTC monthly] Step 1/1 — Backtesting")
    backtesting_btc_main()

    print("\n[BTC monthly] Pipeline completed.\n")


# -------------------------------------------------------------------
#  BTC WEEKLY PIPELINE
# -------------------------------------------------------------------
def run_btc_weekly_pipeline() -> None:
    """
    Run the BTC *weekly* pipeline:
      1) prepare_btc_weekly.py         (in src/data_pipeline/)
      2) modeling_btc_weekly.py        (in src/)
      3) backtesting_btc_weekly.py     (in src/)
    """
    print("\n==================== BTC WEEKLY PIPELINE ====================\n")

    # ⚠️ Ici on importe depuis src.data_pipeline
    from .data_pipeline.prepare_btc_weekly import main as prepare_btc_weekly_main
    from .modeling_btc_weekly import main as modeling_btc_weekly_main
    from .backtesting_btc_weekly import main as backtesting_btc_weekly_main

    print("[BTC weekly] Step 1/3 — Prepare dataset")
    prepare_btc_weekly_main()

    print("\n[BTC weekly] Step 2/3 — Modeling")
    modeling_btc_weekly_main()

    print("\n[BTC weekly] Step 3/3 — Backtesting")
    backtesting_btc_weekly_main()

    print("\n[BTC weekly] Pipeline completed.\n")


# -------------------------------------------------------------------
#  PLOTTING PIPELINE
# -------------------------------------------------------------------
def run_plotting_pipeline() -> None:
    """
    Generate all final figures (SP500, BTC monthly, BTC weekly).

    Uses src/plot_results.py::main()
    """
    print("\n==================== PLOTTING PIPELINE ====================\n")

    try:
        from .plot_results import main as plot_results_main
    except ImportError:
        print("[PLOTTING] Could not import plot_results.main, skipping figure generation.")
        return

    plot_results_main()

    print("\n[PLOTTING] All figures generated (see fig/ directory).\n")


# -------------------------------------------------------------------
#  GLOBAL FINSIGHT PIPELINE
# -------------------------------------------------------------------
def run_finsight_pipeline() -> None:
    """
    Run the full Finsight pipeline in order:
      1) SP500 (SPY) backtest
      2) BTC monthly pipeline (backtest only)
      3) BTC weekly pipeline
      4) Plotting (all equity curves + bar charts)
    """
    run_sp500_pipeline()
    run_btc_monthly_pipeline()
    run_btc_weekly_pipeline()
    run_plotting_pipeline()
