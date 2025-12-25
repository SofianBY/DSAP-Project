# src/plot_results.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# chemins de base (relatifs au repo)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RESULTS_DIR = BASE_DIR / "data" / "results"
FIG_DIR = BASE_DIR / "fig"

FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curves(csv_path: Path, fig_path: Path, title: str):
    if not csv_path.exists():
        print(f"[WARN] Equity curves file not found: {csv_path}")
        return

    print(f"[OK] Loading equity curves from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    plt.figure(figsize=(10, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (start = 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[OK] Saved equity curve: {fig_path}")


def plot_performance_bar(perf_path: Path, fig_path: Path):
    """
    Lit un performance_summary_*.csv et trace un histogramme
    de 'annualized_return' pour *toutes* les stratégies qu'il contient.
    """
    if not perf_path.exists():
        print(f"[WARN] Performance summary missing: {perf_path}")
        return

    df = pd.read_csv(perf_path, index_col=0)

    if "annualized_return" not in df.columns:
        print("[SKIP] Annualized return not in performance summary.")
        return

    # On garde l'ordre du CSV
    strategies = df.index.tolist()
    values = df["annualized_return"].values

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(strategies)), values)
    plt.xticks(range(len(strategies)), strategies, rotation=45, ha="right")
    plt.ylabel("Annualized return")
    plt.title("Annualized return by strategy")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"[OK] Saved bar chart: {fig_path}")


def main():
    print(">>> Generating figures...")

    # -------- SP500 --------

sp500_dir = DATA_RESULTS_DIR / "sp500"

sp500_eq_candidates = [
    sp500_dir / "equity_curves_sp500.csv",
    sp500_dir / "equity_curves.csv",
]
sp500_eq = next((p for p in sp500_eq_candidates if p.exists()), None)

if sp500_eq is not None:
    plot_equity_curves(
        sp500_eq,
        FIG_DIR / "sp500_equity_curves.png",
        "SP500 – Equity curves",
    )
else:
    print("[WARN] SP500 equity not found, skipping.")


    # -------- BTC monthly --------
    btc_dir = DATA_RESULTS_DIR / "btc"
    btc_eq = btc_dir / "equity_curves_btc.csv"
    btc_perf = btc_dir / "performance_summary_btc.csv"

    plot_equity_curves(
        btc_eq,
        FIG_DIR / "btc_monthly_equity_curves.png",
        "BTC (monthly) – Equity curves",
    )
    plot_performance_bar(
        btc_perf,
        FIG_DIR / "btc_monthly_annualized_return_bar.png",
    )

    # -------- BTC weekly --------
    btcw_dir = DATA_RESULTS_DIR / "btc_weekly"
    btcw_eq = btcw_dir / "equity_curves_btc.csv"
    btcw_perf = btcw_dir / "performance_summary_btc.csv"

    plot_equity_curves(
        btcw_eq,
        FIG_DIR / "btc_weekly_equity_curves.png",
        "BTC (weekly) – Equity curves",
    )
    plot_performance_bar(
        btcw_perf,
        FIG_DIR / "btc_weekly_annualized_return_bar.png",
    )

    print(">>> plot_results.py finished")


if __name__ == "__main__":
    main()
