"""
backtesting_btc_weekly.py

Walk-forward backtest on BTC at weekly frequency:
- uses GradientBoosting (set with reasonable hyperparameters)
- expanding-window: start after 2 years of data
- compares:
    * Buy & Hold
    * Momentum
    * Mean Reversion
    * ML Adaptive (strategy selection)
    * Oracle (best of the 3 ex post)
- saves results to data/results/btc_weekly/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "BTC_prices.csv"
LABEL_FILE = DATA_DIR / "processed" / "labeled_data_btc_weekly.csv"
RESULT_DIR = DATA_DIR / "results" / "btc_weekly"

FEATURE_COLS = [
    "ret_1d",
    "vol_21d",
    "vol_63d",
    "px_over_sma_10",
    "px_over_sma_50",
    "px_over_sma_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_zscore_21d",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def compute_performance(returns: pd.Series, freq_per_year: int = 52) -> dict:
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1

    n = len(returns)
    ann_return = (1 + total_return) ** (freq_per_year / n) - 1 if n > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(freq_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def load_btc_prices_weekly():
    df = pd.read_csv(RAW_FILE)

    # Date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"])

    # Choose price column
    px_col = None
    for c in df.columns:
        cl = c.lower()
        if "adj" in cl and "close" in cl:
            px_col = c
            break
    if px_col is None:
        for c in df.columns:
            if "close" in c.lower():
                px_col = c
                break
    if px_col is None:
        px_col = df.columns[1]

    df["Adj Close"] = pd.to_numeric(df[px_col], errors="coerce")
    df = df.dropna(subset=["Adj Close"])
    df = df[["date", "Adj Close"]].set_index("date").sort_index()

    # Resample to weekly
    weekly_px = df["Adj Close"].resample("W-FRI").last().dropna()
    weekly_ret = weekly_px.pct_change().dropna()

    return weekly_px, weekly_ret


def compute_static_strategy_returns(weekly_ret: pd.Series) -> pd.DataFrame:
    # Buy & Hold
    bh_ret = weekly_ret.copy()

    # Momentum (4 weeks)
    momo_signal = (weekly_ret.rolling(4).sum() > 0).shift(1).fillna(0).astype(int)
    momo_ret = momo_signal * weekly_ret

    # Mean reversion
    mr_signal = (weekly_ret.shift(1) < 0).astype(int)
    mr_ret = mr_signal * weekly_ret

    out = pd.DataFrame(
        {
            "Buy & Hold": bh_ret,
            "Momentum": momo_ret,
            "Mean Reversion": mr_ret,
        },
        index=weekly_ret.index,
    )
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(">>> backtesting_btc_weekly.py started")

    # 1) Weekly dataset with labels
    print(f"Loading BTC weekly labeled dataset from: {LABEL_FILE}")
    df = pd.read_csv(LABEL_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    X = df[FEATURE_COLS].astype(float)
    y = df["best_strategy"]

    print("BTC weekly dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts())

    # 2) Model (same hyperparams as in modeling, reasonable)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.7,
        random_state=42,
    )

    # 3) Walk-forward expanding window
    min_train_weeks = 104  # ~ 2 years
    dates = X.index
    preds = []

    for i in range(min_train_weeks, len(X)):
        train_idx = slice(0, i)
        test_idx = slice(i, i + 1)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)[0]

        preds.append((dates[i], y_pred))

    ml_pred = pd.Series(
        data=[p[1] for p in preds],
        index=[p[0] for p in preds],
        name="ml_strategy",
    )

    print(f"Number of weeks with ML predictions: {len(ml_pred)}")

    # 4) Weekly prices and static strategies
    weekly_px, weekly_ret = load_btc_prices_weekly()
    static_rets = compute_static_strategy_returns(weekly_ret)

    # Align dates
    common_idx = static_rets.index.intersection(ml_pred.index)
    static_rets = static_rets.loc[common_idx]
    ml_pred = ml_pred.loc[common_idx]

    # 5) ML strategy returns
    bh = static_rets["Buy & Hold"]
    mom = static_rets["Momentum"]
    mr = static_rets["Mean Reversion"]

    ml_ret = []
    for dt in common_idx:
        strat = ml_pred.loc[dt]
        if strat == "Buy & Hold":
            ml_ret.append(bh.loc[dt])
        elif strat == "Momentum":
            ml_ret.append(mom.loc[dt])
        else:
            ml_ret.append(mr.loc[dt])

    ml_ret = pd.Series(ml_ret, index=common_idx, name="ML Adaptive")

    # Oracle: best of the 3 ex post
    oracle_ret = static_rets.max(axis=1)
    oracle_ret.name = "Oracle"

    # 6) Performance
    freq = 52
    perf = {}
    perf["Buy & Hold"] = compute_performance(bh.loc[common_idx], freq)
    perf["Momentum"] = compute_performance(mom.loc[common_idx], freq)
    perf["Mean Reversion"] = compute_performance(mr.loc[common_idx], freq)
    perf["ML Adaptive"] = compute_performance(ml_ret, freq)
    perf["Oracle"] = compute_performance(oracle_ret, freq)

    # 7) Print summary
    print("\n=== BTC Weekly Performance summary (walk-forward) ===\n")
    for name, stats in perf.items():
        print(f"{name}:")
        print(f"  Total return:          {stats['total_return'] * 100:8.2f}%")
        print(f"  Annualized return:       {stats['annualized_return'] * 100:6.2f}%")
        print(f"  Annualized vol:         {stats['annualized_vol'] * 100:6.2f}%")
        print(f"  Sharpe ratio (rf=0):     {stats['sharpe']:.2f}")
        print(f"  Max drawdown:          {stats['max_drawdown'] * 100:8.2f}%")
        print()

    # 8) Save outputs
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Return series
    all_rets = static_rets.loc[common_idx].copy()
    all_rets["ML Adaptive"] = ml_ret
    all_rets["Oracle"] = oracle_ret
    all_rets.to_csv(RESULT_DIR / "weekly_returns_btc.csv")

    # Equity curves
    equity = (1 + all_rets).cumprod()
    equity.to_csv(RESULT_DIR / "equity_curves_btc.csv")

    # Performance summary
    perf_df = (
        pd.DataFrame(perf)
        .T[["total_return", "annualized_return", "annualized_vol", "sharpe", "max_drawdown"]]
    )
    perf_df.to_csv(RESULT_DIR / "performance_summary_btc.csv")

    print(f"Saved BTC weekly returns to: {RESULT_DIR / 'weekly_returns_btc.csv'}")
    print(f"Saved BTC weekly equity curves to: {RESULT_DIR / 'equity_curves_btc.csv'}")
    print(f"Saved BTC weekly performance summary to: {RESULT_DIR / 'performance_summary_btc.csv'}")
    print(">>> backtesting_btc_weekly.py finished")


if __name__ == "__main__":
    main()
