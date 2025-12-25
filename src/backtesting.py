"""
backtesting.py — SP500 simple backtest + ML Adaptive (SPY).

This script:
  1) Loads daily SPY prices from data/raw.
  2) Computes monthly returns.
  3) Builds three simple strategies:
       - Buy & Hold
       - Momentum (invested if previous month > 0)
       - Mean Reversion (invested if previous month < 0)
     + an ex-post Oracle (best of the 3 each month).
  4) Builds a monthly ML dataset:
       - features = functions of past returns (ret_1m, ret_3m, ret_6m, vol_3m, vol_6m)
       - label = ex-post optimal strategy for the following month
     and runs a walk-forward backtest of the "ML Adaptive" strategy.
  5) Computes performance statistics.
  6) Saves results to:
       data/results/sp500/monthly_returns_sp500.csv
       data/results/sp500/equity_curves_sp500.csv
       data/results/sp500/performance_summary_sp500.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
SP500_RESULTS_DIR = BASE_DIR / "data" / "results" / "sp500"
SP500_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. UTILITY: FIND THE SPY PRICE FILE
# ---------------------------------------------------------------------

def find_spy_price_file() -> Path:
    """
    Looks for a reasonable SPY price file in data/raw.
    Tries a few standard names, then falls back to the first CSV found.
    """
    candidates = [
        "SPY_prices.csv",
        "SPY.csv",
        "spy_prices.csv",
        "spy.csv",
        "sp500_prices.csv",
        "SP500_prices.csv",
    ]
    for name in candidates:
        path = RAW_DIR / name
        if path.exists():
            return path

    # fallback: any csv (better than nothing)
    csv_files = list(RAW_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"[SP500] No CSV file found in {RAW_DIR}. "
        "Add for example 'SPY_prices.csv'."
    )


# ---------------------------------------------------------------------
# 3. LOAD PRICES AND COMPUTE MONTHLY RETURNS
# ---------------------------------------------------------------------

def load_sp500_monthly_returns() -> pd.Series:
    """
    Loads daily SP500/ETF SPY prices and computes
    monthly close-to-close returns.

    Hardened:
      - automatic detection of the date column,
      - automatic detection of the price column (Adj Close, Close, etc.),
      - explicit numeric conversion.
    """
    price_file = find_spy_price_file()
    print(f"[SP500] Loading raw prices from: {price_file}")

    df = pd.read_csv(price_file)

    if df.empty:
        raise ValueError(f"[SP500] Price file {price_file} is empty.")

    # 1) Date column
    date_col = None
    for candidate in ["date", "Date", "Datetime", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]  # fallback

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # 2) Price column
    price_candidates = ["Adj Close", "Adj_Close", "Close", "close", "Price", "price"]
    price_col = None
    for c in price_candidates:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError(
            f"[SP500] Unable to find a price column in {price_file}. "
            f"Available columns: {list(df.columns)}"
        )

    # 3) Convert to numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    if df.empty:
        raise ValueError("[SP500] No valid prices after numeric conversion.")

    # 4) Monthly returns
    # 'M' triggers a FutureWarning, but is still correct here.
    monthly_close = df[price_col].resample("M").last()
    monthly_ret = monthly_close.pct_change().dropna()

    print(f"[SP500] Monthly returns shape: {monthly_ret.shape}")
    return monthly_ret


# ---------------------------------------------------------------------
# 4. BUILD SIMPLE STRATEGIES
# ---------------------------------------------------------------------

def build_sp500_strategies(monthly_ret: pd.Series) -> pd.DataFrame:
    """
    Builds the returns of simple strategies based on
    SP500 monthly returns.
    """
    monthly_ret = monthly_ret.sort_index()

    bh = monthly_ret.copy()
    prev_ret = monthly_ret.shift(1)

    mom = np.where(prev_ret > 0, monthly_ret, 0.0)
    mr = np.where(prev_ret < 0, monthly_ret, 0.0)

    df_strat = pd.DataFrame(
        {
            "Buy & Hold": bh,
            "Momentum": mom,
            "Mean Reversion": mr,
        },
        index=monthly_ret.index,
    )

    print(
        f"[SP500] Strategy return matrix shape: {df_strat.shape}\n"
        f"[SP500] Columns: {list(df_strat.columns)}"
    )
    return df_strat


# ---------------------------------------------------------------------
# 5. ML DATASET FOR SP500 (features + best_strategy label)
# ---------------------------------------------------------------------

def build_sp500_ml_dataset(
    monthly_ret: pd.Series,
    strat_rets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds a monthly ML dataset for SP500.

    Features (observed at the end of month t):
      - ret_1m: return of month t
      - ret_3m: sum of returns t, t-1, t-2
      - ret_6m: sum of returns t..t-5
      - vol_3m: volatility (std) of returns over 3 months
      - vol_6m: volatility over 6 months

    Label (best_strategy) for month t:
      = strategy that will have the highest return in month t+1
        among {Buy & Hold, Momentum, Mean Reversion}.
    """
    monthly_ret = monthly_ret.sort_index()
    strat_rets = strat_rets.sort_index()

    # Future returns by strategy (t+1)
    future_rets = strat_rets.shift(-1)
    best_strategy = future_rets[["Buy & Hold", "Momentum", "Mean Reversion"]].idxmax(axis=1)

    df = pd.DataFrame(index=monthly_ret.index.copy())
    df["ret_1m"] = monthly_ret
    df["ret_3m"] = monthly_ret.rolling(window=3).sum()
    df["ret_6m"] = monthly_ret.rolling(window=6).sum()
    df["vol_3m"] = monthly_ret.rolling(window=3).std()
    df["vol_6m"] = monthly_ret.rolling(window=6).std()

    df["best_strategy"] = best_strategy

    df_ml = df.dropna()
    print(f"[SP500|ML] ML dataset shape: {df_ml.shape}")
    print("[SP500|ML] Label distribution:")
    print(df_ml["best_strategy"].value_counts())
    return df_ml


# ---------------------------------------------------------------------
# 6. WALK-FORWARD ML ADAPTIVE BACKTEST ON SP500
# ---------------------------------------------------------------------

def run_sp500_ml_adaptive(
    df_ml: pd.DataFrame,
    strat_rets: pd.DataFrame,
    min_train: int = 60,
) -> pd.Series:
    """
    Monthly walk-forward backtest on SP500 with Gradient Boosting.

    At each date t >= min_train:
      - train the model on ML data from the start up to t-1,
      - predict the strategy to use in month t,
      - take the return of that strategy in month t.
    """
    df_ml = df_ml.sort_index()
    strat_rets = strat_rets.sort_index()

    feature_cols = [c for c in df_ml.columns if c != "best_strategy"]
    X = df_ml[feature_cols]
    y = df_ml["best_strategy"]

    dates = X.index.to_list()
    if len(dates) <= min_train:
        raise ValueError(
            f"[SP500|ML] Too few observations ({len(dates)}) "
            f"for a training window of {min_train} months."
        )

    ml_returns = []

    # Model: scaler + Gradient Boosting
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            )),
        ]
    )

    for i in range(min_train, len(dates)):
        train_idx = dates[:i]
        test_date = dates[i]

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[[test_date]]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        if pred not in strat_rets.columns:
            # safety fallback
            ret = strat_rets.loc[test_date, "Buy & Hold"]
        else:
            ret = strat_rets.loc[test_date, pred]

        ml_returns.append((test_date, ret))

    ml_ret_series = pd.Series(
        data=[r for (_, r) in ml_returns],
        index=[d for (d, _) in ml_returns],
        name="ML Adaptive",
    )

    print(
        f"[SP500|ML] Walk-forward ML Adaptive generated for "
        f"{len(ml_ret_series)} months (training window ≥ {min_train} months)."
    )
    return ml_ret_series


# ---------------------------------------------------------------------
# 7. PERFORMANCE STATISTICS
# ---------------------------------------------------------------------

def compute_performance_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Total return, annualized return, annualized vol, Sharpe (rf=0),
    and max drawdown for a series of monthly returns.
    """
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0

    n_months = returns.shape[0]
    ann_factor = 12.0 / n_months
    annualized_return = (1 + total_return) ** ann_factor - 1.0

    annualized_vol = returns.std(ddof=1) * np.sqrt(12.0)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    max_drawdown = drawdowns.min()

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": float(annualized_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }


# ---------------------------------------------------------------------
# 8. MAIN: SP500 PIPELINE
# ---------------------------------------------------------------------

def main():
    print("\n========== SP500 BACKTEST ==========\n")

    # 1) Monthly returns
    monthly_ret = load_sp500_monthly_returns()

    # 2) Simple strategies
    strat_rets = build_sp500_strategies(monthly_ret)

    # 3) Ex-post Oracle (best of the 3 simple strategies)
    oracle = strat_rets.max(axis=1)
    oracle.name = "Oracle (best of 3, ex post)"

    # 4) Initial combination (without ML)
    rets_all = pd.concat([strat_rets, oracle], axis=1)

    # 5) SP500 ML Adaptive (walk-forward)
    try:
        df_ml = build_sp500_ml_dataset(monthly_ret, strat_rets)
        ml_ret = run_sp500_ml_adaptive(df_ml, strat_rets, min_train=60)

        # Align to the full monthly returns index
        ml_ret_aligned = rets_all.index.to_series().map(ml_ret).fillna(0.0)
        rets_all["ML Adaptive"] = ml_ret_aligned.values

    except Exception as e:
        print(f"[SP500|ML] Error while building ML Adaptive: {e}")
        print("[SP500|ML] ML Adaptive will be disabled (returns = 0).")
        rets_all["ML Adaptive"] = 0.0

    # 6) Reorder columns for readability
    cols_order = [
        "Buy & Hold",
        "Momentum",
        "Mean Reversion",
        "ML Adaptive",
        "Oracle (best of 3, ex post)",
    ]
    rets_all = rets_all[cols_order]

    # 7) Equity curves
    equity_curves = (1 + rets_all).cumprod()

    # 8) Performance summary
    perf_rows = []
    for col in rets_all.columns:
        stats = compute_performance_stats(rets_all[col])
        perf_rows.append({"strategy": col, **stats})

    perf_df = pd.DataFrame(perf_rows).set_index("strategy")

    print("\n=== SP500 Performance summary (monthly backtest) ===\n")
    for strategy, row in perf_df.iterrows():
        print(f"{strategy}:")
        print(f"  Total return:          {row['total_return'] * 100:.2f}%")
        print(f"  Annualized return:       {row['annualized_return'] * 100:.2f}%")
        print(f"  Annualized vol:         {row['annualized_vol'] * 100:.2f}%")
        print(f"  Sharpe ratio (rf=0):     {row['sharpe']:.2f}")
        print(f"  Max drawdown:          {row['max_drawdown'] * 100:.2f}%\n")

    # 9) Save outputs
    rets_path = SP500_RESULTS_DIR / "monthly_returns_sp500.csv"
    eq_path = SP500_RESULTS_DIR / "equity_curves_sp500.csv"
    perf_path = SP500_RESULTS_DIR / "performance_summary_sp500.csv"

    rets_all.to_csv(rets_path)
    equity_curves.to_csv(eq_path)
    perf_df.to_csv(perf_path)

    print(f"[SP500] Saved monthly returns to: {rets_path}")
    print(f"[SP500] Saved equity curves to:   {eq_path}")
    print(f"[SP500] Saved performance summary to: {perf_path}")
    print("\n>>> backtesting.py (SP500) finished\n")


if __name__ == "__main__":
    main()
