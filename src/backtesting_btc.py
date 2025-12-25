"""
backtesting_btc.py — Backtest of BTC strategies + ML adaptive strategy.

This script:
  1) Loads the monthly labeled BTC dataset (best_strategy per month).
  2) Trains a Gradient Boosting classifier in a walk-forward way
     (5-year rolling window, one-step-ahead predictions).
  3) Builds simple benchmark strategies (Buy & Hold, Momentum, Mean Reversion).
  4) Applies the ML-based adaptive strategy and an ex-post Oracle.
  5) Computes performance statistics and saves the results to:
       - data/results/btc/monthly_returns_btc.csv
       - data/results/btc/equity_curves_btc.csv
       - data/results/btc/performance_summary_btc.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import PROCESSED_DATA_DIR
from .config_btc import RAW_PRICE_FILE, BTC_RESULTS_DIR

# ---------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------

LABEL_COL = "best_strategy"

# Features used in modeling_btc.py (we keep the same set for consistency)
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

# We fix an explicit mapping for readability and to avoid label-order issues
STRATEGY_TO_INT: Dict[str, int] = {
    "Buy & Hold": 0,
    "Mean Reversion": 1,
    "Momentum": 2,
}
INT_TO_STRATEGY: Dict[int, str] = {v: k for k, v in STRATEGY_TO_INT.items()}


# ---------------------------------------------------------------------
# 2. LOAD LABELED BTC DATA
# ---------------------------------------------------------------------

def load_btc_monthly_dataset() -> pd.DataFrame:
    """
    Load the BTC monthly labeled dataset created by prepare_btc_dataset.py.
    Ensures a datetime index and keeps only the requested feature columns
    plus the label.
    """
    file_path = PROCESSED_DATA_DIR / "labeled_data_btc.csv"
    print(f"Loading BTC labeled dataset from: {file_path}")

    df = pd.read_csv(file_path)

    if "date" not in df.columns:
        # Fallback: assume first column is the date
        date_col = df.columns[0]
    else:
        date_col = "date"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # Keep only numeric features that we expect, plus the label
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"The following expected BTC features are missing: {missing_features}"
        )

    cols_to_keep = FEATURE_COLS + [LABEL_COL]
    df = df[cols_to_keep].dropna(subset=[LABEL_COL])

    print(f"BTC monthly dataset shape: {df.shape}")
    return df


# ---------------------------------------------------------------------
# 3. BUILD FEATURES / TARGET
# ---------------------------------------------------------------------

def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Split the BTC monthly dataframe into X (features) and y (encoded target).
    """
    print(f"Features shape: ({df.shape[0]}, {len(FEATURE_COLS)})")
    print("Target distribution:\n", df[LABEL_COL].value_counts())

    X = df[FEATURE_COLS].copy()

    # Encode strategies with our fixed mapping
    y_str = df[LABEL_COL].astype(str)
    unknown = set(y_str.unique()) - set(STRATEGY_TO_INT.keys())
    if unknown:
        raise ValueError(f"Unexpected strategy labels in BTC data: {unknown}")

    y = y_str.map(STRATEGY_TO_INT).values

    print("\nEncoded classes (BTC):")
    for k, v in STRATEGY_TO_INT.items():
        print(f"  {v} -> {k}")

    return X, y


# ---------------------------------------------------------------------
# 4. MODEL: GRADIENT BOOSTING (SAME FAMILY AS modeling_btc, SLIGHTLY REGULARIZED)
# ---------------------------------------------------------------------

def build_gb_pipeline() -> Pipeline:
    """
    Build a Gradient Boosting pipeline with standardization + GB classifier.
    Parameters are a bit conservative to reduce overfitting.
    """
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("gb", gb),
        ]
    )
    return pipe


# ---------------------------------------------------------------------
# 5. WALK-FORWARD PREDICTIONS (5-YEAR ROLLING WINDOW, 1-MONTH AHEAD)
# ---------------------------------------------------------------------

def walk_forward_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    min_train_years: int = 5,
) -> pd.DataFrame:
    """
    Perform walk-forward evaluation with a 5-year rolling training window.

    For each month t:
      - use all data between (t - 5 years) and (t - 1 month) as training,
      - predict the best strategy for month t.

    Returns a DataFrame with:
      index: test month end dates
      columns: ["y_true", "y_pred"]
    """
    dates = X.index.to_period("M").to_timestamp("M")  # month-end timestamps
    X = X.copy()
    X.index = dates

    # We also align y with dates
    y_series = pd.Series(y, index=dates)

    model = build_gb_pipeline()

    results = []

    print("=== BTC walk-forward ML predictions (robust GB, 5-year rolling window) ===")

    for test_date in dates:
        train_start = test_date - pd.DateOffset(years=min_train_years)
        train_end = test_date - pd.DateOffset(months=1)

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = dates == test_date

        if train_mask.sum() < 24:  # need at least 2 years of data to start
            continue

        X_train = X.loc[train_mask]
        y_train = y_series.loc[train_mask].values

        X_test = X.loc[test_mask]
        y_test = y_series.loc[test_mask].values

        if X_test.empty:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Use all possible labels for confusion matrix to avoid shape warnings
        cm = confusion_matrix(
            y_test, y_pred, labels=[0, 1, 2]
        )

        print(
            f"Train: {X_train.index[0].date()} \u2192 {X_train.index[-1].date()} | "
            f"Test: {test_date.date()}"
        )
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  Balanced Acc.: {bal_acc:.4f}")

        results.append(
            {
                "date": test_date,
                "y_true": int(y_test[0]),
                "y_pred": int(y_pred[0]),
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
            }
        )

    if not results:
        raise RuntimeError("No walk-forward predictions were generated for BTC.")

    df_pred = pd.DataFrame(results).set_index("date").sort_index()

    print(f"\nNumber of months with ML predictions: {df_pred.shape[0]}")
    return df_pred


# ---------------------------------------------------------------------
# 6. LOAD BTC MONTHLY RETURNS (ROBUST VERSION)
# ---------------------------------------------------------------------

def load_btc_monthly_returns() -> pd.Series:
    """
    Load BTC raw price data and compute monthly close-to-close returns.

    Robust to:
      - different date column names (date / Date / Datetime / first column),
      - different price column names (Adj Close / Adj_Close / Close),
      - string / mixed dtypes in the CSV.
    """
    print(f"Loading BTC raw prices from: {RAW_PRICE_FILE}")

    df = pd.read_csv(RAW_PRICE_FILE)

    if df.empty:
        raise ValueError(f"BTC price file {RAW_PRICE_FILE} is empty.")

    # 1) Detect date column
    date_col = None
    for candidate in ["date", "Date", "Datetime"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]  # fallback

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # 2) Detect price column
    price_candidates = ["Adj Close", "Adj_Close", "Close", "close"]
    price_col = None
    for c in price_candidates:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        raise ValueError(
            f"Could not find a price column in BTC CSV. "
            f"Columns available: {list(df.columns)}"
        )

    # Force numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    if df.empty:
        raise ValueError("No valid BTC price rows after numeric conversion.")

    monthly_close = df[price_col].resample("M").last()
    monthly_ret = monthly_close.pct_change().dropna()

    print(f"BTC monthly returns shape: {monthly_ret.shape}")
    return monthly_ret


# ---------------------------------------------------------------------
# 7. BUILD STRATEGY RETURNS FOR BTC
# ---------------------------------------------------------------------

def build_btc_strategy_returns(monthly_ret: pd.Series) -> pd.DataFrame:
    """
    Build simple benchmark strategies based on monthly BTC returns.

    Definitions (consistent and transparent):
      - Buy & Hold: always invested in BTC.
      - Momentum: invested in BTC in month t if return(t-1) > 0, else 0.
      - Mean Reversion: invested in BTC in month t if return(t-1) < 0, else 0.
    """
    monthly_ret = monthly_ret.sort_index()

    bh = monthly_ret.copy()

    prev_ret = monthly_ret.shift(1)
    mom = np.where(prev_ret > 0, monthly_ret, 0.0)
    mr = np.where(prev_ret < 0, monthly_ret, 0.0)

    strategy_rets = pd.DataFrame(
        {
            "Buy & Hold": bh,
            "Momentum": mom,
            "Mean Reversion": mr,
        },
        index=monthly_ret.index,
    )

    print(
        f"\nBTC strategy return matrix shape: {strategy_rets.shape}\n"
        f"Strategy return columns: {list(strategy_rets.columns)}"
    )
    return strategy_rets


# ---------------------------------------------------------------------
# 8. PERFORMANCE STATISTICS
# ---------------------------------------------------------------------

def compute_performance_stats(returns: pd.Series) -> Dict[str, float]:
    """
    Compute total return, annualized return, annualized vol, Sharpe (rf=0),
    and max drawdown for a monthly return series.
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
# 9. MAIN BACKTEST PIPELINE FOR BTC
# ---------------------------------------------------------------------

def main():
    print(">>> backtesting_btc.py (robust version) started")

    # 1) Load BTC labeled monthly dataset
    df = load_btc_monthly_dataset()
    print(f"BTC monthly dataset shape: {df.shape}")

    X, y = prepare_features_and_target(df)
    print(f"Using features:\n {FEATURE_COLS}\n")

    # 2) Walk-forward ML predictions (one-step ahead)
    df_pred = walk_forward_predictions(X, y, min_train_years=5)

    # 3) Load BTC monthly returns and build strategy returns
    monthly_ret = load_btc_monthly_returns()
    strategy_rets = build_btc_strategy_returns(monthly_ret)

    # 4) Align ML prediction horizon with strategy returns
    #    We only keep months where we have both a prediction and returns.
    common_index = df_pred.index.intersection(strategy_rets.index)
    df_pred = df_pred.loc[common_index]
    strategy_rets = strategy_rets.loc[common_index]

    print(
        f"\nAligned period for BTC backtest: "
        f"{common_index[0].date()} \u2192 {common_index[-1].date()}"
    )
    print(f"Number of backtest months: {len(common_index)}")

    # 5) Build ML adaptive and Oracle strategies
    # ML Adaptive: choose the strategy predicted by the model each month.
    ml_rets = []
    oracle_rets = []

    for date, row in df_pred.iterrows():
        pred_class = int(row["y_pred"])
        pred_strategy = INT_TO_STRATEGY[pred_class]

        # Default to Buy & Hold if something goes wrong
        if pred_strategy not in strategy_rets.columns:
            pred_strategy = "Buy & Hold"

        ml_ret = strategy_rets.loc[date, pred_strategy]
        ml_rets.append(ml_ret)

        # Oracle: pick the best of the 3 strategies ex post
        oracle_rets.append(strategy_rets.loc[date].max())

    ml_rets = pd.Series(ml_rets, index=common_index, name="ML Adaptive")
    oracle_rets = pd.Series(oracle_rets, index=common_index, name="Oracle")

    # 6) Build final returns matrix (all strategies)
    rets_all = pd.concat(
        [
            strategy_rets.loc[common_index, ["Buy & Hold", "Momentum", "Mean Reversion"]],
            ml_rets,
            oracle_rets,
        ],
        axis=1,
    )

    # 7) Compute equity curves
    equity_curves = (1 + rets_all).cumprod()

    # 8) Compute performance summary
    perf_rows = []
    for col in rets_all.columns:
        stats = compute_performance_stats(rets_all[col])
        perf_rows.append(
            {
                "strategy": col,
                **stats,
            }
        )

    perf_df = pd.DataFrame(perf_rows).set_index("strategy")

    print("\n=== BTC Performance summary (monthly backtest) ===\n")
    for strategy, row in perf_df.iterrows():
        print(f"{strategy}:")
        print(f"  Total return:          {row['total_return'] * 100:.2f}%")
        print(f"  Annualized return:       {row['annualized_return'] * 100:.2f}%")
        print(f"  Annualized vol:         {row['annualized_vol'] * 100:.2f}%")
        print(f"  Sharpe ratio (rf=0):     {row['sharpe']:.2f}")
        print(f"  Max drawdown:          {row['max_drawdown'] * 100:.2f}%\n")

    # 9) Save results
    BTC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rets_path = BTC_RESULTS_DIR / "monthly_returns_btc.csv"
    eq_path = BTC_RESULTS_DIR / "equity_curves_btc.csv"
    perf_path = BTC_RESULTS_DIR / "performance_summary_btc.csv"

    rets_all.to_csv(rets_path)
    equity_curves.to_csv(eq_path)
    perf_df.to_csv(perf_path)

    print(f"Saved BTC monthly returns to: {rets_path}")
    print(f"Saved BTC equity curves to: {eq_path}")
    print(f"Saved BTC performance summary to: {perf_path}")
    print(">>> backtesting_btc.py finished")


if __name__ == "__main__":
    main()
