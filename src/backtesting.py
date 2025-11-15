# src/backtesting.py

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src import config


def load_daily_labeled_data() -> pd.DataFrame:
    """
    Load labeled_data.csv (daily frequency) and set a proper DatetimeIndex on 'date'.
    """
    path = config.PROCESSED_DATA_DIR / "labeled_data.csv"
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"No 'date' column found in labeled_data. Columns: {df.columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    return df


def build_monthly_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the daily labeled dataset to a monthly level.

    - We use the last available daily row of each month (resample('M').last()).
    - We keep all numeric features (including enriched macro/tech features).
    - We compute monthly SPY returns (based on Adj Close).
    - We construct:
        * bh_return: buy & hold monthly return
        * mom_return: simple 1-month momentum strategy
          (invest this month if last month's return was positive, else stay in cash)
    """
    # Monthly snapshot: last row of each calendar month
    monthly = df_daily.resample("M").last()

    if "Adj Close" not in monthly.columns:
        raise ValueError("Adj Close column is required to compute monthly returns.")

    # Monthly buy & hold return based on end-of-month Adj Close
    monthly["month_return"] = monthly["Adj Close"].pct_change()

    # Buy & hold = always invested
    monthly["bh_return"] = monthly["month_return"]

    # Simple 1M momentum: invest if previous month_return > 0, else 0 exposure
    prev_ret = monthly["month_return"].shift(1)
    momentum_signal = (prev_ret > 0).astype(float)
    monthly["mom_signal"] = momentum_signal
    monthly["mom_return"] = momentum_signal * monthly["month_return"]

    # Drop first few rows where returns are NaN
    monthly = monthly.dropna(subset=["month_return", "bh_return", "mom_return"])

    return monthly


def build_features_and_target(monthly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and binary target y for the ML model.

    We want the model to decide at the beginning of month t whether
    to use Buy & Hold or Momentum during month t.

    To avoid look-ahead:
      - Features X_t are taken from month t-1 (state of the market at the end of month t-1).
      - Target y_t is defined using returns of month t:
          y_t = 1  if mom_return_t > bh_return_t  (Momentum is better)
                0  otherwise (Buy & Hold is better or equal)

    We then align X and y on the same time index.
    """
    # Define target based on realized monthly returns of both strategies
    alt_better = (monthly["mom_return"] > monthly["bh_return"]).astype(int)
    y = alt_better

    # Features: all numeric columns *except* returns and signals
    drop_cols = [
        "month_return", "bh_return", "mom_return", "mom_signal"
    ]
    feature_cols = [
        c for c in monthly.columns
        if pd.api.types.is_numeric_dtype(monthly[c]) and c not in drop_cols
    ]

    # Features at time t-1 (information available at the end of previous month)
    X_raw = monthly[feature_cols]
    X_shifted = X_raw.shift(1)

    # Align X and y: drop rows with NaNs in X or y
    data = pd.concat([X_shifted, y], axis=1).dropna()
    X = data[feature_cols]
    y_aligned = data[0]  # the name of the concatenated Series is 0

    return X, y_aligned


def rolling_logistic_backtest(
    monthly: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 60,
) -> Dict[str, pd.Series]:
    """
    Perform an expanding-window backtest with a Logistic Regression classifier.

    At each step t:
      - Train on all months before t (at least min_train_size).
      - Predict y_hat_t (0=Buy & Hold, 1=Momentum) for month t.
      - Apply the chosen strategy to obtain the realized return for t.

    We return the equity curves (starting at 1.0) for:
      - static buy & hold
      - static momentum
      - ML adaptive strategy
    """
    dates = X.index
    n = len(dates)

    if n <= min_train_size + 1:
        raise ValueError(
            f"Not enough monthly observations ({n}) for min_train_size={min_train_size}."
        )

    # Containers for returns
    bh_rets = []
    mom_rets = []
    ml_rets = []
    test_dates = []

    for i in range(min_train_size, n):
        train_idx = dates[:i]
        test_idx = dates[i]

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[[test_idx]]

        # Standardization + Logistic Regression with class_weight='balanced'
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)[0]

        # Get realized returns for month t
        ret_bh = monthly.loc[test_idx, "bh_return"]
        ret_mom = monthly.loc[test_idx, "mom_return"]

        # Static strategies
        bh_rets.append(ret_bh)
        mom_rets.append(ret_mom)

        # ML adaptive strategy
        if y_pred == 0:
            ml_rets.append(ret_bh)
        else:
            ml_rets.append(ret_mom)

        test_dates.append(test_idx)

    # Convert to Series aligned on test period
    bh_rets = pd.Series(bh_rets, index=test_dates, name="bh_return")
    mom_rets = pd.Series(mom_rets, index=test_dates, name="mom_return")
    ml_rets = pd.Series(ml_rets, index=test_dates, name="ml_return")

    # Build equity curves (starting at 1.0)
    bh_equity = (1 + bh_rets).cumprod()
    mom_equity = (1 + mom_rets).cumprod()
    ml_equity = (1 + ml_rets).cumprod()

    return {
        "bh_equity": bh_equity,
        "mom_equity": mom_equity,
        "ml_equity": ml_equity,
        "bh_rets": bh_rets,
        "mom_rets": mom_rets,
        "ml_rets": ml_rets,
    }


def compute_performance_stats(returns: pd.Series, periods_per_year: int = 12) -> Dict[str, float]:
    """
    Compute simple performance statistics for a series of periodic returns:
      - total_return
      - annualized_return
      - annualized_volatility
      - sharpe_ratio (rf=0)
      - max_drawdown
    """
    rets = returns.dropna()
    if len(rets) == 0:
        return {k: np.nan for k in [
            "total_return", "annualized_return", "annualized_volatility",
            "sharpe_ratio", "max_drawdown"
        ]}

    equity = (1 + rets).cumprod()
    total_return = equity.iloc[-1] - 1.0

    # Annualized return (geometric)
    n_periods = len(rets)
    years = n_periods / periods_per_year
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = np.nan

    # Annualized volatility
    vol = rets.std() * np.sqrt(periods_per_year)

    # Sharpe ratio (rf=0)
    sharpe = annualized_return / vol if vol > 0 else np.nan

    # Max drawdown
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    max_dd = drawdowns.min()

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def main():
    print(">>> backtesting.py started")

    # 1) Load daily labeled data
    df_daily = load_daily_labeled_data()
    print("Daily labeled data shape:", df_daily.shape)

    # 2) Aggregate to monthly and build strategy returns
    monthly = build_monthly_frame(df_daily)
    print("Monthly frame shape:", monthly.shape)

    # 3) Build features (X) and binary target (y)
    X, y = build_features_and_target(monthly)
    print("Monthly features shape:", X.shape)
    print("Binary target distribution:")
    print(y.value_counts())

    # 4) Run rolling logistic regression backtest
    results = rolling_logistic_backtest(monthly, X, y, min_train_size=60)

    bh_rets = results["bh_rets"]
    mom_rets = results["mom_rets"]
    ml_rets = results["ml_rets"]

    # 5) Compute performance statistics
    bh_stats = compute_performance_stats(bh_rets)
    mom_stats = compute_performance_stats(mom_rets)
    ml_stats = compute_performance_stats(ml_rets)

    print("\n=== Performance summary (monthly backtest) ===")
    def fmt_stats(name, stats):
        print(f"\n{name}:")
        print(f"  Total return:        {stats['total_return']:.2%}")
        print(f"  Annualized return:   {stats['annualized_return']:.2%}")
        print(f"  Annualized vol:      {stats['annualized_volatility']:.2%}")
        print(f"  Sharpe ratio (rf=0): {stats['sharpe_ratio']:.2f}")
        print(f"  Max drawdown:        {stats['max_drawdown']:.2%}")

    fmt_stats("Buy & Hold", bh_stats)
    fmt_stats("Momentum", mom_stats)
    fmt_stats("ML Adaptive (BH vs Momentum)", ml_stats)

    # 6) Show last few points of equity curves
    print("\nLast 10 equity values:")
    bh_eq = results["bh_equity"]
    mom_eq = results["mom_equity"]
    ml_eq = results["ml_equity"]

    equity_df = pd.concat([bh_eq, mom_eq, ml_eq], axis=1)
    equity_df.columns = ["BH", "Momentum", "ML Adaptive"]
    print(equity_df.tail(10))

    print(">>> backtesting.py finished")


if __name__ == "__main__":
    main()
