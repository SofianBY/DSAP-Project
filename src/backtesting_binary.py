# src/backtesting_binary.py
"""
Binary backtest:
    - Baseline strategy: Buy & Hold (always invested in SPY).
    - Active block: choose between Momentum and Mean Reversion.

The ML model answers a simpler question:
    "Should we stay in Buy & Hold next month, or switch to an active strategy?"

Target definition (monthly horizon):
    y_t = 0  -> Buy & Hold is better or equal to the best active strategy
    y_t = 1  -> The best active strategy (Momentum or Mean Reversion) beats Buy & Hold

At decision time for month t:
    - Features are based on information available at the end of month t-1
      (we use lagged features).
    - If the model predicts y_hat_t = 0 -> we stay in Buy & Hold.
    - If y_hat_t = 1 -> we switch to an active strategy:
            * If previous month return > 0 -> use Momentum
            * If previous month return < 0 -> use Mean Reversion
            * If previous month return == 0 -> stay in Buy & Hold (fallback)

We compare:
    - Static Buy & Hold
    - Static Momentum
    - Static Mean Reversion
    - ML Binary strategy (BH vs Active)
    - Oracle (ex post best of the three strategies)
"""

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src import config


# ---------------------------------------------------------------------
# 1) Load daily labeled data
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 2) Build monthly frame and strategy returns
# ---------------------------------------------------------------------
def build_monthly_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the daily labeled dataset to a monthly level.

    - Use the last available daily row of each month (resample('M').last()).
    - Compute monthly SPY returns (based on Adj Close).
    - Build 3 strategy returns:
        * bh_return: buy & hold (always invested)
        * mom_return: simple 1M momentum (invest if previous month_return > 0)
        * mr_return: simple 1M mean reversion (invest if previous month_return < 0)
    - Also store prev_month_return (month_return_{t-1}), which is used
      by the decision rule to choose between Momentum and Mean Reversion.
    """
    # Monthly snapshot: last row of each calendar month
    monthly = df_daily.resample("M").last()

    if "Adj Close" not in monthly.columns:
        raise ValueError("Adj Close column is required to compute monthly returns.")

    # Monthly return based on end-of-month Adj Close
    monthly["month_return"] = monthly["Adj Close"].pct_change()

    # Buy & Hold: always invested
    monthly["bh_return"] = monthly["month_return"]

    # Previous month return (used later by the rule-based switch)
    monthly["prev_month_return"] = monthly["month_return"].shift(1)

    # Momentum: invest this month if previous month_return > 0
    prev_ret = monthly["prev_month_return"]
    mom_signal = (prev_ret > 0).astype(float)
    monthly["mom_signal"] = mom_signal
    monthly["mom_return"] = mom_signal * monthly["month_return"]

    # Mean reversion: invest this month if previous month_return < 0
    mr_signal = (prev_ret < 0).astype(float)
    monthly["mr_signal"] = mr_signal
    monthly["mr_return"] = mr_signal * monthly["month_return"]

    # Drop early rows where returns are NaN (e.g. first month)
    monthly = monthly.dropna(subset=["month_return", "bh_return", "mom_return", "mr_return"])

    return monthly


# ---------------------------------------------------------------------
# 3) Build features and BINARY target (BH vs Active)
# ---------------------------------------------------------------------
def build_features_and_binary_target(monthly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and binary target y.

    We want the model to decide at the beginning of month t whether:
        - we should stay in Buy & Hold, or
        - we should switch to an active strategy (Momentum OR Mean Reversion).

    Target definition:
        active_best_t = max(mom_return_t, mr_return_t)
        y_t = 1 if active_best_t > bh_return_t
              0 otherwise

    To avoid look-ahead:
        - Features X_t are taken from month t-1 (state of the market at the end
          of month t-1).
        - Target y_t uses realized returns of month t.

    Implementation:
        - Start from the monthly dataframe (one row per month).
        - Compute y_t.
        - Build a set of numeric features, excluding:
            * the strategy returns and signals
            * the helper variable "prev_month_return"
        - Use lagged features: X_lagged_t = X_raw_{t-1}.
        - Align X and y on the same index.
    """
    # --- Target y: should we go active (1) or stay BH (0) ? ---
    active_best = monthly[["mom_return", "mr_return"]].max(axis=1)
    y = (active_best > monthly["bh_return"]).astype(int)
    y.name = "go_active"

    # --- Features: all numeric columns except returns and signals ---
    drop_cols = [
        "month_return",
        "bh_return",
        "mom_return",
        "mr_return",
        "mom_signal",
        "mr_signal",
        "prev_month_return",  # used only by the rule, not by ML
    ]

    feature_cols = [
        c for c in monthly.columns
        if pd.api.types.is_numeric_dtype(monthly[c]) and c not in drop_cols
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found for modeling.")

    X_raw = monthly[feature_cols]

    # Lag features by one month: information available at end of t-1
    X_lagged = X_raw.shift(1)

    # Align X and y on the same index (drop rows with NaNs)
    data = pd.concat([X_lagged, y], axis=1).dropna()
    X = data[feature_cols]
    y_aligned = data["go_active"]

    return X, y_aligned


# ---------------------------------------------------------------------
# 4) Rolling binary backtest (BH vs Active)
# ---------------------------------------------------------------------
def rolling_binary_backtest(
    monthly: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 60,
    prob_threshold: float = 0.55,
) -> Dict[str, pd.Series]:
    """
    Perform an expanding-window backtest with a binary Logistic Regression.

    At each step t:
        - Train on all months before t (at least min_train_size).
        - Predict P(go_active_t = 1) for month t.
        - If probability >= prob_threshold -> switch to Active for month t.
          Otherwise -> stay in Buy & Hold.

    When we go Active, we choose between Momentum and Mean Reversion
    using a simple, transparent rule:
        - If prev_month_return_t > 0  -> Momentum
        - If prev_month_return_t < 0  -> Mean Reversion
        - Else                        -> Buy & Hold (fallback)

    We return equity curves (starting at 1.0) and monthly returns for:
        - static Buy & Hold
        - static Momentum
        - static Mean Reversion
        - ML Binary (BH vs Active)
        - Oracle (ex post best of the three strategies)
    """
    dates = X.index
    n = len(dates)

    if n <= min_train_size + 1:
        raise ValueError(
            f"Not enough monthly observations ({n}) for min_train_size={min_train_size}."
        )

    bh_rets = []
    mom_rets = []
    mr_rets = []
    ml_rets = []
    oracle_rets = []
    decisions = []  # 0 = BH, 1 = Active
    probs_active = []
    test_dates = []

    for i in range(min_train_size, n):
        train_idx = dates[:i]
        test_idx = dates[i]

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[[test_idx]]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Binary Logistic Regression with class balancing
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        clf.fit(X_train_scaled, y_train)

        # Predict probability of "go_active = 1"
        prob = clf.predict_proba(X_test_scaled)[0, 1]
        probs_active.append(prob)

        go_active = int(prob >= prob_threshold)
        decisions.append(go_active)

        # Realized returns of the 3 strategies for the test month
        ret_bh = monthly.loc[test_idx, "bh_return"]
        ret_mom = monthly.loc[test_idx, "mom_return"]
        ret_mr = monthly.loc[test_idx, "mr_return"]

        # Oracle: ex post best of the three strategies
        ret_oracle = max(ret_bh, ret_mom, ret_mr)

        # ML Binary strategy
        if go_active == 0:
            # Stay in Buy & Hold
            ret_ml = ret_bh
        else:
            # Switch to active: choose Momentum or Mean Reversion
            prev_ret = monthly.loc[test_idx, "prev_month_return"]

            if prev_ret > 0:
                ret_ml = ret_mom
            elif prev_ret < 0:
                ret_ml = ret_mr
            else:
                # If we have no clear sign on the previous month, stay BH
                ret_ml = ret_bh

        bh_rets.append(ret_bh)
        mom_rets.append(ret_mom)
        mr_rets.append(ret_mr)
        ml_rets.append(ret_ml)
        oracle_rets.append(ret_oracle)

        test_dates.append(test_idx)

    # Convert to aligned Series
    bh_rets = pd.Series(bh_rets, index=test_dates, name="bh_return")
    mom_rets = pd.Series(mom_rets, index=test_dates, name="mom_return")
    mr_rets = pd.Series(mr_rets, index=test_dates, name="mr_return")
    ml_rets = pd.Series(ml_rets, index=test_dates, name="ml_binary_return")
    oracle_rets = pd.Series(oracle_rets, index=test_dates, name="oracle_return")
    decisions = pd.Series(decisions, index=test_dates, name="go_active_decision")
    probs_active = pd.Series(probs_active, index=test_dates, name="prob_go_active")

    # Equity curves (starting at 1.0)
    bh_equity = (1 + bh_rets).cumprod()
    mom_equity = (1 + mom_rets).cumprod()
    mr_equity = (1 + mr_rets).cumprod()
    ml_equity = (1 + ml_rets).cumprod()
    oracle_equity = (1 + oracle_rets).cumprod()

    return {
        "bh_rets": bh_rets,
        "mom_rets": mom_rets,
        "mr_rets": mr_rets,
        "ml_rets": ml_rets,
        "oracle_rets": oracle_rets,
        "bh_equity": bh_equity,
        "mom_equity": mom_equity,
        "mr_equity": mr_equity,
        "ml_equity": ml_equity,
        "oracle_equity": oracle_equity,
        "decisions": decisions,
        "probs_active": probs_active,
    }


# ---------------------------------------------------------------------
# 5) Performance statistics
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 6) MAIN
# ---------------------------------------------------------------------
def main():
    print(">>> backtesting_binary.py started")

    # 1) Load daily labeled data
    df_daily = load_daily_labeled_data()
    print("Daily labeled data shape:", df_daily.shape)

    # 2) Aggregate to monthly and build strategy returns
    monthly = build_monthly_frame(df_daily)
    print("Monthly frame shape:", monthly.shape)

    # 3) Build features (X) and BINARY target (BH vs Active)
    X, y = build_features_and_binary_target(monthly)
    print("Monthly features shape:", X.shape)
    print("Binary target distribution (0=stay BH, 1=go Active):")
    print(y.value_counts())

    # 4) Run rolling binary logistic backtest
    results = rolling_binary_backtest(
        monthly,
        X,
        y,
        min_train_size=60,
        prob_threshold=0.55,  # can be tuned
    )

    bh_rets = results["bh_rets"]
    mom_rets = results["mom_rets"]
    mr_rets = results["mr_rets"]
    ml_rets = results["ml_rets"]
    oracle_rets = results["oracle_rets"]

    # 5) Compute performance statistics
    bh_stats = compute_performance_stats(bh_rets)
    mom_stats = compute_performance_stats(mom_rets)
    mr_stats = compute_performance_stats(mr_rets)
    ml_stats = compute_performance_stats(ml_rets)
    oracle_stats = compute_performance_stats(oracle_rets)

    print("\n=== Performance summary (binary backtest: BH vs Active) ===")

    def fmt_stats(name: str, stats: Dict[str, float]):
        print(f"\n{name}:")
        print(f"  Total return:        {stats['total_return']:.2%}")
        print(f"  Annualized return:   {stats['annualized_return']:.2%}")
        print(f"  Annualized vol:      {stats['annualized_volatility']:.2%}")
        print(f"  Sharpe ratio (rf=0): {stats['sharpe_ratio']:.2f}")
        print(f"  Max drawdown:        {stats['max_drawdown']:.2%}")

    fmt_stats("Buy & Hold", bh_stats)
    fmt_stats("Momentum", mom_stats)
    fmt_stats("Mean Reversion", mr_stats)
    fmt_stats("ML Binary (BH vs Active)", ml_stats)
    fmt_stats("Oracle (best of 3, ex post)", oracle_stats)

    # 6) Decisions: share of months where the model goes Active
    decisions = results["decisions"]
    print("\nML Binary decisions (share of test months):")
    print(decisions.value_counts(normalize=True).rename(index={0: "Stay BH", 1: "Go Active"}))

    # 7) Show last 10 equity values
    print("\nLast 10 equity values:")
    equity_df = pd.concat(
        [
            results["bh_equity"],
            results["mom_equity"],
            results["mr_equity"],
            results["ml_equity"],
            results["oracle_equity"],
        ],
        axis=1,
    )
    equity_df.columns = [
        "BH",
        "Momentum",
        "Mean Reversion",
        "ML Binary",
        "Oracle",
    ]
    print(equity_df.tail(10))

    print(">>> backtesting_binary.py finished")


if __name__ == "__main__":
    main()
