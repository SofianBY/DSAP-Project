# src/backtesting.py

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

    - Use the last available daily row of each month (resample('M').last()).
    - Compute monthly SPY returns (based on Adj Close).
    - Build 3 strategy returns:
        * bh_return: buy & hold (always invested)
        * mom_return: simple 1M momentum (invest if previous month_return > 0)
        * mr_return: simple 1M mean reversion (invest if previous month_return < 0)
    """
    # Monthly snapshot: last row of each calendar month
    monthly = df_daily.resample("M").last()

    if "Adj Close" not in monthly.columns:
        raise ValueError("Adj Close column is required to compute monthly returns.")

    # Monthly return based on end-of-month Adj Close
    monthly["month_return"] = monthly["Adj Close"].pct_change()

    # Buy & hold: always invested
    monthly["bh_return"] = monthly["month_return"]

    # Momentum: invest this month if previous month_return > 0
    prev_ret = monthly["month_return"].shift(1)
    mom_signal = (prev_ret > 0).astype(float)
    monthly["mom_signal"] = mom_signal
    monthly["mom_return"] = mom_signal * monthly["month_return"]

    # Mean reversion: invest this month if previous month_return < 0
    mr_signal = (prev_ret < 0).astype(float)
    monthly["mr_signal"] = mr_signal
    monthly["mr_return"] = mr_signal * monthly["month_return"]

    # Drop early rows where returns are NaN
    monthly = monthly.dropna(subset=["month_return", "bh_return", "mom_return", "mr_return"])

    return monthly


def build_features_and_target(monthly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and multi-class target y.

    We want the model to decide at the beginning of month t which of the three
    strategies will be best during month t.

    To avoid look-ahead:
      - Features X_t are taken from month t-1 (state of the market at the end of month t-1).
      - Target y_t is defined using realized returns of month t:
            y_t = argmax(bh_return_t, mom_return_t, mr_return_t)
             0  -> Buy & Hold is best
             1  -> Momentum is best
             2  -> Mean Reversion is best

    We then align X and y on the same time index.
    """
    # Build target using realized returns of the 3 strategies
    returns_df = monthly[["bh_return", "mom_return", "mr_return"]].copy()
    # Argmax over columns axis -> 0=bh, 1=mom, 2=mr
    best_idx = returns_df.values.argmax(axis=1)
    y = pd.Series(best_idx, index=monthly.index, name="best_strategy_3class")

    # Features: all numeric columns except returns and signals
    drop_cols = [
        "month_return", "bh_return", "mom_return", "mr_return",
        "mom_signal", "mr_signal"
    ]
    feature_cols = [
        c for c in monthly.columns
        if pd.api.types.is_numeric_dtype(monthly[c]) and c not in drop_cols
    ]

    # Raw features at time t
    X_raw = monthly[feature_cols]

    # Use lagged features: information available at the end of t-1
    X_lagged = X_raw.shift(1)

    # Align X and y: drop rows with NaNs in X or y
    data = pd.concat([X_lagged, y], axis=1).dropna()
    X = data[feature_cols]
    y_aligned = data["best_strategy_3class"]

    return X, y_aligned


def rolling_multiclass_backtest(
    monthly: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int = 60,
    proba_threshold: float = 0.55,
    proba_margin: float = 0.05,
) -> Dict[str, pd.Series]:
    """
    Perform an expanding-window backtest with a multinomial Logistic Regression classifier.

    At each step t:
      - Train on all months before t (at least min_train_size).
      - Predict class probabilities for month t:
            0 -> Buy & Hold
            1 -> Momentum
            2 -> Mean Reversion

    Decision rule (Buy & Hold by default):
      - Let p_bh, p_mom, p_mr be the predicted probabilities.
      - Let p_alt = max(p_mom, p_mr).

      - If p_alt < proba_threshold OR (p_alt - p_bh) < proba_margin:
            -> stay in Buy & Hold for month t.
      - Else:
            -> switch to the active strategy (Momentum or Mean Reversion)
               with the highest probability.

    We also compute an "Oracle" strategy:
      - For each test month, Oracle chooses ex post the best of the three
        strategy returns (BH, Momentum, Mean Reversion).
      - This is not tradable but provides an upper bound for performance.

    Returns:
      A dictionary with:
        - bh_rets, mom_rets, mr_rets, ml_rets, oracle_rets (monthly returns)
        - bh_equity, mom_equity, mr_equity, ml_equity, oracle_equity (equity curves)
        - chosen_strategy: Series of 0/1/2 indicating the ML choice at each test month
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
    chosen_strategies = []
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

        # Multinomial Logistic Regression with class balancing
        # multi_class="multinomial" is the default for recent sklearn versions
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        clf.fit(X_train_scaled, y_train)

        # Predicted probabilities for the three classes
        proba = clf.predict_proba(X_test_scaled)[0]
        p_bh = proba[0]   # class 0 -> Buy & Hold
        p_mom = proba[1]  # class 1 -> Momentum
        p_mr = proba[2]   # class 2 -> Mean Reversion

        # Realized returns of the 3 strategies for the test month
        ret_bh = monthly.loc[test_idx, "bh_return"]
        ret_mom = monthly.loc[test_idx, "mom_return"]
        ret_mr = monthly.loc[test_idx, "mr_return"]

        bh_rets.append(ret_bh)
        mom_rets.append(ret_mom)
        mr_rets.append(ret_mr)

        # Oracle: ex post best of the three for this month (not tradable)
        ret_oracle = max(ret_bh, ret_mom, ret_mr)
        oracle_rets.append(ret_oracle)

        # === Buy & Hold by default decision rule ===
        p_alt = max(p_mom, p_mr)

        # Case 1: not enough confidence -> stay in Buy & Hold
        if (p_alt < proba_threshold) or ((p_alt - p_bh) < proba_margin):
            ml_rets.append(ret_bh)
            chosen_strategies.append(0)  # 0 = BH
        # Case 2: strong active signal -> choose Momentum or Mean Reversion
        else:
            if p_mom >= p_mr:
                ml_rets.append(ret_mom)
                chosen_strategies.append(1)  # 1 = Momentum
            else:
                ml_rets.append(ret_mr)
                chosen_strategies.append(2)  # 2 = Mean Reversion

        test_dates.append(test_idx)

    # Convert to aligned Series
    bh_rets = pd.Series(bh_rets, index=test_dates, name="bh_return")
    mom_rets = pd.Series(mom_rets, index=test_dates, name="mom_return")
    mr_rets = pd.Series(mr_rets, index=test_dates, name="mr_return")
    ml_rets = pd.Series(ml_rets, index=test_dates, name="ml_return")
    oracle_rets = pd.Series(oracle_rets, index=test_dates, name="oracle_return")
    chosen_strategies = pd.Series(chosen_strategies, index=test_dates, name="chosen_strategy")

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
        "chosen_strategy": chosen_strategies,
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

    # 3) Build features (X) and multi-class target (3 strategies)
    X, y = build_features_and_target(monthly)
    print("Monthly features shape:", X.shape)
    print("Multi-class target distribution (0=BH, 1=Mom, 2=MR):")
    print(y.value_counts())

    # 4) Run rolling multinomial logistic backtest
    results = rolling_multiclass_backtest(
        monthly,
        X,
        y,
        min_train_size=60,
        proba_threshold=0.55,
        proba_margin=0.05,
    )

    bh_rets = results["bh_rets"]
    mom_rets = results["mom_rets"]
    mr_rets = results["mr_rets"]
    ml_rets = results["ml_rets"]
    oracle_rets = results["oracle_rets"]
    chosen = results["chosen_strategy"]

    # 5) Compute performance statistics
    bh_stats = compute_performance_stats(bh_rets)
    mom_stats = compute_performance_stats(mom_rets)
    mr_stats = compute_performance_stats(mr_rets)
    ml_stats = compute_performance_stats(ml_rets)
    oracle_stats = compute_performance_stats(oracle_rets)

    print("\n=== Performance summary (monthly backtest: 3 strategies + ML + Oracle) ===")

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
    fmt_stats("ML Adaptive (BH by default)", ml_stats)
    fmt_stats("Oracle (best of 3, ex post)", oracle_stats)

    # 6) How often does ML choose each strategy?
    mapping = {0: "Buy & Hold", 1: "Momentum", 2: "Mean Reversion"}
    print("\nML Adaptive decisions (share of test months):")
    print(
        chosen.map(mapping)
        .value_counts(normalize=True)
        .rename("proportion")
        .round(3)
    )

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
    equity_df.columns = ["BH", "Momentum", "Mean Reversion", "ML Adaptive", "Oracle"]
    print(equity_df.tail(10))

    print(">>> backtesting.py finished")


if __name__ == "__main__":
    main()
