"""
backtesting.py

Monthly backtest for:
  - Buy & Hold
  - Momentum
  - Mean Reversion
  - ML Adaptive strategy (multi-class: choose between the 3)
  - Oracle (best ex post among the 3)

The time horizon remains MONTHLY (as in the project proposal),
but we build RICHER MONTHLY FEATURES by aggregating daily data,
including technical indicators and macro variables.
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class StrategyStats:
    """Container for performance metrics of one strategy."""

    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float


def load_labeled_daily_data(path: str = "data/processed/labeled_data.csv") -> pd.DataFrame:
    """
    Load the daily labeled dataset produced by:
      - features.py
      - labeling.py
      - enrich_features.py

    In practice, our CSV has the date as the first column (index).
    We read that first column as the index and convert it to datetime.
    """
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        raise ValueError(
            "Could not parse the index of labeled_data.csv as dates. "
            "Check how the file is saved in labeling.py / enrich_features.py."
        ) from e
    df.index.name = "Date"
    return df.sort_index()


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Build an equity curve from a series of periodic returns."""
    r = returns.fillna(0.0)
    equity = (1.0 + r).cumprod()
    return initial * equity


def compute_performance_stats(returns: pd.Series, freq: int = 12) -> StrategyStats:
    """
    Compute standard performance metrics from a return series.

    Args:
        returns: periodic returns (here, monthly).
        freq: number of periods per year (12 for monthly).

    Metrics:
        - total_return
        - annualized_return
        - annualized_vol
        - sharpe (rf = 0)
        - max_drawdown
    """
    r = returns.dropna()
    if r.empty:
        return StrategyStats(
            total_return=np.nan,
            annualized_return=np.nan,
            annualized_vol=np.nan,
            sharpe=np.nan,
            max_drawdown=np.nan,
        )

    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0

    n_periods = len(r)
    annualized_return = (1.0 + total_return) ** (freq / n_periods) - 1.0

    annualized_vol = r.std() * np.sqrt(freq)

    sharpe = np.nan
    if annualized_vol > 0:
        sharpe = annualized_return / annualized_vol

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan

    return StrategyStats(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        annualized_vol=float(annualized_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
    )


def build_monthly_frame(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the daily labeled dataset to a richer monthly level.

    Steps:
      1) Attach a month period to each daily row.
      2) For each month, compute:
           - realized return statistics (mean, vol, skew, kurtosis),
           - fraction of positive days,
           - average technical indicators (RSI, MACD, etc.),
           - macro averages (VIX, 10Y rate) and their monthly change.
      3) Keep ONE row per month: the last calendar day of each month,
         carrying all these aggregated features.
      4) Compute the 3 strategy returns at monthly frequency:
           * bh_return   : buy & hold (always invested)
           * mom_return  : momentum (invest if previous month_return > 0)
           * mr_return   : mean reversion (invest if previous month_return < 0)
    """
    df = df_daily.copy()

    if "Adj Close" not in df.columns:
        raise ValueError("Adj Close column is required to compute monthly returns.")

    if "daily_return" not in df.columns:
        df["daily_return"] = df["Adj Close"].pct_change()

    df["month"] = df.index.to_period("M")
    group = df.groupby("month")

    # Return distribution inside the month
    df["month_ret_mean"] = group["daily_return"].transform("mean")
    df["month_ret_vol"] = group["daily_return"].transform("std")
    df["month_ret_skew"] = group["daily_return"].transform(lambda x: x.skew())
    df["month_ret_kurt"] = group["daily_return"].transform(lambda x: x.kurt())
    df["month_pos_frac"] = group["daily_return"].transform(lambda x: (x > 0).mean())

    # Macro features (if present)
    if "vix_level" in df.columns:
        df["vix_month_level"] = group["vix_level"].transform("mean")
        df["vix_month_trend"] = group["vix_level"].transform(lambda x: x.iloc[-1] - x.iloc[0])

    if "rate_10y" in df.columns:
        df["rate_10y_month_level"] = group["rate_10y"].transform("mean")
        df["rate_10y_month_change"] = group["rate_10y"].transform(lambda x: x.iloc[-1] - x.iloc[0])

    # Technical indicators aggregated over the month (if present)
    tech_cols = [
        "rsi_14",
        "rolling_volatility",
        "sma_20",
        "sma_50",
        "ema_20",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    for col in tech_cols:
        if col in df.columns:
            df[f"{col}_month_mean"] = group[col].transform("mean")

    # Reduce to one row per month (last daily row of the month)
    monthly = group.tail(1).copy()
    monthly.index = monthly["month"].dt.to_timestamp("M")

    # Compute realized monthly return
    monthly["month_return"] = monthly["Adj Close"].pct_change()

    # Strategy returns:
    # Buy & Hold
    monthly["bh_return"] = monthly["month_return"]

    # Momentum: invest this month if previous month_return > 0
    prev_ret = monthly["month_return"].shift(1)
    monthly["mom_signal"] = (prev_ret > 0).astype(float)
    monthly["mom_return"] = monthly["mom_signal"] * monthly["month_return"]

    # Mean Reversion: invest this month if previous month_return < 0
    monthly["mr_signal"] = (prev_ret < 0).astype(float)
    monthly["mr_return"] = monthly["mr_signal"] * monthly["month_return"]

    # Drop rows where strategy returns are NaN (first month)
    monthly = monthly.dropna(subset=["month_return", "bh_return", "mom_return", "mr_return"])

    monthly = monthly.drop(columns=["month"])

    return monthly


def prepare_features_and_target(
    monthly: pd.DataFrame, target_col: str = "best_strategy_3class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and target vector y from the monthly frame.

    We explicitly exclude:
      - target columns
      - realized returns of the strategies
      - any equity curves (if present)
      - raw price columns that are not needed as features
    """
    if target_col not in monthly.columns:
        raise ValueError(f"Target column '{target_col}' not found in monthly frame.")

    cols_to_exclude = {
        target_col,
        "best_strategy",
        "month_return",
        "bh_return",
        "mom_return",
        "mr_return",
        "bh_equity",
        "momentum_equity",
        "mean_reversion_equity",
        "ml_adaptive_equity",
        "oracle_equity",
        "Adj Close",
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
    }

    feature_cols = [c for c in monthly.columns if c not in cols_to_exclude]

    X = monthly[feature_cols].copy()
    y = monthly[target_col].copy()

    # Drop rows where the target is NaN
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y.astype(int)


def run_backtest(return_details: bool = False) -> Dict[str, Any]:
    """
    Run a MONTHLY backtest with:
      - Buy & Hold
      - Momentum
      - Mean Reversion
      - ML Adaptive (multi-class)
      - Oracle (best ex post)

    Target definition (no look-ahead within the same month):
      For each month t, the label is the strategy that achieves
      the highest return in month t+1:
        * 0 = Buy & Hold
        * 1 = Momentum
        * 2 = Mean Reversion
    """
    # 1) Load daily labeled dataset (technical + macro already merged)
    df_daily = load_labeled_daily_data()
    print(f"Daily labeled data shape: {df_daily.shape}")

    # 2) Build richer monthly frame
    monthly = build_monthly_frame(df_daily)
    print(f"Monthly frame shape: {monthly.shape}")

    # 3) Construct the monthly multi-class label from NEXT month's strategy returns
    future = pd.DataFrame(
        {
            "bh": monthly["bh_return"].shift(-1),
            "mom": monthly["mom_return"].shift(-1),
            "mr": monthly["mr_return"].shift(-1),
        },
        index=monthly.index,
    )
    best_future = future.idxmax(axis=1)
    mapping = {"bh": 0, "mom": 1, "mr": 2}
    monthly["best_strategy_3class"] = best_future.map(mapping)

    # 4) Prepare features and target
    X, y = prepare_features_and_target(monthly, target_col="best_strategy_3class")
    print(f"Monthly features shape: {X.shape}")
    print("Multi-class target distribution (0=BH, 1=Mom, 2=MR):")
    print(y.value_counts().sort_index().rename("count"))
    print()

    # 5) ML pipeline: Logistic Regression with class_balance
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="auto",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # 6) Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    ml_returns = pd.Series(index=X.index, dtype=float)
    ml_chosen_class = pd.Series(index=X.index, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        test_dates = X_test.index
        fold_ml_ret = []
        fold_chosen_class = []

        for date, pred_class in zip(test_dates, y_pred):
            if pred_class == 0:
                r = monthly.loc[date, "bh_return"]
            elif pred_class == 1:
                r = monthly.loc[date, "mom_return"]
            elif pred_class == 2:
                r = monthly.loc[date, "mr_return"]
            else:
                r = monthly.loc[date, "bh_return"]
            fold_ml_ret.append(r)
            fold_chosen_class.append(pred_class)

        ml_returns.loc[test_dates] = fold_ml_ret
        ml_chosen_class.loc[test_dates] = fold_chosen_class

    ml_returns = ml_returns.fillna(0.0)

    # 7) Build equity curves
    bh_equity = equity_curve(monthly["bh_return"])
    mom_equity = equity_curve(monthly["mom_return"])
    mr_equity = equity_curve(monthly["mr_return"])
    ml_equity = equity_curve(ml_returns)

    oracle_returns = monthly[["bh_return", "mom_return", "mr_return"]].max(axis=1)
    oracle_equity = equity_curve(oracle_returns)

    monthly["bh_equity"] = bh_equity
    monthly["momentum_equity"] = mom_equity
    monthly["mean_reversion_equity"] = mr_equity
    monthly["ml_adaptive_equity"] = ml_equity
    monthly["oracle_equity"] = oracle_equity

    # 8) Performance statistics
    stats: Dict[str, StrategyStats] = {
        "Buy & Hold": compute_performance_stats(monthly["bh_return"]),
        "Momentum": compute_performance_stats(monthly["mom_return"]),
        "Mean Reversion": compute_performance_stats(monthly["mr_return"]),
        "ML Adaptive (BH by default)": compute_performance_stats(ml_returns),
        "Oracle (best of 3, ex post)": compute_performance_stats(oracle_returns),
    }

    print("=== Performance summary (monthly backtest: 3 strategies + ML + Oracle) ===\n")
    for name, s in stats.items():
        print(f"{name}:")
        print(f"  Total return:        {s.total_return * 100:8.2f}%")
        print(f"  Annualized return:   {s.annualized_return * 100:8.2f}%")
        print(f"  Annualized vol:      {s.annualized_vol * 100:8.2f}%")
        print(f"  Sharpe ratio (rf=0): {s.sharpe:8.2f}")
        print(f"  Max drawdown:        {s.max_drawdown * 100:8.2f}%")
        print()

    # 9) ML decision statistics
    ml_decisions = ml_chosen_class.dropna().astype(int)
    decision_counts = ml_decisions.value_counts(normalize=True).sort_index()
    decision_map = {0: "Buy & Hold", 1: "Momentum", 2: "Mean Reversion"}
    decision_label_index = decision_counts.index.map(decision_map)

    print("ML Adaptive decisions (share of test months):")
    decision_series = pd.Series(
        decision_counts.values,
        index=pd.Index(decision_label_index, name="chosen_strategy"),
        name="proportion",
    )
    print(decision_series)
    print()

    # 10) Last 10 equity values
    equity_curves = pd.DataFrame(
        {
            "BH": bh_equity,
            "Momentum": mom_equity,
            "Mean Reversion": mr_equity,
            "ML Adaptive": ml_equity,
            "Oracle": oracle_equity,
        },
        index=monthly.index,
    )

    print("Last 10 equity values:")
    print(equity_curves.tail(10))
    print()

    results: Dict[str, Any] = {
        "performance_summary": pd.DataFrame(
            [
                {
                    "strategy": name,
                    "total_return": s.total_return,
                    "annualized_return": s.annualized_return,
                    "annualized_vol": s.annualized_vol,
                    "sharpe": s.sharpe,
                    "max_drawdown": s.max_drawdown,
                }
                for name, s in stats.items()
            ]
        ).set_index("strategy"),
        "equity_curves": equity_curves,
        "monthly_returns": pd.DataFrame(
            {
                "BH": monthly["bh_return"],
                "Momentum": monthly["mom_return"],
                "Mean Reversion": monthly["mr_return"],
                "ML Adaptive": ml_returns,
                "Oracle": oracle_returns,
            },
            index=monthly.index,
        ),
        "ml_decisions": decision_series,
        "monthly": monthly,
    }

    if return_details:
        return results
    else:
        return {
            "performance_summary": results["performance_summary"],
            "ml_decisions": results["ml_decisions"],
        }


if __name__ == "__main__":
    print(">>> backtesting.py started")
    _ = run_backtest(return_details=False)
    print(">>> backtesting.py finished")
