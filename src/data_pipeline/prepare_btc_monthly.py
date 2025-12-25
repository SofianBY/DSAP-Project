# src/prepare_btc_dataset.py
"""
Prepare a labeled dataset for Bitcoin (BTC-USD).

Steps:
1. Load daily BTC prices from data/raw/BTC_prices.csv.
2. Compute technical features (returns, volatility, RSI, MACD, moving averages).
3. Aggregate features to monthly frequency.
4. Define three simple strategies (BH, Momentum, Mean Reversion).
5. For each month, label the "best" strategy (highest realized monthly return).
6. Save the final monthly dataset as data/processed/labeled_data_btc.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config_btc import RAW_PRICE_FILE
from src.config import PROCESSED_DATA_DIR


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI) for a price series.
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def load_btc_prices() -> pd.DataFrame:
    """
    Load BTC daily prices from RAW_PRICE_FILE.

    The CSV is assumed to come from yfinance, typically with a first column
    named 'Date' (or sometimes an unnamed index). We normalize this to a
    datetime index named 'date', and we coerce price columns to numeric.
    """
    print(f"Loading BTC prices from: {RAW_PRICE_FILE}")

    # Read raw CSV
    df = pd.read_csv(RAW_PRICE_FILE)

    if df.empty:
        raise ValueError(f"{RAW_PRICE_FILE} is empty.")

    # Detect date column (case insensitive)
    cols_lower = [c.lower() for c in df.columns]

    if "date" in cols_lower:
        # Column is 'date' or 'Date'
        date_col = df.columns[cols_lower.index("date")]
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # Fallback: assume first column is the date-like index (typical yfinance)
        first_col = df.columns[0]
        df["date"] = pd.to_datetime(df[first_col], errors="coerce")

    # Drop rows where date parsing failed, sort, and set index
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    # Expected numeric columns
    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in BTC price file: {missing}")

    # 🔹 Force all price/volume columns to numeric (float), coercing errors to NaN
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where Adj Close is NaN (essential for returns)
    df = df.dropna(subset=["Adj Close"])

    print(f"BTC price data shape: {df.shape}")
    return df


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily technical features for BTC.
    """
    df = df.copy()

    # Daily returns
    df["ret_1d"] = df["Adj Close"].pct_change()

    # Rolling volatility (21-day and 63-day)
    df["vol_21d"] = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"] = df["ret_1d"].rolling(63).std() * np.sqrt(252)

    # Moving averages
    df["sma_10"] = df["Adj Close"].rolling(10).mean()
    df["sma_21"] = df["Adj Close"].rolling(21).mean()
    df["sma_50"] = df["Adj Close"].rolling(50).mean()
    df["sma_200"] = df["Adj Close"].rolling(200).mean()

    # Price vs. moving averages
    df["px_over_sma_10"] = df["Adj Close"] / df["sma_10"] - 1
    df["px_over_sma_50"] = df["Adj Close"] / df["sma_50"] - 1
    df["px_over_sma_200"] = df["Adj Close"] / df["sma_200"] - 1

    # RSI and MACD
    df["rsi_14"] = compute_rsi(df["Adj Close"], window=14)

    ema_12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Rolling returns over different horizons
    df["ret_5d"] = df["Adj Close"].pct_change(5)
    df["ret_21d"] = df["Adj Close"].pct_change(21)
    df["ret_63d"] = df["Adj Close"].pct_change(63)

    # Volume-related feature
    df["vol_zscore_21d"] = (
        (df["Volume"] - df["Volume"].rolling(21).mean())
        / df["Volume"].rolling(21).std()
    )

    return df


def build_monthly_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily features to monthly frequency using end-of-month values.
    """
    feature_cols = [
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

    monthly_features = df_daily[feature_cols].resample("M").last()
    monthly_features = monthly_features.dropna()

    print(f"Monthly features shape: {monthly_features.shape}")
    return monthly_features


def build_strategy_labels(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a monthly DataFrame with realized returns for:
      - Buy & Hold
      - Momentum (invest if last month return > 0)
      - Mean Reversion (invest if last month return < 0)
    Then, for each month, label the best strategy.
    """
    # Monthly closing prices
    monthly_close = df_prices["Adj Close"].resample("M").last()
    monthly_ret = monthly_close.pct_change()

    # Buy & Hold: always invested
    bh_ret = monthly_ret.copy()

    # Momentum: invest this month if last month return > 0
    mom_signal = (monthly_ret.shift(1) > 0).astype(float)
    mom_ret = mom_signal * monthly_ret

    # Mean Reversion: invest this month if last month return < 0
    mr_signal = (monthly_ret.shift(1) < 0).astype(float)
    mr_ret = mr_signal * monthly_ret

    strat_df = pd.DataFrame(
        {
            "bh_ret": bh_ret,
            "mom_ret": mom_ret,
            "mr_ret": mr_ret,
        }
    ).dropna()

    # Determine best strategy each month
    strat_returns = np.vstack(
        [strat_df["bh_ret"].values, strat_df["mom_ret"].values, strat_df["mr_ret"].values]
    ).T  # shape (n_months, 3)

    best_idx = np.argmax(strat_returns, axis=1)

    # Map 0/1/2 -> strategy name
    idx_to_name = {0: "Buy & Hold", 1: "Momentum", 2: "Mean Reversion"}
    best_names = [idx_to_name[i] for i in best_idx]

    labels = strat_df.copy()
    labels["best_strategy"] = best_names

    print(f"Label distribution (BTC):")
    print(labels["best_strategy"].value_counts())

    return labels[["bh_ret", "mom_ret", "mr_ret", "best_strategy"]]


def build_labeled_monthly_btc() -> pd.DataFrame:
    """
    End-to-end construction of monthly BTC dataset with features and labels.
    """
    # 1. Load daily prices
    prices = load_btc_prices()

    # 2. Daily features
    daily_features = build_daily_features(prices)

    # 3. Monthly features
    monthly_features = build_monthly_features(daily_features)

    # 4. Strategy labels
    monthly_labels = build_strategy_labels(prices)

    # Align features and labels on common monthly index
    common_index = monthly_features.index.intersection(monthly_labels.index)
    monthly_features = monthly_features.loc[common_index]
    monthly_labels = monthly_labels.loc[common_index]

    df_monthly = monthly_features.copy()
    df_monthly["best_strategy"] = monthly_labels["best_strategy"]

    # Reset index with a 'date' column for modeling
    df_monthly = df_monthly.reset_index().rename(columns={"index": "date", "date": "date"})

    print(f"Final BTC labeled dataset shape: {df_monthly.shape}")
    return df_monthly


def main():
    print(">>> prepare_btc_dataset.py started")

    df_monthly = build_labeled_monthly_btc()

    output_path = PROCESSED_DATA_DIR / "labeled_data_btc.csv"
    df_monthly.to_csv(output_path, index=False)
    print(f"Saved BTC labeled dataset to: {output_path}")

    print(">>> prepare_btc_dataset.py finished")


if __name__ == "__main__":
    main()

