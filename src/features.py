import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_price_data() -> pd.DataFrame:
    """
    Load the cleaned price data from the raw folder.

    We keep the original CSV structure:
      - source: data/raw/SPY_prices.csv
      - index: DatetimeIndex named 'date'
    """
    filepath = RAW_DATA_DIR / "SPY_prices.csv"
    df = pd.read_csv(
        filepath,
        skiprows=2,
        names=["date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["date"],
        index_col="date",
    )

    # Ensure numeric columns and drop fully empty rows
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a richer set of daily technical indicators.

    Idea:
    - Keep your original indicators (for continuity).
    - Add more short-term / medium-term signals that the ML model can use to
      distinguish between regimes where Momentum or Mean Reversion works better.

    All indicators are computed at DAILY frequency.
    They will later be aggregated at the MONTHLY level in the rest of the pipeline.
    """
    df = df.copy()

    # === 1) Basic daily returns ===
    df["daily_return"] = df["Adj Close"].pct_change()

    # === 2) Rolling volatility ===
    # Original 20-day volatility (kept for compatibility)
    df["rolling_volatility_20d"] = df["daily_return"].rolling(window=20).std()

    # New: shorter and longer horizons
    df["rolling_volatility_10d"] = df["daily_return"].rolling(window=10).std()
    df["rolling_volatility_60d"] = df["daily_return"].rolling(window=60).std()

    # === 3) Moving averages and price vs. trend ===
    df["sma_20"] = df["Adj Close"].rolling(window=20).mean()
    df["sma_50"] = df["Adj Close"].rolling(window=50).mean()
    df["sma_200"] = df["Adj Close"].rolling(window=200).mean()

    df["ema_20"] = df["Adj Close"].ewm(span=20, adjust=False).mean()

    # Distance to moving averages (trend strength)
    df["price_over_sma20"] = df["Adj Close"] / df["sma_20"] - 1.0
    df["price_over_sma50"] = df["Adj Close"] / df["sma_50"] - 1.0
    df["price_over_sma200"] = df["Adj Close"] / df["sma_200"] - 1.0

    # === 4) Momentum over different horizons ===
    # 1-month (~20 trading days) and 3-month (~60 days) momentum
    df["mom_20d"] = df["Adj Close"] / df["Adj Close"].shift(20) - 1.0
    df["mom_60d"] = df["Adj Close"] / df["Adj Close"].shift(60) - 1.0

    # Simple weekly return (5 trading days)
    df["weekly_return_5d"] = df["Adj Close"] / df["Adj Close"].shift(5) - 1.0

    # === 5) RSI (Relative Strength Index) ===
    delta = df["Adj Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    window_rsi = 14
    avg_gain = pd.Series(gain, index=df.index).rolling(window=window_rsi).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=window_rsi).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # === 6) Intraday range and short-term volatility proxy ===
    # High-Low range scaled by close: captures intraday uncertainty.
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]

    # 10-day average intraday range = another volatility style indicator
    df["intraday_range_10d_ma"] = df["intraday_range"].rolling(window=10).mean()

    # === 7) Drawdown and max drawdown (short horizon) ===
    rolling_max_60d = df["Adj Close"].rolling(window=60, min_periods=1).max()
    df["drawdown_60d"] = df["Adj Close"] / rolling_max_60d - 1.0

    # === 8) Clean up: drop initial rows with many NaNs ===
    min_lookback = 200  # because of sma_200 and mom_60d
    df = df.iloc[min_lookback:].copy()

    # Also drop any remaining rows where key indicators are missing
    key_cols = [
        "daily_return",
        "rolling_volatility_10d",
        "rolling_volatility_20d",
        "rolling_volatility_60d",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_20",
        "price_over_sma20",
        "price_over_sma50",
        "price_over_sma200",
        "mom_20d",
        "mom_60d",
        "weekly_return_5d",
        "rsi_14",
        "intraday_range",
        "intraday_range_10d_ma",
        "drawdown_60d",
    ]
    df = df.dropna(subset=key_cols)

    return df


def save_features(df: pd.DataFrame) -> None:
    """
    Save the resulting dataset with features to data/processed/.
    """
    filepath = PROCESSED_DATA_DIR / "features.csv"
    df.to_csv(filepath)
    print(f"Features saved to: {filepath}")


def main() -> None:
    print(">>> features.py started")

    df_prices = load_price_data()
    print(f"Raw price data shape: {df_prices.shape}")

    df_features = compute_technical_indicators(df_prices)
    print(f"Feature dataframe shape: {df_features.shape}")

    save_features(df_features)
    print("Feature computation completed successfully!")
    print(">>> features.py finished")


if __name__ == "__main__":
    main()

