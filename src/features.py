import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_price_data():
    """
    Load the cleaned price data from the raw folder.
    """
    filepath = RAW_DATA_DIR / "SPY_prices.csv"
    df = pd.read_csv(
        filepath,
        skiprows=2,
        names=["date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["date"],
        index_col="date",
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute several technical indicators based on the price data.
    """
    # --- Returns ---
    df["daily_return"] = df["Adj Close"].pct_change()

    # --- Rolling volatility (20 days) ---
    df["rolling_volatility"] = df["daily_return"].rolling(window=20).std()

    # --- Simple Moving Averages ---
    df["sma_20"] = df["Adj Close"].rolling(window=20).mean()
    df["sma_50"] = df["Adj Close"].rolling(window=50).mean()

    # --- Exponential Moving Average ---
    df["ema_20"] = df["Adj Close"].ewm(span=20, adjust=False).mean()

    # --- RSI (Relative Strength Index) ---
    delta = df["Adj Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Drop rows with missing values ---
    df = df.dropna()

    return df


def save_features(df: pd.DataFrame):
    """
    Save the resulting dataset with features to data/processed/.
    """
    filepath = PROCESSED_DATA_DIR / "features.csv"
    df.to_csv(filepath)
    print(f"Features saved to: {filepath}")


def main():
    df = load_price_data()
    df_features = compute_technical_indicators(df)
    save_features(df_features)
    print("Feature computation completed successfully!")


if __name__ == "__main__":
    main()
