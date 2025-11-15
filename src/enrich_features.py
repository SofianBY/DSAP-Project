# src/enrich_features.py

import numpy as np
import pandas as pd
import yfinance as yf

from src import config


def load_labeled_data() -> pd.DataFrame:
    """
    Load labeled_data.csv and set a proper DatetimeIndex on 'date'.
    """
    path = config.PROCESSED_DATA_DIR / "labeled_data.csv"
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    else:
        raise ValueError(f"No 'date' column found. Columns: {df.columns}")

    return df


def download_macro_series(start: str, end: str) -> pd.DataFrame:
    """
    Download VIX (^VIX) and US 10Y yield (^TNX) from Yahoo Finance
    in a robust way, ensuring we extract adjusted close prices properly.
    """
    tickers = ["^VIX", "^TNX"]

    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    # Case 1: MultiIndex columns (typical when downloading >1 ticker)
    # e.g. ('Adj Close', '^VIX'), ('Adj Close', '^TNX')
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            data = data["Adj Close"]  # keep only Adj Close layer
        else:
            raise ValueError(f"Expected 'Adj Close' level in MultiIndex: {data.columns}")

        # Flatten if needed
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(0, axis=1)

    # Case 2: SingleIndex columns (rare if yfinance changes API)
    else:
        possible_cols = [c for c in data.columns if "Adj Close" in c or c == "Close"]
        if len(possible_cols) == 0:
            raise ValueError(f"No 'Adj Close' or 'Close' in columns: {data.columns}")
        data = data[possible_cols]
        # Rename generically
        data.columns = ["VIX", "TNX"][: len(data.columns)]

    # Rename to standard names
    data = data.rename(columns={
        "^VIX": "VIX",
        "^TNX": "TNX"
    })

    # Fill missing values and ensure chronological order
    data = data.sort_index().ffill()

    return data


def add_vix_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Add VIX-based features: level, 5-day change, 20-day moving average,
    and 20-day volatility of the VIX.
    """
    vix = macro["VIX"].reindex(df.index).ffill()

    df["vix_level"] = vix
    df["vix_5d_change"] = vix.pct_change(5)
    df["vix_20d_ma"] = vix.rolling(20).mean()
    df["vix_20d_vol"] = vix.rolling(20).std()

    return df


def add_rate_features(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Add interest rate features (US 10Y).
    TNX is quoted approximately as yield * 10, so we divide by 10 to get %.
    """
    tnx = macro["TNX"].reindex(df.index).ffill()

    # Level of 10Y yield in %
    df["rate_10y"] = tnx / 10.0
    # 20-day change in the 10Y yield
    df["rate_10y_20d_change"] = df["rate_10y"].diff(20)

    return df


def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MACD, MACD signal, and MACD histogram based on Adj Close.
    Classic parameters: 12/26/9.
    """
    if "Adj Close" not in df.columns:
        raise ValueError("Adj Close column not found in df for MACD computation.")

    price = df["Adj Close"]

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    return df


def add_breadth_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple market "breadth" proxy:
    fraction of positive daily returns within the current month.
    """
    if "daily_return" not in df.columns:
        raise ValueError("daily_return column not found for breadth proxy.")

    positive = (df["daily_return"] > 0).astype(float)
    month = df.index.to_period("M")

    # For each day, attach the fraction of positive days in its month
    month_pos_frac = positive.groupby(month).transform("mean")

    df["month_pos_frac"] = month_pos_frac.values

    return df


def main():
    print(">>> enrich_features.py started")

    # 1) Load existing labeled_data
    df = load_labeled_data()
    print("Original labeled_data shape:", df.shape)

    # 2) Download macro series (VIX + TNX) over the same period
    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    print(f"Downloading macro series from {start} to {end}...")
    macro = download_macro_series(start, end)
    print("Macro series shape:", macro.shape)

    # 3) Add macro features
    df = add_vix_features(df, macro)
    df = add_rate_features(df, macro)

    # 4) Add MACD features
    df = add_macd_features(df)

    # 5) Add breadth proxy
    df = add_breadth_proxy(df)

    # 6) Save enriched labeled_data.csv (overwrite)
    out_path = config.PROCESSED_DATA_DIR / "labeled_data.csv"
    df.reset_index().rename(columns={"index": "date"}).to_csv(out_path, index=False)
    print(f"Saved enriched labeled_data to: {out_path}")
    print("New shape:", df.shape)

    print(">>> enrich_features.py finished")


if __name__ == "__main__":
    main()

