# src/data_download_btc.py
"""
Download BTC-USD daily prices from Yahoo Finance and save them to data/raw/BTC_prices.csv
without touching the existing SP500 data.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import RAW_DATA_DIR
from src.config_btc import TICKER, START_DATE, END_DATE


def download_btc_price_data() -> pd.DataFrame:
    """
    Download BTC-USD prices from Yahoo Finance.

    Columns: Open, High, Low, Close, Adj Close, Volume
    Index: Datetime (sorted, named 'date')
    """
    print(f"Downloading data for {TICKER} from {START_DATE} to {END_DATE or 'today'} ...")

    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=True,
    )

    # Clean index
    df = df.dropna(how="all").sort_index()
    df.index.name = "date"

    print(f"BTC data successfully downloaded: {len(df)} rows.")
    return df


def save_btc_price_data(df: pd.DataFrame) -> Path:
    """
    Save BTC prices to data/raw/BTC_prices.csv
    (we do NOT overwrite SPY_prices.csv here).
    """
    filepath = RAW_DATA_DIR / "BTC_prices.csv"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(f"BTC data saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    btc_df = download_btc_price_data()
    save_btc_price_data(btc_df)
