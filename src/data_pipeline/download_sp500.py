import yfinance as yf
import pandas as pd
from src.config import RAW_DATA_DIR, TICKER, START_DATE, END_DATE, PRICE_FREQUENCY



def download_price_data():
    """
    Download market price data from Yahoo Finance and return a clean DataFrame.

    Columns: Open, High, Low, Close, Adj Close, Volume
    Index: Datetime (sorted)
    """
    print(f"Downloading data for {TICKER} from {START_DATE} to {END_DATE or 'today'} ...")
    
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval=PRICE_FREQUENCY,
        auto_adjust=False,
        progress=True,
    )

    # Drop missing rows and ensure ascending index
    df = df.dropna(how="all").sort_index()
    df.index.name = "date"

    print(f"Data successfully downloaded: {len(df)} rows.")
    return df


def save_price_data(df):
    """
    Save the DataFrame as a CSV file inside data/raw/.
    """
    filepath = RAW_DATA_DIR / f"{TICKER}_prices.csv"
    df.to_csv(filepath)
    print(f"Data saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    df = download_price_data()
    save_price_data(df)

