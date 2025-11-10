import pandas as pd
from src.config import RAW_DATA_DIR


def load_price_data():
    """
    Load the raw price data CSV saved by data_download.py.
    Cleans extra header rows and ensures numeric columns.
    """
    filepath = RAW_DATA_DIR / "SPY_prices.csv"

    df = pd.read_csv(
        filepath,
        skiprows=2,  # skip the first two weird header rows
        header=None,
        names=["date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
        parse_dates=["date"],
        index_col="date",
    )

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop any row where index is not a valid date or where all values are NaN
    df = df[df.index.notnull()].dropna(how="all")

    return df


def main():
    df = load_price_data()

    print("Head of the dataset:")
    print(df.head())

    print("\nInfo:")
    print(df.info())

    print("\nSummary statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()



