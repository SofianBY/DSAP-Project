import pandas as pd
from src.config import PROCESSED_DATA_DIR


def load_features() -> pd.DataFrame:
    """
    Load the processed dataset with technical indicators.
    """
    filepath = PROCESSED_DATA_DIR / "features.csv"
    df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    return df


def create_monthly_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data monthly and compute the best-performing strategy label.

    Strategies:
    - buy_hold: monthly return of Adj Close
    - momentum: invest only if previous month return > 0
    - mean_reversion: invest only if previous month return < 0

    The label 'best_strategy' corresponds to the strategy that
    achieves the highest return in the NEXT month.
    """

    # 1) Compute monthly returns based on month-end prices
    monthly_prices = df["Adj Close"].resample("ME").last()
    monthly_returns = monthly_prices.pct_change()

    # 2) Prepare a DataFrame to store strategy returns
    strategy_returns = pd.DataFrame(index=monthly_returns.index)
    strategy_returns["buy_hold"] = monthly_returns

    # Previous month return
    prev_ret = monthly_returns.shift(1)

    # 3) Momentum strategy: invest this month if previous month was positive
    #    -> if prev_ret > 0, take monthly_returns, else 0
    strategy_returns["momentum"] = monthly_returns.where(prev_ret > 0, 0.0)

    # 4) Mean reversion: invest this month if previous month was negative
    strategy_returns["mean_reversion"] = monthly_returns.where(prev_ret < 0, 0.0)

    # 5) For each month, look at the NEXT month and see which strategy wins
    future_strategy_returns = strategy_returns[["buy_hold", "momentum", "mean_reversion"]].shift(-1)
    strategy_returns["best_strategy"] = future_strategy_returns.idxmax(axis=1)

    # Keep only months where the next month exists (i.e. drop last NaN)
    strategy_returns = strategy_returns.dropna(subset=["best_strategy"])

    # 6) Map monthly labels back to daily data
    monthly_labels = strategy_returns["best_strategy"]

    # Create a mapping from month period -> label
    label_map = {
        month.to_period("M"): label
        for month, label in monthly_labels.items()
    }

    # For each daily date, get its month and map to the corresponding label
    df_labeled = df.copy()
    df_labeled["month_period"] = df_labeled.index.to_period("M")
    df_labeled["best_strategy"] = df_labeled["month_period"].map(label_map)

    # We can drop the helper column
    df_labeled = df_labeled.drop(columns=["month_period"])

    return df_labeled


def save_labeled_data(df: pd.DataFrame) -> None:
    """
    Save the dataset with labels to data/processed/.
    """
    filepath = PROCESSED_DATA_DIR / "labeled_data.csv"
    df.to_csv(filepath)
    print(f"Labeled dataset saved to: {filepath}")


def main():
    df = load_features()
    df_labeled = create_monthly_labels(df)
    save_labeled_data(df_labeled)
    print("Labeling completed successfully!")


if __name__ == "__main__":
    main()
