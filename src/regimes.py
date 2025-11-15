from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src import config


def load_labeled_data() -> pd.DataFrame:
    file_path = config.PROCESSED_DATA_DIR / "labeled_data.csv"
    df = pd.read_csv(file_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    else:
        raise ValueError(f"No 'date' column found. Columns present: {df.columns}")

    return df


def build_monthly_dataset(df: pd.DataFrame):
    df = df.copy()
    df["month"] = df.index.to_period("M")
    monthly = df.groupby("month").tail(1).copy()
    monthly.index = monthly["month"].dt.to_timestamp("M")
    monthly = monthly.drop(columns=["month"])

    target_col = "best_strategy"

    feature_cols = [
        c for c in monthly.columns
        if c != target_col and pd.api.types.is_numeric_dtype(monthly[c])
    ]

    X = monthly[feature_cols].copy()
    y = monthly[target_col].copy()

    valid_idx = X.dropna().index
    return X.loc[valid_idx], y.loc[valid_idx]


def cluster_market_regimes(X_monthly, y_monthly, n_clusters=3):
    feature_cols = X_monthly.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_monthly)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(X_scaled)

    result = X_monthly.copy()
    result["best_strategy"] = y_monthly.values
    result["regime"] = regimes

    print("\n=== Market Regimes (K-Means) ===")
    for r in sorted(result["regime"].unique()):
        subset = result[result["regime"] == r]
        print(f"\n--- Regime {r} ---")
        print(f"Months: {len(subset)}")

        print("\nAverage features:")
        print(subset[feature_cols].mean().round(3))

        print("\nBest strategy distribution:")
        print(subset["best_strategy"].value_counts(normalize=True).round(3))

    return result


def main():
    print(">>> regimes.py started")

    df = load_labeled_data()
    print("Daily data shape:", df.shape)

    X_monthly, y_monthly = build_monthly_dataset(df)
    print("Monthly data shape:", X_monthly.shape)
    print("Target distribution:")
    print(y_monthly.value_counts())

    result = cluster_market_regimes(X_monthly, y_monthly, n_clusters=3)

    output_path = config.PROCESSED_DATA_DIR / "monthly_regimes.csv"
    result.to_csv(output_path)
    print(f"\nSaved monthly regimes to: {output_path}")

    print(">>> regimes.py finished")


if __name__ == "__main__":
    main()
