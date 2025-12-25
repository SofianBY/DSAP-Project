import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src import config


# ----------------------------------------------------------
# 1) Load daily labeled data
# ----------------------------------------------------------

def load_labeled_data() -> pd.DataFrame:
    """
    Load labeled daily data from data/processed/labeled_data.csv
    """
    file_path = config.PROCESSED_DATA_DIR / "labeled_data.csv"
    df = pd.read_csv(file_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    else:
        raise ValueError(f"No 'date' column found. Columns present: {df.columns}")

    return df


# ----------------------------------------------------------
# 2) Build monthly dataset
# ----------------------------------------------------------

def build_monthly_dataset(df: pd.DataFrame):
    """
    Builds a monthly dataset from daily data:
        - 1 row = last day of each month
        - features = numeric columns
        - target = best_strategy of the month"""
   
    
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


# ----------------------------------------------------------
# 3) Build binary target
# ----------------------------------------------------------

def build_binary_target(y_monthly: pd.Series) -> np.ndarray:
    """
    Transforme la cible multi-classes en cible binaire :
      0 = buy_hold (stratégie de base)
      1 = alternative (momentum OU mean_reversion)
    """
    y_bin = (y_monthly != "buy_hold").astype(int)
    print("\nBinary target distribution (0=buy_hold, 1=alternative):")
    print(y_bin.value_counts())
    return y_bin.values


# ----------------------------------------------------------
# 4) TimeSeriesSplit - Monthly Logistic Regression (binary)
# ----------------------------------------------------------

def time_series_cv_log_reg_monthly(X: pd.DataFrame, y_bin: np.ndarray, n_splits: int = 5):
    """
    TimeSeriesSplit sur les données mensuelles avec Logistic Regression binaire.
    """
    print("\n=== Monthly Logistic Regression (binary target) ===")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_acc = []
    fold_bal_acc = []

    last_X_test = None
    last_y_test = None
    final_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]

        model = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        start, end = X_test.index[0].date(), X_test.index[-1].date()
        print(f"Fold {fold} | {start} → {end}")
        print(f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}\n")

        fold_acc.append(acc)
        fold_bal_acc.append(bal_acc)

        final_model = model
        last_X_test = X_test
        last_y_test = y_test

    print("Summary (Monthly Binary Logistic Regression):")
    print("Accuracies:      ", np.round(fold_acc, 4))
    print("Balanced Accs:   ", np.round(fold_bal_acc, 4))
    print(f"Avg Accuracy:     {np.mean(fold_acc):.4f}")
    print(f"Avg Balanced Acc: {np.mean(fold_bal_acc):.4f}\n")

    if last_X_test is not None:
        y_last_pred = final_model.predict(last_X_test)
        print("Classification report on last fold (binary target):")
        print(classification_report(last_y_test, y_last_pred, zero_division=0))

    return final_model


# ----------------------------------------------------------
# 5) MAIN
# ----------------------------------------------------------

def main():
    print(">>> modeling_monthly.py started")

    df = load_labeled_data()
    print("Daily data shape:", df.shape)

    X_monthly, y_monthly = build_monthly_dataset(df)
    print("Monthly data shape:", X_monthly.shape)
    print("Monthly target distribution:")
    print(y_monthly.value_counts())

    y_bin = build_binary_target(y_monthly)

    final_model = time_series_cv_log_reg_monthly(X_monthly, y_bin, n_splits=5)

    print(">>> modeling_monthly.py finished")


if __name__ == "__main__":
    main()

