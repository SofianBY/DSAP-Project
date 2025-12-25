# src/modeling_btc.py — Robust BTC modeling

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


DATA_PATH = Path("data/processed/labeled_data_btc.csv")


# ----------------------------------------------------------
# LOAD BTC MONTHLY DATA
# ----------------------------------------------------------

def load_btc_data():
    df = pd.read_csv(DATA_PATH)

    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in labeled_data_btc.csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")

    return df


# ----------------------------------------------------------
# PREP FEATURES
# ----------------------------------------------------------

def prepare_features(df):
    target_col = "best_strategy"

    features = [
        col for col in df.columns
        if col not in ["best_strategy", "date"] and np.issubdtype(df[col].dtype, np.number)
    ]

    X = df[features].copy()
    y = df[target_col].copy()

    valid_idx = X.dropna().index
    return X.loc[valid_idx], y.loc[valid_idx], features


# ----------------------------------------------------------
# 5-year rolling training window
# ----------------------------------------------------------

def time_series_rolling_cv(X, y, window_years=5, freq="M"):
    """Yield rolling train/test splits using a fixed training window size."""
    dates = X.index

    # each test period is one month
    for i in range(window_years * 12, len(dates)):
        train_start = i - window_years * 12
        train_idx = np.arange(train_start, i)
        test_idx = np.array([i])

        yield train_idx, test_idx


# ----------------------------------------------------------
# Robust model definitions
# ----------------------------------------------------------

def get_models():

    # Logistic Regression
    log_reg = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    # Random Forest (shallow & conservative)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=3,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42
    )

    # Gradient Boosting — ROBUST VERSION
    gb = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=1,
        subsample=0.8,
        random_state=42
    )

    return {
        "Logistic Regression": log_reg,
        "Random Forest": rf,
        "Gradient Boosting": gb
    }


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    print(">>> modeling_btc.py (robust version) started")

    df = load_btc_data()
    X, y, feats = prepare_features(df)

    print(f"Dataset shape: {X.shape}")
    print(f"Features used ({len(feats)}): {feats}")
    print("\nTarget distribution:\n", y.value_counts())

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print("\nEncoded classes:")
    for idx, c in enumerate(le.classes_):
        print(f"  {idx} -> {c}")

    models = get_models()

    print("\n=== Rolling-window evaluation (5-year training window) ===")

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        accuracies = []
        balances = []

        for train_idx, test_idx in time_series_rolling_cv(X, y_enc):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            bal = balanced_accuracy_score(y_test, y_pred)

            accuracies.append(acc)
            balances.append(bal)

        print(f"Mean Accuracy:     {np.mean(accuracies):.4f}")
        print(f"Mean Balanced Acc: {np.mean(balances):.4f}")

    print("\n>>> modeling_btc.py finished")


if __name__ == "__main__":
    main()
