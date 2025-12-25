"""
modeling_btc_weekly.py

Evaluate several classifiers to predict the best weekly strategy on BTC,
from the dataset produced by prepare_btc_weekly.py."""


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IN_FILE = DATA_DIR / "processed" / "labeled_data_btc_weekly.csv"

FEATURE_COLS = [
    "ret_1d",
    "vol_21d",
    "vol_63d",
    "px_over_sma_10",
    "px_over_sma_50",
    "px_over_sma_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_zscore_21d",
]


def load_dataset():
    print(f"Loading BTC weekly dataset from: {IN_FILE}")
    df = pd.read_csv(IN_FILE, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    print("BTC weekly data shape:", df.shape)

    X = df[FEATURE_COLS].astype(float)
    y = df["best_strategy"]

    print("Target distribution:")
    print(y.value_counts())
    return df, X, y


def evaluate_model(name, model, X, y, n_splits=5):
    print(f"\n=== {name} (BTC weekly): TimeSeriesSplit cross-validation ===")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []
    baccs = []

    last_y_true = None
    last_y_pred = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)

        accs.append(acc)
        baccs.append(bacc)

        print(
            f"Fold {fold} | {X.index[train_idx[0]].date()} → {X.index[train_idx[-1]].date()}"
        )
        print(f"  Accuracy:       {acc:.4f}")
        print(f"  Balanced Acc.:  {bacc:.4f}\n")

        last_y_true = y_test
        last_y_pred = y_pred

    print(f"{name} summary over {n_splits} folds:")
    print("  Accuracies:      ", np.round(accs, 4))
    print("  Balanced Accs:   ", np.round(baccs, 4))
    print(f"  Avg Accuracy:     {np.mean(accs):.4f}")
    print(f"  Avg Balanced Acc: {np.mean(baccs):.4f}")

    if last_y_true is not None:
        print(f"\n{name} - Classification report (last fold):")
        print(classification_report(last_y_true, last_y_pred))


def main():
    print(">>> modeling_btc_weekly.py started")

    df, X, y = load_dataset()

    # Logistic Regression (with scaling)
    log_reg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="auto",
                    max_iter=500,
                    C=0.5,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.7,
        random_state=42,
    )

    evaluate_model("Logistic Regression", log_reg, X, y)
    evaluate_model("Random Forest", rf, X, y)
    evaluate_model("Gradient Boosting", gb, X, y)

    print("\n>>> modeling_btc_weekly.py finished")


if __name__ == "__main__":
    main()
