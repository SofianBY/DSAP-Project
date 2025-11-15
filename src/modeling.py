# src/modeling.py

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score

from src import config


# ----------------------------------------------------------
# 1) LOAD LABELED DATA
# ----------------------------------------------------------

def load_labeled_data() -> pd.DataFrame:
    """
    Load the labeled dataset from the processed directory.
    Ensures Date is parsed and used as index.
    """
    file_path = config.PROCESSED_DATA_DIR / "labeled_data.csv"
    df = pd.read_csv(file_path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    else:
        # fallback if Date column is missing
        df.index = pd.to_datetime(df.index, errors="ignore")
        df = df.sort_index()

    return df


# ----------------------------------------------------------
# 2) PREPARE FEATURES AND TARGET
# ----------------------------------------------------------

def prepare_features_and_target(df: pd.DataFrame):
    target_col = "best_strategy"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from data.")

    df = df.dropna(subset=[target_col])

    # numeric columns only
    feature_cols = [
        col for col in df.columns
        if col != target_col and pd.api.types.is_numeric_dtype(df[col])
    ]

    X = df[feature_cols].copy()
    valid_idx = X.dropna().index

    X = X.loc[valid_idx]
    y = df.loc[valid_idx, target_col]

    return X, y


# ----------------------------------------------------------
# 3) ENCODE TARGET
# ----------------------------------------------------------

def encode_target(y: pd.Series):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


# ----------------------------------------------------------
# 4) TIME SERIES CROSS VALIDATION (GENERIC FOR ALL MODELS)
# ----------------------------------------------------------

def time_series_cv_model(name: str, model, X, y, n_splits=5):
    """
    Generic function for time-series cross-validation.
    Prints accuracy and balanced accuracy for each fold.
    """
    print(f"\n=== {name}: TimeSeriesSplit cross-validation ===")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies = []
    fold_balanced = []

    last_X_test = None
    last_y_test = None
    final_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        try:
            print(f"Fold {fold}  | test period: {X_test.index[0]} → {X_test.index[-1]}")
        except:
            print(f"Fold {fold}")

        print(f"Accuracy: {acc:.4f}   | Balanced Acc: {bal_acc:.4f}\n")

        fold_accuracies.append(acc)
        fold_balanced.append(bal_acc)

        final_model = model
        last_X_test = X_test
        last_y_test = y_test

    print(f"{name} summary:")
    print(f"Accuracies:       {np.round(fold_accuracies, 4)}")
    print(f"Balanced Accs:    {np.round(fold_balanced, 4)}")
    print(f"Avg Accuracy:     {np.mean(fold_accuracies):.4f}")
    print(f"Avg Balanced Acc: {np.mean(fold_balanced):.4f}\n")

    # classification report on last test fold
    if last_X_test is not None:
        y_last_pred = final_model.predict(last_X_test)
        print(f"{name} - Classification report (last fold):")
        print(classification_report(last_y_test, y_last_pred, zero_division=0))

    return final_model, fold_accuracies, fold_balanced


# ----------------------------------------------------------
# 5) MAIN
# ----------------------------------------------------------

def main():
    print(">>> modeling.py started")

    # load data
    df = load_labeled_data()
    print("Data shape:", df.shape)

    # features + target
    X, y = prepare_features_and_target(df)
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts())

    # encode target
    y_encoded, le = encode_target(y)
    print("\nEncoded classes:", list(le.classes_))

    n_splits = 5

    # ------------------------------------------------------
    # MODELS WITH CLASS WEIGHTS
    # ------------------------------------------------------

    # 1) Logistic Regression
    log_reg = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    _, acc_log, bal_log = time_series_cv_model(
        "Logistic Regression",
        log_reg, X, y_encoded, n_splits
    )

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )

    _, acc_rf, bal_rf = time_series_cv_model(
        "Random Forest",
        rf, X, y_encoded, n_splits
    )

    # 3) Gradient Boosting (pas de class_weight)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    _, acc_gb, bal_gb = time_series_cv_model(
        "Gradient Boosting",
        gb, X, y_encoded, n_splits
    )

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n======================")
    print("SUMMARY OF MODELS")
    print("======================")
    print(f"Logistic Regression: Acc={np.mean(acc_log):.4f}, Balanced={np.mean(bal_log):.4f}")
    print(f"Random Forest:       Acc={np.mean(acc_rf):.4f}, Balanced={np.mean(bal_rf):.4f}")
    print(f"Gradient Boosting:   Acc={np.mean(acc_gb):.4f}, Balanced={np.mean(bal_gb):.4f}")

    print(">>> modeling.py finished")


if __name__ == "__main__":
    main()



