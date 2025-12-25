import pandas as pd

# 1) Load the file created by your backtest
path = "data/results/sp500/monthly_returns_sp500.csv"
df = pd.read_csv(path)

# 2) The three possible strategies
strategies = ["Buy & Hold", "Momentum", "Mean Reversion"]

# 3) For each month, infer which strategy ML Adaptive selected
def infer_choice(row, tol=1e-10):
    ml = row["ML Adaptive"]

    # If ML is NaN → no choice
    if pd.isna(ml):
        return "None"

    # If ML exactly matches BH / Momentum / MR → return the strategy name
    for s in strategies:
        if pd.notna(row[s]) and abs(row[s] - ml) < tol:
            return s

    # If ML is (almost) 0 → consider that the model stays in cash
    if abs(ml) < tol:
        return "Cash/0"

    # Otherwise (just in case) → unknown
    return "Unknown"

df["ml_choice"] = df.apply(infer_choice, axis=1)

# 4) Raw counts
print("\nNumber of months per strategy chosen by ML:")
print(df["ml_choice"].value_counts())

# 5) Clean proportions in % (keep only true choices BH / MOM / MR)
mask = df["ml_choice"].isin(strategies)
proportions = df.loc[mask, "ml_choice"].value_counts(normalize=True) * 100

print("\nProportion (%) of months per strategy (ML Adaptive):")
print(proportions.round(2))
