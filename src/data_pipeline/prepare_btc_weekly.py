"""
prepare_btc_weekly.py

Builds a weekly dataset for BTC:
- reads the raw BTC_prices.csv file
- computes technical features at daily frequency
- aggregates them to weekly frequency (W-FRI)
- builds a label: best strategy over the next 4 weeks
  among Buy & Hold, Momentum, Mean Reversion
- saves data/processed/labeled_data_btc_weekly.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# IMPORTANT:
# __file__ = .../src/data_pipeline/prepare_btc_weekly.py
# parents[0] = .../src/data_pipeline
# parents[1] = .../src
# parents[2] = .../ (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "BTC_prices.csv"
OUT_FILE = DATA_DIR / "processed" / "labeled_data_btc_weekly.csv"


# ---------------------------------------------------------------------
# 1. Load daily prices
# ---------------------------------------------------------------------
def load_btc_daily() -> pd.DataFrame:
    print(f"Loading BTC prices from: {RAW_FILE}")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"[ERROR] BTC raw file not found at {RAW_FILE}\n"
            "Please check that the file exists at:\n"
            "  data/raw/BTC_prices.csv (at the project root)\n"
            "and not inside src/data/raw."
        )

    df = pd.read_csv(RAW_FILE)

    # Date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"])

    # Price column (Adj Close if possible)
    px_col = None
    for c in df.columns:
        cl = c.lower()
        if "adj" in cl and "close" in cl:
            px_col = c
            break
    if px_col is None:
        for c in df.columns:
            if "close" in c.lower():
                px_col = c
                break
    if px_col is None:
        # fallback: take the second column
        px_col = df.columns[1]

    df["Adj Close"] = pd.to_numeric(df[px_col], errors="coerce")
    df = df.dropna(subset=["Adj Close"])
    df = df[["date", "Adj Close"]].set_index("date").sort_index()

    print("BTC daily price data shape:", df.shape)
    return df


# ---------------------------------------------------------------------
# 2. Daily technical features
# ---------------------------------------------------------------------
def build_daily_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    px = df_prices["Adj Close"]

    df = pd.DataFrame(index=df_prices.index)
    df["Adj Close"] = px

    # Returns
    df["ret_1d"] = px.pct_change()
    df["ret_5d"] = px.pct_change(5)
    df["ret_21d"] = px.pct_change(21)
    df["ret_63d"] = px.pct_change(63)

    # Volatilities
    df["vol_21d"] = df["ret_1d"].rolling(21).std()
    df["vol_63d"] = df["ret_1d"].rolling(63).std()

    # Price / moving averages ratios
    sma10 = px.rolling(10).mean()
    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    df["px_over_sma_10"] = px / sma10 - 1
    df["px_over_sma_50"] = px / sma50 - 1
    df["px_over_sma_200"] = px / sma200 - 1

    # RSI 14
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD (12,26,9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal

    # Volatility z-score
    vol_mean = df["vol_21d"].rolling(252).mean()
    vol_std = df["vol_21d"].rolling(252).std()
    df["vol_zscore_21d"] = (df["vol_21d"] - vol_mean) / vol_std

    return df


# ---------------------------------------------------------------------
# 3. Weekly aggregation + label construction
# ---------------------------------------------------------------------
def resample_weekly_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    # take the last observation of each week (Friday)
    weekly = df_daily.resample("W-FRI").last()
    weekly = weekly.dropna()
    print("Weekly feature matrix shape:", weekly.shape)
    return weekly


def compute_weekly_strategy_returns(weekly_px: pd.Series) -> pd.DataFrame:
    """
    Builds weekly returns for the three base strategies:
    - Buy & Hold
    - Momentum (long if the previous 4-week performance > 0)
    - Mean Reversion (long if the previous week's return < 0)
    """
    weekly_ret = weekly_px.pct_change().fillna(0.0)

    # Buy & Hold: always long
    bh_ret = weekly_ret.copy()

    # Momentum: long if sum of last 4 weeks > 0
    momo_signal = (weekly_ret.rolling(4).sum() > 0).shift(1).fillna(0).astype(int)
    momo_ret = momo_signal * weekly_ret

    # Mean reversion: long if previous week's return < 0
    mr_signal = (weekly_ret.shift(1) < 0).astype(int)
    mr_ret = mr_signal * weekly_ret

    out = pd.DataFrame(
        {
            "BH": bh_ret,
            "Momentum": momo_ret,
            "MeanReversion": mr_ret,
        },
        index=weekly_ret.index,
    )
    return out


def build_labels_from_strategies(
    strat_rets: pd.DataFrame, horizon_weeks: int = 4
) -> pd.Series:
    """
    For each week t, we look at the cumulative performance of the
    three strategies over the next horizon_weeks (t+1 ... t+h)
    and select the best one.
    """
    idx = strat_rets.index
    labels = []

    for i in range(len(idx)):
        # future window
        start = i + 1
        end = i + 1 + horizon_weeks
        if end > len(idx):
            labels.append(np.nan)
            continue
        future_slice = strat_rets.iloc[start:end]
        future_cum = (1 + future_slice).prod() - 1  # cumulative over horizon h
        best_col = future_cum.values.argmax()
        labels.append(best_col)

    labels = pd.Series(labels, index=idx, name="best_strategy_code")
    label_map = {0: "Buy & Hold", 1: "Momentum", 2: "Mean Reversion"}
    best_name = labels.map(label_map)
    best_name.name = "best_strategy"
    return labels, best_name


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------
def main():
    print(">>> prepare_btc_weekly.py started")

    prices_daily = load_btc_daily()
    daily_feats = build_daily_features(prices_daily)
    weekly_feats = resample_weekly_features(daily_feats)

    # weekly price series (consistent with weekly_feats)
    weekly_px = weekly_feats["Adj Close"]
    strat_rets = compute_weekly_strategy_returns(weekly_px)
    labels_code, labels_name = build_labels_from_strategies(strat_rets, horizon_weeks=4)

    df_weekly = weekly_feats.copy()
    df_weekly["best_strategy_code"] = labels_code
    df_weekly["best_strategy"] = labels_name

    # drop rows without labels (end of sample)
    df_weekly = df_weekly.dropna(subset=["best_strategy"])

    print("Final BTC weekly labeled dataset shape:", df_weekly.shape)
    print("Label distribution (weekly BTC):")
    print(df_weekly["best_strategy"].value_counts())

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_weekly.reset_index().to_csv(OUT_FILE, index=False)
    print(f"Saved BTC weekly labeled dataset to: {OUT_FILE}")
    print(">>> prepare_btc_weekly.py finished")


if __name__ == "__main__":
    main()
