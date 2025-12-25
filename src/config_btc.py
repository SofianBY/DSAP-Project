# src/config_btc.py
"""
config_btc.py — configuration specific to the BTC asset.
It reuses the base directories defined in src.config but
keeps its own ticker and date range.
"""

from pathlib import Path

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

# === BTC-specific market parameters ===
TICKER = "BTC-USD"
START_DATE = "2014-01-01"
END_DATE = "2025-11-30"

# === BTC-specific paths ===
# Raw prices file (written by data_download_btc.py)
RAW_PRICE_FILE = RAW_DATA_DIR / "BTC_prices.csv"

# Processed data folder for BTC (to archive features / labels if needed)
BTC_PROCESSED_DIR = PROCESSED_DATA_DIR / "btc"
BTC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Results folder for BTC (figures + performance tables)
BTC_RESULTS_DIR = RESULTS_DIR / "btc"
BTC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
