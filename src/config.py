from pathlib import Path

# === Project paths ===
# Root directory of the project (DSAP-Project)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Automatically create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Market parameters ===
TICKER = "SPY"           # Default asset (S&P 500 ETF)
START_DATE = "2000-01-01"
END_DATE = None           # None = up to today
PRICE_FREQUENCY = "1d"    # Daily data

# === General notes ===
# This file centralizes all project settings and paths.
# You can import these constants in other scripts, e.g.:
#     from src.config import RAW_DATA_DIR, TICKER
