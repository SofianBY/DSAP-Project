from pathlib import Path

# === Project paths ===
# Root directory of the project (DSAP-Project)
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR  # alias if you want to use PROJECT_ROOT elsewhere

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"  # where equity_curves, stats, plots are saved

# Automatically create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Market parameters (default: SPY pipeline) ===
TICKER = "SPY"            # Default asset (S&P 500 ETF)
START_DATE = "2000-01-01"
END_DATE = None           # None = up to today
PRICE_FREQUENCY = "1d"    # Daily data

# === General notes ===
# This file centralizes all project settings and paths.
# You can import these constants in other scripts, e.g.:
#     from src.config import RAW_DATA_DIR, RESULTS_DIR, TICKER
