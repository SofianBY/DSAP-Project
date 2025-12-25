import warnings
from src.main import run_finsight_pipeline

# On coupe tous les warnings moches (sklearn, pandas, etc.)
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("\n==================== FULL FINSIGHT PIPELINE ====================\n")
    run_finsight_pipeline()
