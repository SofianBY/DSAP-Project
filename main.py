# main.py — top-level entry point required by the course
# This simply calls the full Finsight pipeline defined in src/main.py

from src.main import run_finsight_pipeline


def main():
    """Wrapper so that the graders can call `python main.py`."""
    run_finsight_pipeline()


if __name__ == "__main__":
    main()

