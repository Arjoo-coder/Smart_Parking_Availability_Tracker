"""Run the machine-learning pipeline with one command:

    python main.py                      # simulate data -> clean -> EDA -> train/compare -> save model
    python main.py --train-only         # keep the saved dataset, only re-train the models
    python main.py --no-tune            # skip the hyper-parameter search (faster)
    python main.py --skip-eda           # skip the exploratory charts
    python main.py --data my_data.csv   # use GENUINE parking data instead of the simulation

The project uses SIMULATED data unless you supply your own file with --data
(see README, section "Using genuine parking data").
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from src import config, data_checks, data_generator, eda, preprocessing, train

SUPPLIED_DATA_PATH = config.DATA_RAW_DIR / "parking_data_supplied.csv"


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def use_supplied_data(path):
    """Validate a user-supplied CSV and copy it into data/raw/ (the simulated file is NOT touched)."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: the file '{path}' does not exist.")
    try:
        df = pd.read_csv(path)
    except Exception as error:
        sys.exit(f"ERROR: the file could not be read as a CSV ({error}).")
    problems = data_checks.validate_raw_schema(df)
    if problems:
        print("The supplied data cannot be used yet:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(2)
    shutil.copyfile(path, SUPPLIED_DATA_PATH)
    print(f"[data] Genuine data accepted: {len(df):,} rows. Copy saved as {SUPPLIED_DATA_PATH}")
    return SUPPLIED_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Smart Parking Availability Predictor - pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="skip exploratory charts")
    parser.add_argument("--train-only", action="store_true",
                        help="reuse the saved processed dataset and only re-train the models")
    parser.add_argument("--no-tune", action="store_true", help="skip the hyper-parameter search")
    parser.add_argument("--data", metavar="CSV", help="use this genuine parking CSV instead of simulating data")
    args = parser.parse_args()

    start = time.time()
    config.ensure_directories()

    if args.train_only:
        if args.data:
            sys.exit("ERROR: use either --train-only or --data, not both.")
        banner("STEP 1-3  Skipped (--train-only): using the saved dataset")
    else:
        if args.data:
            banner("STEP 1/4  Check the supplied GENUINE dataset")
            raw_path = use_supplied_data(args.data)
        else:
            banner("STEP 1/4  Generate SIMULATED parking dataset")
            data_generator.generate_and_save()
            raw_path = config.RAW_DATA_PATH

        banner("STEP 2/4  Clean data and engineer features")
        preprocessing.run_preprocessing(raw_path=raw_path, is_simulated=not args.data)

        if args.skip_eda:
            banner("STEP 3/4  EDA skipped (--skip-eda)")
        else:
            banner("STEP 3/4  Exploratory data analysis")
            eda.run_eda()

    banner("STEP 4/4  Train, tune, compare and save models")
    train.run_training(tune=False if args.no_tune else None)

    banner(f"DONE in {time.time() - start:.1f} seconds")
    print("Next: `streamlit run app/streamlit_app.py`  (dashboard)")
    print("  or: `python -m src.predict --lot LOT_A --datetime \"2026-11-06 10:00\" --hours 6`")


if __name__ == "__main__":
    main()
