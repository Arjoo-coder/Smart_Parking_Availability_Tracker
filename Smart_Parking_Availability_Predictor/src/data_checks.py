"""Safety checks that keep the data, the model and the dashboard consistent.

Three groups of checks:
  1. check_split()           - the test set is properly separated (no data leakage)
  2. dataset_fingerprint()   - a short "ID" of the dataset a model was trained on
  3. validate_raw_schema()   - is a user-supplied CSV usable? (used by `main.py --data`)
"""

import numpy as np
import pandas as pd

from src import config
from src.features import FEATURE_COLUMNS, LEAKY_COLUMNS, NUMERIC_FEATURES, TARGET

REQUIRED_RAW_COLUMNS = ["timestamp", "parking_lot_id", "location", "total_spaces",
                        "occupied_spaces", "available_spaces", "weather", "special_event"]


# ----------------------------------------------------------------------------
# 1) Train / test separation
# ----------------------------------------------------------------------------
def check_split(train, test, full):
    """Return a list of {"check", "ok", "detail"} dictionaries about the train/test split."""
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("Test days come strictly after training days",
        train["timestamp"].max() < test["timestamp"].min(),
        f"training ends {train['timestamp'].max()}, test starts {test['timestamp'].min()}")

    key = ["timestamp", "parking_lot_id"]
    shared = pd.merge(train[key], test[key], on=key).shape[0]
    add("No row appears in both training and test data", shared == 0, f"{shared} shared rows")

    add("Training + test rows = all rows", len(train) + len(test) == len(full),
        f"{len(train):,} + {len(test):,} = {len(full):,}")

    add("No duplicate readings (one row per lot and hour)",
        not full.duplicated(subset=key).any(), "checked on the full dataset")

    leaky = sorted(set(LEAKY_COLUMNS) & set(FEATURE_COLUMNS))
    add("Answer columns are not used as inputs", not leaky,
        "occupied_spaces, available_spaces and occupancy_rate are excluded" if not leaky else f"leak: {leaky}")
    add("The raw timestamp is not used as an input", "timestamp" not in FEATURE_COLUMNS,
        "only hour/weekday features derived from it are used")

    unseen = sorted(set(test["parking_lot_id"]) - set(train["parking_lot_id"]))
    add("Every test lot also appears in training", not unseen, f"unseen lots: {unseen or 'none'}")

    suspicious = []
    for column in NUMERIC_FEATURES:
        corr = train[column].corr(train[TARGET])
        if pd.notna(corr) and abs(corr) > 0.99:
            suspicious.append(f"{column} (r={corr:.3f})")
    add("No input is a near-copy of the target", not suspicious,
        "largest |correlation| with the target is below 0.99" if not suspicious else ", ".join(suspicious))

    fraction = len(test) / len(full)
    add("Test share matches the setting", abs(fraction - config.TEST_FRACTION) < 0.02,
        f"{fraction:.1%} of rows (setting: {config.TEST_FRACTION:.0%})")
    return checks


def assert_clean_split(checks):
    """Stop the program if any leakage check failed."""
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise ValueError("Train/test checks failed: " + "; ".join(f"{c['check']} ({c['detail']})" for c in failed))


# ----------------------------------------------------------------------------
# 2) Dataset fingerprint (stored with the model, compared by the dashboard)
# ----------------------------------------------------------------------------
def dataset_fingerprint(df):
    """A small, readable summary that changes whenever the data changes.

    It is based on the VALUES (not the file bytes), so it does not change when a
    file is opened and saved on another operating system.
    """
    return {
        "rows": int(len(df)),
        "first_timestamp": str(pd.to_datetime(df["timestamp"]).min()),
        "last_timestamp": str(pd.to_datetime(df["timestamp"]).max()),
        "lots": sorted(df["parking_lot_id"].unique().tolist()),
        "sum_occupied_spaces": int(df["occupied_spaces"].sum()),
        "sum_total_spaces": int(df["total_spaces"].sum()),
    }


# ----------------------------------------------------------------------------
# 3) Schema check for a user-supplied dataset
# ----------------------------------------------------------------------------
def validate_raw_schema(df):
    """Return a list of problems (empty list = the file can be used)."""
    problems = []
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required column(s): {', '.join(missing)}."]

    if len(df) < 500:
        problems.append(f"Only {len(df)} rows. At least 500 hourly rows are needed to train and test a model.")

    stamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if stamps.isna().mean() > 0.05:
        problems.append("More than 5% of the timestamps could not be read as dates.")
    elif stamps.nunique() < 48:
        problems.append("The data should cover at least 2 days of readings.")

    unknown = sorted(set(df["parking_lot_id"].dropna().astype(str)) - set(config.PARKING_LOTS))
    if unknown:
        problems.append(f"Unknown parking_lot_id value(s): {', '.join(unknown)}. Add them to "
                        "PARKING_LOTS in src/config.py (location name, total_spaces, type) or fix the file.")

    for column in ("total_spaces", "occupied_spaces"):
        numbers = pd.to_numeric(df[column], errors="coerce")
        if numbers.isna().mean() > 0.05:
            problems.append(f"Column '{column}' contains too many non-numeric values.")
    occupied = pd.to_numeric(df["occupied_spaces"], errors="coerce")
    total = pd.to_numeric(df["total_spaces"], errors="coerce")
    if ((occupied < 0) | (occupied > total)).mean() > 0.05:
        problems.append("More than 5% of rows have occupied_spaces below 0 or above total_spaces.")
    return problems
