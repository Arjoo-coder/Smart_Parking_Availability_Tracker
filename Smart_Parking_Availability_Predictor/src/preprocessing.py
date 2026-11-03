"""Data cleaning and preprocessing.

Reads the raw CSV, fixes problems, adds engineered features and saves the
result to data/processed/. A small cleaning report is also saved so you can
see exactly what was changed.
"""

import json
from pathlib import Path

import pandas as pd

from src import config
from src.features import add_time_features


def load_raw_data(path=None):
    path = path or config.RAW_DATA_PATH
    """Read the raw CSV file."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw data not found: {path}\nRun `python main.py` (or "
            "`python -m src.data_generator`) to create it first.")
    return pd.read_csv(path)


def clean_data(df):
    """Clean the raw data. Returns (clean_dataframe, report_dictionary)."""
    report = {"rows_before_cleaning": int(len(df))}
    df = df.copy()

    # 1) Convert the timestamp text into real date-times; drop unreadable ones.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    report["rows_dropped_bad_timestamp"] = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    # 2) Remove duplicates: a lot can only have ONE reading per timestamp.
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "parking_lot_id"], keep="first")
    report["duplicate_rows_removed"] = int(before - len(df))

    # 3) Missing weather -> "Unknown" (a neutral label; avoids guessing).
    report["missing_weather_filled"] = int(df["weather"].isna().sum())
    df["weather"] = df["weather"].fillna("Unknown").astype(str).str.strip().str.title()

    # 4) Missing event flag -> 0 (no event).
    df["special_event"] = df["special_event"].fillna(0).astype(int)

    # 5) Rows without the core numbers cannot be used.
    before = len(df)
    df = df.dropna(subset=["total_spaces", "occupied_spaces"])
    report["rows_dropped_missing_core_values"] = int(before - len(df))

    # 6) Remove impossible values (negative or more cars than spaces).
    df["total_spaces"] = df["total_spaces"].astype(int)
    df["occupied_spaces"] = df["occupied_spaces"].astype(int)
    invalid = (df["occupied_spaces"] < 0) | (df["occupied_spaces"] > df["total_spaces"])
    report["rows_dropped_invalid_occupancy"] = int(invalid.sum())
    df = df[~invalid]

    # 7) Re-compute the dependent columns so everything is consistent.
    available = df["total_spaces"] - df["occupied_spaces"]
    report["available_spaces_corrected"] = int((df["available_spaces"] != available).sum())
    df["available_spaces"] = available
    df["occupancy_rate"] = (df["occupied_spaces"] / df["total_spaces"]).round(4)

    # 8) Sort by time (important for the time-based train/test split later).
    df = df.sort_values(["timestamp", "parking_lot_id"]).reset_index(drop=True)
    report["rows_after_cleaning"] = int(len(df))
    return df, report


def run_preprocessing(raw_path=None, out_path=None, is_simulated=True):
    """Full preprocessing: load -> clean -> add features -> save.

    `is_simulated` is stored next to the processed data so the model and the dashboard
    always know whether the numbers are simulated or genuine (see main.py --data).
    """
    config.ensure_directories()
    raw_path = raw_path or config.RAW_DATA_PATH
    out_path = out_path or config.PROCESSED_DATA_PATH

    raw = load_raw_data(raw_path)
    clean, report = clean_data(raw)
    processed = add_time_features(clean)   # hour, day_of_week, is_weekend, sin/cos ...
    processed.to_csv(out_path, index=False)

    with open(config.DATA_SOURCE_PATH, "w", encoding="utf-8") as f:
        json.dump({"is_simulated": bool(is_simulated), "source_file": Path(raw_path).name}, f, indent=2)

    report_path = config.RESULTS_DIR / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[preprocess] Cleaning report:")
    for key, value in report.items():
        print(f"    {key}: {value}")
    print(f"[preprocess] Processed data saved: {out_path}")
    return processed


if __name__ == "__main__":
    run_preprocessing()
