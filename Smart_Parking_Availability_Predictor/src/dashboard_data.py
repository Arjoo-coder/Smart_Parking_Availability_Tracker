"""Data helpers for the Streamlit dashboard (Phase 2).

This file has NO Streamlit code on purpose: everything here is plain
Pandas, so it can be tested from the command line and reused anywhere.

Every number shown in the dashboard comes from one of the saved Phase 1
artifacts - nothing is hardcoded and nothing is generated randomly:
    data/processed/parking_data_processed.csv   -> snapshots, charts
    models/best_model.joblib                    -> predictions (via src.predict)
    outputs/results/test_predictions.csv        -> actual vs predicted
    outputs/results/model_comparison.csv, model_metadata.json, per_lot_test_error.csv
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.features import FEATURE_COLUMNS
from src.predict import status_label

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_NAMES_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MAX_FORECAST_DAYS = 365       # predictions further ahead than this are rejected
WEATHER_WARNING_DAYS = 14     # beyond this, the weather is a "what-if" choice


# ----------------------------------------------------------------------------
# Loading saved artifacts
# ----------------------------------------------------------------------------
def load_dashboard_data(path=config.PROCESSED_DATA_PATH):
    """Load the processed Phase 1 dataset (raises a clear error if missing)."""
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["occupancy_pct"] = df["occupied_spaces"] / df["total_spaces"] * 100
    df["day_name"] = df["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    return df


def _read_csv_if_exists(path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else None


def load_test_predictions():
    """Hold-out test predictions written by train.py (None if not available)."""
    return _read_csv_if_exists(config.RESULTS_DIR / "test_predictions.csv",
                               parse_dates=["timestamp"])


def load_model_comparison():
    return _read_csv_if_exists(config.MODEL_COMPARISON_PATH)


def load_model_metadata():
    if config.MODEL_METADATA_PATH.exists():
        return json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def load_per_lot_error():
    """Average test error in parking spaces for each location: {location: spaces}."""
    table = _read_csv_if_exists(config.RESULTS_DIR / "per_lot_test_error.csv")
    if table is None:
        return {}
    return dict(zip(table["location"], table["mae_available_spaces"]))


# ----------------------------------------------------------------------------
# Snapshots (the "current" picture shown on Home / Availability)
# ----------------------------------------------------------------------------
# The data is simulated and ends on a fixed date, so "current" means "the
# snapshot the user selects" - by default the latest reading in the dataset.
SNAPSHOT_PRESETS = [
    ("Weekday morning rush", "weekday", 9),
    ("Weekday lunchtime", "weekday", 12),
    ("Weekday evening", "weekday", 18),
    ("Weekend afternoon", "weekend", 15),
    ("Night", "any", 3),
]


def snapshot_choices(df):
    """Return {label: Timestamp} - the latest reading plus typical busy/quiet moments."""
    stamps = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
    choices = {"Latest reading": stamps.max()}
    for label, kind, hour in SNAPSHOT_PRESETS:
        mask = np.asarray(stamps.hour == hour)
        if kind == "weekday":
            mask &= np.asarray(stamps.dayofweek < 5)
        elif kind == "weekend":
            mask &= np.asarray(stamps.dayofweek >= 5)
        matches = stamps[mask]
        if len(matches):
            choices[f"{label} ({hour:02d}:00)"] = matches.max()
    return choices


def snapshot_at(df, timestamp):
    """One row per parking lot at a given timestamp, with a status label."""
    rows = df[df["timestamp"] == pd.Timestamp(timestamp)]
    if rows.empty:
        raise ValueError(f"No readings found for {timestamp}")
    snap = rows[["timestamp", "parking_lot_id", "location", "total_spaces",
                 "occupied_spaces", "available_spaces", "weather", "special_event"]].copy()
    snap["occupancy_rate"] = snap["occupied_spaces"] / snap["total_spaces"]
    snap["occupancy_pct"] = snap["occupancy_rate"] * 100
    snap["status"] = snap["occupancy_rate"].map(status_label)
    return snap.sort_values("parking_lot_id").reset_index(drop=True)


def previous_snapshot(df, timestamp, hours=1):
    """The snapshot `hours` earlier, or None if the dataset does not go back that far."""
    earlier = pd.Timestamp(timestamp) - pd.Timedelta(hours=hours)
    try:
        return snapshot_at(df, earlier)
    except ValueError:
        return None


def overall_summary(snapshot):
    """Totals across all locations for a snapshot."""
    total = int(snapshot["total_spaces"].sum())
    occupied = int(snapshot["occupied_spaces"].sum())
    return {
        "total_spaces": total,
        "occupied_spaces": occupied,
        "available_spaces": total - occupied,
        "occupancy_pct": occupied / total * 100 if total else 0.0,
        "num_locations": int(len(snapshot)),
    }


# ----------------------------------------------------------------------------
# Aggregations for charts
# ----------------------------------------------------------------------------
def _filter_lots(df, location_names):
    if location_names:
        return df[df["location"].isin(location_names)]
    return df


def hourly_profile(df, location_names=None, day_type="All days"):
    """Average demand per hour of the day (optionally weekdays / weekends only)."""
    data = _filter_lots(df, location_names)
    if day_type == "Weekdays":
        data = data[data["is_weekend"] == 0]
    elif day_type == "Weekends":
        data = data[data["is_weekend"] == 1]
    return (data.groupby(["location", "hour"])
                .agg(occupancy_pct=("occupancy_pct", "mean"),
                     occupied_spaces=("occupied_spaces", "mean"),
                     available_spaces=("available_spaces", "mean"))
                .reset_index())


def day_of_week_profile(df, location_names=None):
    """Average occupancy for each weekday (Monday ... Sunday)."""
    data = _filter_lots(df, location_names)
    return (data.groupby(["location", "day_of_week"])
                .agg(occupancy_pct=("occupancy_pct", "mean"))
                .reset_index())


def location_summary(df):
    """One row per location: capacity, average and peak demand, busiest hour."""
    hourly = (df.groupby(["parking_lot_id", "hour"])["occupancy_pct"].mean().reset_index())
    peak = hourly.loc[hourly.groupby("parking_lot_id")["occupancy_pct"].idxmax()]
    peak = peak.set_index("parking_lot_id")

    summary = (df.groupby(["parking_lot_id", "location"])
                 .agg(total_spaces=("total_spaces", "first"),
                      avg_occupancy_pct=("occupancy_pct", "mean"),
                      avg_available_spaces=("available_spaces", "mean"))
                 .reset_index())
    summary["peak_hour"] = summary["parking_lot_id"].map(peak["hour"]).astype(int)
    summary["peak_occupancy_pct"] = summary["parking_lot_id"].map(peak["occupancy_pct"])
    return summary


def historical_reference(df, lot_id, hour, is_weekend):
    """What this lot looked like at this hour in the dataset (context for a prediction)."""
    rows = df[(df["parking_lot_id"] == lot_id) & (df["hour"] == hour)
              & (df["is_weekend"] == int(is_weekend))]
    if rows.empty:
        return None
    return {"avg_occupancy_pct": float(rows["occupancy_pct"].mean()),
            "avg_available_spaces": float(rows["available_spaces"].mean()),
            "n": int(len(rows))}


# ----------------------------------------------------------------------------
# Actual vs predicted (only the hold-out TEST period is valid)
# ----------------------------------------------------------------------------
def prediction_window(test_predictions, location, start, days):
    """Rows of the test period for one location between start and start + days."""
    start = pd.Timestamp(start).normalize()
    end = start + pd.Timedelta(days=days)
    rows = test_predictions[(test_predictions["location"] == location)
                            & (test_predictions["timestamp"] >= start)
                            & (test_predictions["timestamp"] < end)]
    return rows.sort_values("timestamp")


def prediction_metrics(window):
    """MAE / RMSE / R2 on AVAILABLE SPACES for the rows given (None if too few rows)."""
    if len(window) < 2:
        return None
    actual = window["actual_available_spaces"]
    predicted = window["predicted_available_spaces"]
    return {
        "rows": int(len(window)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


# ----------------------------------------------------------------------------
# Prediction inputs
# ----------------------------------------------------------------------------
def lot_supports_events(lot_id):
    """Only lots with event_prob > 0 have special events in the data."""
    return config.PARKING_LOTS[lot_id]["event_prob"] > 0


def next_date_for_weekday(weekday_index, today):
    """The next date (today included) that falls on the given weekday (Mon=0)."""
    today = pd.Timestamp(today).normalize()
    return today + pd.Timedelta(days=(weekday_index - today.dayofweek) % 7)


def validate_prediction_inputs(lot_id, when, weather, event_today, today):
    """Check the prediction inputs. Returns (errors, warnings) as lists of messages.

    errors   -> the prediction must NOT be made
    warnings -> the prediction is made, but the user should know about a caveat
    """
    errors, warnings = [], []

    if lot_id not in config.PARKING_LOTS:
        errors.append(f"Unknown parking location '{lot_id}'.")
    if weather not in config.WEATHER_OPTIONS:
        errors.append(f"Unknown weather '{weather}'. Choose one of: {', '.join(config.WEATHER_OPTIONS)}.")
    try:
        when = pd.Timestamp(when)
        if pd.isna(when):
            raise ValueError
    except (ValueError, TypeError):
        errors.append("The date and time could not be understood.")
        return errors, warnings
    if errors:
        return errors, warnings

    days_ahead = (when.normalize() - pd.Timestamp(today).normalize()).days
    if days_ahead > MAX_FORECAST_DAYS:
        errors.append(f"That date is {days_ahead} days ahead. Please choose a date within "
                      f"{MAX_FORECAST_DAYS} days from today.")
    elif days_ahead > WEATHER_WARNING_DAYS:
        warnings.append("Weather cannot be known this far ahead, so treat the chosen weather "
                        "as a what-if scenario.")
    if when < pd.Timestamp(config.START_DATE):
        warnings.append(f"This date is before the period the model was trained on "
                        f"(starting {config.START_DATE}). The model only uses weekday, hour, "
                        "weather and events, so it still answers, but treat it as a what-if.")

    if event_today:
        if not lot_supports_events(lot_id):
            errors.append(f"No special events are modelled for {config.PARKING_LOTS[lot_id]['location']}.")
        elif not (config.EVENT_START_HOUR <= when.hour <= config.EVENT_END_HOUR):
            warnings.append(f"Events only affect {config.EVENT_START_HOUR}:00-{config.EVENT_END_HOUR}:00 "
                            f"in the data, so at {when.hour:02d}:00 the model treats this as a normal time.")
    return errors, warnings


# ----------------------------------------------------------------------------
# Checks that tie the dashboard to the training pipeline
# ----------------------------------------------------------------------------
def model_feature_check(model):
    """Does the saved model expect exactly the feature columns used in training?

    Returns (ok, message). `feature_names_in_` is stored inside the saved
    scikit-learn pipeline when it was fitted on a DataFrame.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return False, "The saved model does not record its input columns, so they could not be checked."
    expected = list(expected)
    if expected == list(FEATURE_COLUMNS):
        return True, f"Feature check passed: the model expects exactly the {len(expected)} columns used in training."
    missing = [c for c in FEATURE_COLUMNS if c not in expected]
    extra = [c for c in expected if c not in FEATURE_COLUMNS]
    return False, (f"Feature mismatch. Missing in model: {missing or 'none'}; "
                   f"unexpected in model: {extra or 'none'}. Run `python main.py` to retrain.")


def recorded_reading(df, test_predictions, lot_id, when):
    """What the dataset recorded for one lot at one hour, if that hour exists.

    Returns None when the hour is not in the dataset (a real forecast), otherwise a dict:
      kind = "test"     -> hour is in the held-out test period (a fair comparison is possible;
                           includes the model's stored prediction for it)
      kind = "training" -> hour was used for training (comparing would flatter the model)
    """
    when = pd.Timestamp(when)
    rows = df[(df["parking_lot_id"] == lot_id) & (df["timestamp"] == when)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    info = {"kind": "training", "actual_available": int(row["available_spaces"]),
            "weather": row["weather"], "special_event": int(row["special_event"])}
    if test_predictions is not None and not test_predictions.empty:
        match = test_predictions[(test_predictions["parking_lot_id"] == lot_id)
                                 & (test_predictions["timestamp"] == when)]
        if not match.empty:
            info["kind"] = "test"
            info["model_available"] = int(match.iloc[0]["predicted_available_spaces"])
    return info


# ----------------------------------------------------------------------------
# Phase 3: more saved artifacts + consistency checks
# ----------------------------------------------------------------------------
def load_data_source():
    """{"is_simulated": bool, "source_file": str}. Defaults to simulated when the file is missing."""
    if config.DATA_SOURCE_PATH.exists():
        try:
            return json.loads(config.DATA_SOURCE_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"is_simulated": True, "source_file": config.RAW_DATA_PATH.name}


def load_all_test_predictions():
    """Test-day predictions of EVERY model (None if train.py has not written them)."""
    return _read_csv_if_exists(config.ALL_TEST_PREDICTIONS_PATH, parse_dates=["timestamp"])


def load_feature_importance():
    return _read_csv_if_exists(config.FEATURE_IMPORTANCE_PATH)


def load_leakage_checks():
    path = config.RESULTS_DIR / "leakage_checks.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def model_metrics_on(all_predictions, model_name, location=None, start=None, days=None):
    """MAE / RMSE / R2 (in occupancy rate AND spaces) of one model on a slice of the test days."""
    rows = all_predictions[all_predictions["model"] == model_name]
    if location:
        rows = rows[rows["location"] == location]
    if start is not None and days is not None:
        start = pd.Timestamp(start).normalize()
        rows = rows[(rows["timestamp"] >= start) & (rows["timestamp"] < start + pd.Timedelta(days=days))]
    if len(rows) < 2:
        return None
    return {
        "rows": int(len(rows)),
        "mae_spaces": float(mean_absolute_error(rows["actual_available_spaces"], rows["predicted_available_spaces"])),
        "rmse_spaces": float(np.sqrt(mean_squared_error(rows["actual_available_spaces"], rows["predicted_available_spaces"]))),
        "r2": float(r2_score(rows["actual_occupancy_rate"], rows["predicted_occupancy_rate"])),
    }


def consistency_issues(df, metadata, model, test_predictions=None):
    """Compare the dataset, the saved model and the saved results. Returns [(level, message)].

    level is "error" (the numbers on screen could be wrong) or "warning" (worth knowing).
    An empty list means everything fits together.
    """
    from src import data_checks          # local import keeps module start-up light
    issues = []

    ok, message = model_feature_check(model) if model is not None else (True, "")
    if not ok:
        issues.append(("error", message))

    fingerprint = metadata.get("dataset_fingerprint")
    if not metadata:
        issues.append(("warning", "No model metadata found, so the model could not be matched to the dataset. "
                                  "Run `python main.py`."))
    elif fingerprint is None:
        issues.append(("warning", "The saved model was trained before dataset fingerprints existed, so its "
                                  "match with the dataset cannot be verified. Run `python main.py --train-only`."))
    elif fingerprint != data_checks.dataset_fingerprint(df):
        issues.append(("error", "The dataset has changed since the model was trained, so model scores and "
                                "predictions may not match the data shown. Run `python main.py --train-only` "
                                "(or `python main.py`) to retrain."))

    source = load_data_source()
    if metadata and "data_is_simulated" in metadata and bool(metadata["data_is_simulated"]) != bool(source["is_simulated"]):
        issues.append(("error", "The model was trained on a different kind of data (simulated vs genuine) than "
                                "the dataset now saved. Run `python main.py` again."))

    lots_in_data = set(df["parking_lot_id"].unique())
    if lots_in_data != set(config.PARKING_LOTS):
        issues.append(("warning", f"Parking lots in the data {sorted(lots_in_data)} differ from the lots in "
                                  f"src/config.py {sorted(config.PARKING_LOTS)}."))
    else:
        capacity = df.groupby("parking_lot_id")["total_spaces"].max()
        wrong = [lot for lot, total in capacity.items() if total != config.PARKING_LOTS[lot]["total_spaces"]]
        if wrong:
            issues.append(("error", f"total_spaces in the data differs from src/config.py for {', '.join(wrong)}; "
                                    "predicted free spaces would use the wrong capacity."))

    if test_predictions is not None and metadata.get("split"):
        test_start = pd.Timestamp(metadata["split"]["test_start"])
        if test_predictions["timestamp"].min() < test_start:
            issues.append(("error", "Saved test predictions include days from the training period, so the "
                                    "'unseen data' plots would not be fair. Run `python main.py --train-only`."))
    return issues
