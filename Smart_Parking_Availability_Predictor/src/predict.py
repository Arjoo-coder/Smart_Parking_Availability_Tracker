"""Use the saved model to predict parking availability.

Command-line examples:
    python -m src.predict --lot LOT_B --datetime "2026-11-07 15:00" --weather Rainy
    python -m src.predict --lot LOT_B --datetime "2026-11-07 15:00" --hours 6

Phase 2 note: `prepare_features()` was split out of `predict_many()` so the
dashboard can show exactly which feature values are sent to the model.

Phase 3 additions (the old functions keep working exactly as before):
  * `normalize_request()`   - tolerant input handling (missing weather -> "Unknown",
                              "lot_a " -> "LOT_A") with clear errors for invalid values
  * hard limits             - predicted available spaces are always between 0 and the
                              lot's total capacity
  * `predict_slots()`       - hourly forecasts for several upcoming time slots
  * likely range            - "likely_low"/"likely_high" columns from error_bands.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import FEATURE_COLUMNS, add_time_features

UNKNOWN_WEATHER = "Unknown"       # the label the cleaning step gives to missing weather
_MISSING_WEATHER_WORDS = {"", "unknown", "none", "nan", "not sure", "n/a", "na", "null"}
_TRUE_WORDS = {"1", "true", "yes", "y"}
_FALSE_WORDS = {"", "0", "false", "no", "n", "none", "nan"}


def load_model(path=None):
    """Load the trained pipeline saved by train.py."""
    path = Path(path) if path else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}\nRun `python main.py` to train it first.")
    return joblib.load(path)


def event_flag_for_hour(hour, event_today):
    """special_event is 1 only during the evening event hours of an event day.

    This matches how the training data was defined, so we never ask the model
    about a situation (e.g. an event at 9 AM) it has not seen.
    """
    return int(bool(event_today) and config.EVENT_START_HOUR <= hour <= config.EVENT_END_HOUR)


def status_label(rate):
    """Status for an occupancy rate (0-1): Available, Limited or Full."""
    if rate >= config.STATUS_FULL_FROM:
        return "Full"
    if rate >= config.STATUS_LIMITED_FROM:
        return "Limited"
    return "Available"


# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
def _is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT


def normalize_request(request):
    """Clean ONE prediction request. Returns (clean_request, notes).

    Missing weather becomes "Unknown" (a category the model saw in training);
    a missing event flag means "no event". A missing lot or date/time, or a value
    that is not valid, raises ValueError with a message that says what to fix.
    `notes` lists the automatic corrections, so the app can tell the user.
    """
    notes = []

    lot = request.get("parking_lot_id")
    if _is_missing(lot) or not str(lot).strip():
        raise ValueError("No parking location was given. "
                         f"Choose one of: {', '.join(config.PARKING_LOTS)}.")
    lot = str(lot).strip().upper()
    if lot not in config.PARKING_LOTS:
        raise ValueError(f"Unknown parking lot '{lot}'. Choose from: {list(config.PARKING_LOTS)}")

    when = request.get("timestamp")
    if _is_missing(when):
        raise ValueError("No date and time was given.")
    try:
        when = pd.Timestamp(when)
    except (ValueError, TypeError):
        raise ValueError(f"The date and time '{when}' could not be understood.") from None
    if pd.isna(when):
        raise ValueError("No valid date and time was given.")
    if when.tzinfo is not None:
        when = when.tz_localize(None)

    raw_weather = request.get("weather")
    if _is_missing(raw_weather) or not str(raw_weather).strip():
        weather = UNKNOWN_WEATHER
        notes.append("Weather was not given, so 'Unknown' was used (the model then averages over weather).")
    elif str(raw_weather).strip().lower() in _MISSING_WEATHER_WORDS:
        weather = UNKNOWN_WEATHER                 # the user said "not sure" on purpose
    else:
        weather = str(raw_weather).strip().title()
        if weather not in config.WEATHER_OPTIONS:
            raise ValueError(f"Unknown weather '{weather}'. Choose from: {config.WEATHER_OPTIONS} "
                             f"(or leave it out / '{UNKNOWN_WEATHER}' if you are not sure).")

    event = request.get("event_today", False)
    if _is_missing(event):
        event = False
    elif isinstance(event, str):
        word = event.strip().lower()
        if word in _TRUE_WORDS:
            event = True
        elif word in _FALSE_WORDS:
            event = False
        else:
            raise ValueError(f"Could not understand event_today='{event}'. Use True or False.")
    else:
        event = bool(event)

    clean = {"parking_lot_id": lot, "timestamp": when, "weather": weather, "event_today": event}
    return clean, notes


def prepare_features(requests):
    """Turn request dictionaries into the exact feature table the model expects.

    `requests` is a list of dictionaries with the keys:
        parking_lot_id, timestamp, weather, event_today
    (weather and event_today may be missing). Returns (request_frame, feature_frame).
    The feature frame uses the same add_time_features() and FEATURE_COLUMNS as training.
    """
    if requests is None or len(requests) == 0:
        raise ValueError("No prediction requests were given.")
    cleaned = [normalize_request(r)[0] for r in requests]
    frame = pd.DataFrame(cleaned)
    frame["special_event"] = [event_flag_for_hour(ts.hour, flag)
                              for ts, flag in zip(frame["timestamp"], frame["event_today"])]
    features = add_time_features(frame)[FEATURE_COLUMNS]
    return frame, features


# ----------------------------------------------------------------------------
# "Likely range" (typical error per lot, measured on training data only)
# ----------------------------------------------------------------------------
_BANDS_CACHE = {}


def load_error_bands(path=None):
    """{lot_id: typical error in spaces} or {} when train.py has not written the file."""
    path = Path(path) if path else config.ERROR_BANDS_PATH
    if not path.exists():
        return {}
    stamp = path.stat().st_mtime
    if _BANDS_CACHE.get(str(path), (None,))[0] != stamp:
        data = json.loads(path.read_text(encoding="utf-8"))
        _BANDS_CACHE[str(path)] = (stamp, {lot: float(v["p80_spaces"])
                                           for lot, v in data.get("lots", {}).items()})
    return _BANDS_CACHE[str(path)][1]


def predict_many(requests, model=None):
    """Predict for many situations at once. Returns a DataFrame, one row per request.

    Guarantee: 0 <= predicted_available_spaces <= total_spaces (and the same for occupied).
    """
    model = model or load_model()
    frame, features = prepare_features(requests)
    rate = np.asarray(model.predict(features), dtype=float)
    if not np.all(np.isfinite(rate)):
        raise ValueError("The model returned an invalid (non-numeric) prediction. "
                         "Run `python main.py` to retrain it.")
    rate = np.clip(rate, 0.0, 1.0)

    total = np.array([config.PARKING_LOTS[lot]["total_spaces"] for lot in frame["parking_lot_id"]])
    occupied = np.clip(np.round(rate * total).astype(int), 0, total)
    available = np.clip(total - occupied, 0, total)

    result = pd.DataFrame({
        "timestamp": frame["timestamp"],
        "parking_lot_id": frame["parking_lot_id"],
        "location": [config.PARKING_LOTS[lot]["location"] for lot in frame["parking_lot_id"]],
        "total_spaces": total,
        "predicted_occupancy_rate": rate.round(4),
        "predicted_occupied_spaces": occupied,
        "predicted_available_spaces": available,
        "status": [status_label(r) for r in rate],
    })

    bands = load_error_bands()
    if bands:
        band = np.array([bands.get(lot, np.nan) for lot in frame["parking_lot_id"]])
        if not np.isnan(band).any():
            result["likely_low"] = np.clip(np.floor(available - band), 0, total).astype(int)
            result["likely_high"] = np.clip(np.ceil(available + band), 0, total).astype(int)
    return result


def predict_occupancy(lot_id, timestamp, weather="Sunny", event_today=False, model=None):
    """Predict for ONE lot at ONE date/time. Returns a dictionary."""
    result = predict_many([{"parking_lot_id": lot_id, "timestamp": timestamp,
                            "weather": weather, "event_today": event_today}], model=model)
    return result.iloc[0].to_dict()


def predict_full_day(lot_id, date, weather="Sunny", event_today=False, model=None):
    """Predict all 24 hours of one day for one lot. Returns a DataFrame."""
    day = pd.Timestamp(date).normalize()
    requests = [{"parking_lot_id": lot_id, "timestamp": day + pd.Timedelta(hours=h),
                 "weather": weather, "event_today": event_today} for h in range(24)]
    return predict_many(requests, model=model)


def lot_supports_events(lot_id):
    """Only lots with event_prob > 0 have special events in the data."""
    return config.PARKING_LOTS[lot_id]["event_prob"] > 0


def predict_slots(lot_ids, start, hours, weather=None, event_today=False, model=None,
                  max_hours=config.MAX_SLOT_HOURS):
    """Hourly forecast for several upcoming time slots.

    lot_ids     - one or more lot ids
    start       - first slot (rounded down to the full hour)
    hours       - how many hourly slots (1 .. max_hours)
    weather     - the SAME weather is assumed for every slot (the app has no weather
                  forecast), None = unknown
    event_today - a special event happens on each forecast day (only lots that have events)
    Returns a DataFrame with one row per lot and slot (see predict_many) plus `slot`
    (0, 1, 2 ...) and `slot_label`.
    """
    if isinstance(lot_ids, str):
        lot_ids = [lot_ids]
    lot_ids = list(lot_ids or [])
    if not lot_ids:
        raise ValueError("Choose at least one parking location.")
    try:
        hours = int(hours)
    except (TypeError, ValueError):
        raise ValueError("The number of hours must be a whole number.") from None
    if not 1 <= hours <= max_hours:
        raise ValueError(f"Choose between 1 and {max_hours} hours (got {hours}).")
    if _is_missing(start):
        raise ValueError("No start date and time was given.")
    try:
        first = pd.Timestamp(start).floor(pd.Timedelta(hours=1))
    except (ValueError, TypeError):
        raise ValueError(f"The start time '{start}' could not be understood.") from None
    if pd.isna(first):
        raise ValueError("No valid start date and time was given.")

    requests = []
    for lot in lot_ids:
        key = str(lot).strip().upper()
        flag = bool(event_today) and key in config.PARKING_LOTS and lot_supports_events(key)
        for slot in range(hours):
            requests.append({"parking_lot_id": lot, "timestamp": first + pd.Timedelta(hours=slot),
                             "weather": weather, "event_today": flag})
    result = predict_many(requests, model=model)
    result["slot"] = ((result["timestamp"] - first) / pd.Timedelta(hours=1)).astype(int)
    result["slot_label"] = result["timestamp"].dt.strftime("%a %d %b, %H:00")
    return result.sort_values(["slot", "parking_lot_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Predict parking availability (model trained on the project data).")
    parser.add_argument("--lot", required=True, choices=list(config.PARKING_LOTS))
    parser.add_argument("--datetime", required=True, help='e.g. "2026-11-07 15:00"')
    parser.add_argument("--weather", default="Sunny", choices=config.WEATHER_OPTIONS + [UNKNOWN_WEATHER])
    parser.add_argument("--event", action="store_true", help="a special event is happening today")
    parser.add_argument("--hours", type=int, default=1, help="forecast this many hourly slots")
    args = parser.parse_args()

    if args.hours > 1:
        table = predict_slots(args.lot, args.datetime, args.hours, args.weather, args.event)
        print(table[["slot_label", "predicted_available_spaces", "total_spaces",
                     "predicted_occupancy_rate", "status"]].to_string(index=False))
    else:
        r = predict_occupancy(args.lot, args.datetime, args.weather, args.event)
        print(f"Parking lot : {r['parking_lot_id']} - {r['location']}")
        print(f"Date & time : {r['timestamp']}")
        print(f"Predicted   : {r['predicted_available_spaces']} of {r['total_spaces']} spaces free "
              f"({r['predicted_occupancy_rate'] * 100:.0f}% occupied) -> {r['status']}")
        if "likely_low" in r:
            print(f"Typical range: about {r['likely_low']}-{r['likely_high']} free")
    print("(An ESTIMATE from a model trained on the project's data - not a guarantee of real-time availability.)")


if __name__ == "__main__":
    main()
