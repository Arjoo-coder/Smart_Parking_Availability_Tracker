"""Generate a SIMULATED (synthetic) parking dataset.

IMPORTANT: none of this data is real. It is produced by simple rules
(daily patterns + weather + events + random noise) so that the project can be
built and tested without a real parking sensor feed.

How one row is created
----------------------
1. Every parking lot has a typical daily curve (e.g. an office lot fills up
   at 9-5 on weekdays, a mall fills up in the afternoon on weekends).
2. We add small effects for weather and special events.
3. We add random noise so that the data is not perfectly predictable.
4. occupied_spaces = round(occupancy_rate * total_spaces), and
   available_spaces = total_spaces - occupied_spaces, so the numbers always
   add up.
5. A few realistic "data problems" (missing weather values and duplicate
   rows) are added on purpose, so the cleaning step has something to do.
"""

import numpy as np
import pandas as pd

from src import config

# Weather changes slowly. This table says: "if it is Sunny now, the next
# 6-hour block is Sunny with 70%, Cloudy with 20% ..." (same order as WEATHER_OPTIONS).
WEATHER_TRANSITIONS = {
    "Sunny":  [0.70, 0.20, 0.07, 0.03],
    "Cloudy": [0.25, 0.45, 0.25, 0.05],
    "Rainy":  [0.10, 0.35, 0.50, 0.05],
    "Foggy":  [0.25, 0.35, 0.10, 0.30],
}

# How weather shifts occupancy (in occupancy-rate points) for each lot type.
# Example: rain keeps people away from the mall (-0.06) but pushes more
# commuters to park at the station (+0.06).
WEATHER_EFFECT = {
    "office":   {"Sunny": 0.0, "Cloudy": 0.0, "Rainy": 0.04, "Foggy": 0.01},
    "mall":     {"Sunny": 0.0, "Cloudy": 0.0, "Rainy": -0.06, "Foggy": -0.03},
    "hospital": {"Sunny": 0.0, "Cloudy": 0.0, "Rainy": 0.02, "Foggy": 0.0},
    "station":  {"Sunny": 0.0, "Cloudy": 0.0, "Rainy": 0.06, "Foggy": 0.02},
}


# ----------------------------------------------------------------------------
# Small math helpers used to draw the daily curves
# ----------------------------------------------------------------------------
def _bump(hour, center, width):
    """A smooth hill that peaks at `center` hour (value between 0 and 1)."""
    return np.exp(-0.5 * ((hour - center) / width) ** 2)


def _sigmoid(x):
    """A smooth step from 0 to 1."""
    return 1.0 / (1.0 + np.exp(-x))


def base_occupancy(lot_type, hour, is_weekend):
    """Typical occupancy rate (0-1) for a lot type at a given hour.

    `hour` and `is_weekend` are numpy arrays of the same length.
    """
    h = np.asarray(hour, dtype=float)

    if lot_type == "office":
        weekday = 0.07 + 0.83 * _bump(h, 12.5, 3.2)      # busy 9-5 on weekdays
        weekend = 0.05 + 0.15 * _bump(h, 13.0, 3.0)      # nearly empty at weekends
    elif lot_type == "mall":
        weekday = 0.05 + 0.55 * _bump(h, 17.0, 3.5)      # evening shoppers
        weekend = 0.08 + 0.85 * _bump(h, 15.5, 3.8)      # very busy weekend afternoons
    elif lot_type == "hospital":
        weekday = 0.25 + 0.60 * _bump(h, 11.0, 3.5)      # steady, peaks late morning
        weekend = 0.20 + 0.40 * _bump(h, 11.0, 3.5)
    elif lot_type == "station":
        # Park-and-ride: fills up in the morning rush, empties in the evening.
        weekday = 0.10 + 0.80 * (_sigmoid((h - 7.0) / 0.8) - _sigmoid((h - 19.0) / 1.0))
        weekend = 0.08 + 0.25 * _bump(h, 13.0, 4.0)
    else:
        raise ValueError(f"Unknown parking lot type: {lot_type}")

    return np.where(is_weekend, weekend, weekday)


# ----------------------------------------------------------------------------
# Weather (shared by all lots - they are all in the same city)
# ----------------------------------------------------------------------------
def generate_city_weather(timestamps, rng):
    """Return one weather label per timestamp. Weather changes every 6 hours."""
    states = config.WEATHER_OPTIONS
    current = "Sunny"
    weather_by_block = {}

    for date in timestamps.normalize().unique():
        for block in range(4):  # 4 blocks of 6 hours per day
            current = str(rng.choice(states, p=WEATHER_TRANSITIONS[current]))
            weather_by_block[(date, block)] = current

    dates = timestamps.normalize()
    blocks = timestamps.hour // 6
    return np.array([weather_by_block[(d, b)] for d, b in zip(dates, blocks)])


# ----------------------------------------------------------------------------
# One parking lot
# ----------------------------------------------------------------------------
def generate_lot_records(lot_id, lot, timestamps, weather, rng):
    """Create the hourly records of ONE parking lot."""
    hours = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()        # Monday=0 ... Sunday=6
    is_weekend = day_of_week >= 5
    dates = timestamps.normalize()
    n = len(timestamps)

    # 1) typical daily pattern
    base = base_occupancy(lot["type"], hours, is_weekend)

    # 2) weather effect
    effect_table = WEATHER_EFFECT[lot["type"]]
    weather_effect = np.array([effect_table[w] for w in weather])

    # 3) special events: random days, active only in the evening hours
    unique_dates = dates.unique()
    event_days = {d for d in unique_dates if rng.random() < lot["event_prob"]}
    in_event_hours = (hours >= config.EVENT_START_HOUR) & (hours <= config.EVENT_END_HOUR)
    on_event_day = np.array([d in event_days for d in dates])
    special_event = (in_event_hours & on_event_day).astype(int)

    # 4) some days are simply busier or quieter than others
    day_offsets = {d: rng.normal(0.0, 0.04) for d in unique_dates}
    day_effect = np.array([day_offsets[d] for d in dates])

    # 5) random noise that carries over a little from one hour to the next
    shocks = rng.normal(0.0, 0.03, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + shocks[i]

    # combine everything and keep the rate between 0% and 100%
    rate = base + weather_effect + special_event * lot["event_boost"] + day_effect + noise
    rate = np.clip(rate, 0.0, 1.0)

    # make the numbers consistent: occupied + available = total
    total = lot["total_spaces"]
    occupied = np.round(rate * total).astype(int)
    available = total - occupied

    return pd.DataFrame({
        "timestamp": timestamps,
        "parking_lot_id": lot_id,
        "location": lot["location"],
        "total_spaces": total,
        "occupied_spaces": occupied,
        "available_spaces": available,
        "occupancy_rate": np.round(occupied / total, 4),
        "hour": hours,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend.astype(int),
        "weather": weather,
        "special_event": special_event,
    })


def check_consistency(df):
    """Raise an error if the occupancy numbers do not add up."""
    assert (df["occupied_spaces"] + df["available_spaces"] == df["total_spaces"]).all(), \
        "occupied + available must equal total"
    assert (df["occupied_spaces"] >= 0).all() and (df["available_spaces"] >= 0).all(), \
        "negative space counts found"
    assert (df["occupied_spaces"] <= df["total_spaces"]).all(), "occupied > total"


def generate_dataset(seed=config.RANDOM_SEED):
    """Generate the clean simulated dataset (all lots, all hours)."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=config.START_DATE,
                               periods=config.NUM_DAYS * 24,
                               freq=pd.Timedelta(hours=1))
    weather = generate_city_weather(timestamps, rng)

    frames = [generate_lot_records(lot_id, lot, timestamps, weather, rng)
              for lot_id, lot in config.PARKING_LOTS.items()]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["timestamp", "parking_lot_id"]).reset_index(drop=True)

    check_consistency(df)
    return df


def inject_data_issues(df, seed=config.RANDOM_SEED):
    """Add a few realistic problems so the cleaning step has real work to do.

    - about 1% of the weather values become missing (NaN)
    - 30 rows are duplicated (e.g. a sensor sent the same reading twice)
    """
    rng = np.random.default_rng(seed + 1)
    df = df.copy()

    n_missing = int(0.01 * len(df))
    missing_rows = rng.choice(df.index.to_numpy(), size=n_missing, replace=False)
    df.loc[missing_rows, "weather"] = np.nan

    duplicates = df.sample(n=30, random_state=seed)
    return pd.concat([df, duplicates], ignore_index=True)


def generate_and_save(path=None, seed=config.RANDOM_SEED):
    """Generate the dataset and write it to data/raw/ as a CSV file."""
    config.ensure_directories()
    path = path or config.RAW_DATA_PATH
    df = generate_dataset(seed)
    df = inject_data_issues(df, seed)
    df.to_csv(path, index=False)

    print(f"[data] SIMULATED dataset saved: {path}")
    print(f"[data] {len(df):,} rows, {df['parking_lot_id'].nunique()} lots, "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


if __name__ == "__main__":
    generate_and_save()
