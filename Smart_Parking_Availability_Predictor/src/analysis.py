"""Historical analysis of the parking data: peak hours, weekday vs weekend, trends, weather.

Plain Pandas, no Streamlit. Every function takes the processed dataset (with an
`occupancy_pct` column, see dashboard_data.load_dashboard_data) and returns a DataFrame.
Remember: with simulated data these patterns show what the SIMULATION contains.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def hour_ranges(hours):
    """[9, 10, 11, 14] -> '09:00-12:00, 14:00-15:00' (each range ends at the end of its last hour)."""
    hours = sorted(set(int(h) for h in hours))
    if not hours:
        return "-"
    ranges, start, previous = [], hours[0], hours[0]
    for hour in hours[1:]:
        if hour != previous + 1:
            ranges.append((start, previous))
            start = hour
        previous = hour
    ranges.append((start, previous))
    return ", ".join(f"{a:02d}:00-{(b + 1) % 24:02d}:00" for a, b in ranges)


# ----------------------------------------------------------------------------
# 1) Peak hours
# ----------------------------------------------------------------------------
def peak_hours(df, share_of_peak=0.90):
    """One row per location: busiest hour, and the 'peak window' of hours within 90% of the peak.

    Weekday and weekend peaks are reported separately because they often differ.
    """
    rows = []
    for (lot_id, location), group in df.groupby(["parking_lot_id", "location"]):
        def summarize(data):
            by_hour = data.groupby("hour")["occupancy_pct"].mean()
            peak = by_hour.max()
            window = by_hour[by_hour >= peak * share_of_peak].index
            return int(by_hour.idxmax()), float(peak), hour_ranges(window), float(by_hour.min()), int(by_hour.idxmin())
        a = summarize(group)
        wd = summarize(group[group["is_weekend"] == 0])
        we = summarize(group[group["is_weekend"] == 1])
        rows.append({
            "parking_lot_id": lot_id, "location": location,
            "peak_hour": a[0], "peak_occupancy_pct": a[1], "peak_window": a[2],
            "quietest_hour": a[4], "quietest_occupancy_pct": a[3],
            "weekday_peak_hour": wd[0], "weekday_peak_pct": wd[1], "weekday_peak_window": wd[2],
            "weekend_peak_hour": we[0], "weekend_peak_pct": we[1], "weekend_peak_window": we[2],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2) Weekday vs weekend
# ----------------------------------------------------------------------------
def weekday_weekend_summary(df):
    """Average and peak occupancy on weekdays vs weekends, per location."""
    rows = []
    for (lot_id, location), group in df.groupby(["parking_lot_id", "location"]):
        weekday = group[group["is_weekend"] == 0]
        weekend = group[group["is_weekend"] == 1]
        wd_avg, we_avg = weekday["occupancy_pct"].mean(), weekend["occupancy_pct"].mean()
        rows.append({
            "parking_lot_id": lot_id, "location": location,
            "weekday_avg_pct": wd_avg, "weekend_avg_pct": we_avg,
            "difference_pts": we_avg - wd_avg,
            "weekday_peak_pct": weekday.groupby("hour")["occupancy_pct"].mean().max(),
            "weekend_peak_pct": weekend.groupby("hour")["occupancy_pct"].mean().max(),
            "busier_on": "Weekends" if we_avg > wd_avg else "Weekdays",
        })
    return pd.DataFrame(rows)


def weekday_weekend_hourly(df, location):
    """Average occupancy by hour for weekdays and weekends of ONE location."""
    data = df[df["location"] == location].copy()
    data["day_type"] = np.where(data["is_weekend"] == 1, "Weekend", "Weekday")
    return data.groupby(["day_type", "hour"])["occupancy_pct"].mean().reset_index()


# ----------------------------------------------------------------------------
# 3) Trends over time, per location
# ----------------------------------------------------------------------------
def daily_trend(df, window=7):
    """Daily average occupancy per location plus a rolling average (smooths the weekly cycle)."""
    daily = (df.assign(date=df["timestamp"].dt.floor("D"))
               .groupby(["parking_lot_id", "location", "date"])["occupancy_pct"].mean().reset_index())
    daily["rolling_avg"] = (daily.groupby("location")["occupancy_pct"]
                                 .transform(lambda s: s.rolling(window, min_periods=window).mean()))
    return daily


def trend_summary(daily, min_slope_pts_per_week=1.0, alpha=0.05):
    """Is any location getting busier or quieter over the period? (straight-line fit per location)

    A trend is only reported when the slope is statistically clear AND not tiny.
    """
    rows = []
    for location, group in daily.groupby("location"):
        group = group.sort_values("date")
        days = (group["date"] - group["date"].min()).dt.days.values
        if len(group) < 14:
            rows.append({"location": location, "slope_pts_per_week": np.nan, "p_value": np.nan,
                         "verdict": "Not enough days to judge a trend"})
            continue
        fit = stats.linregress(days, group["occupancy_pct"].values)
        slope = fit.slope * 7
        if fit.pvalue < alpha and abs(slope) >= min_slope_pts_per_week:
            verdict = "Getting busier" if slope > 0 else "Getting quieter"
        else:
            verdict = "No clear trend"
        rows.append({"location": location, "slope_pts_per_week": float(slope),
                     "p_value": float(fit.pvalue), "verdict": verdict,
                     "first_week_avg_pct": float(group["occupancy_pct"].head(7).mean()),
                     "last_week_avg_pct": float(group["occupancy_pct"].tail(7).mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4) Weather (and events), compared FAIRLY
# ----------------------------------------------------------------------------
def _typical_occupancy(df):
    """Expected occupancy for each lot, hour and weekday/weekend, from no-event hours.

    Weather is not random across the day (it changes in 6-hour blocks), so a plain
    average by weather would mix up 'rainy' with 'rainy afternoons'. Comparing each
    reading with the TYPICAL value for its lot, hour and day type removes that mix-up.
    """
    base = (df[df["special_event"] == 0]
            .groupby(["parking_lot_id", "hour", "is_weekend"])["occupancy_pct"].mean()
            .rename("typical_pct").reset_index())
    merged = df.merge(base, on=["parking_lot_id", "hour", "is_weekend"], how="left")
    merged["vs_typical_pts"] = merged["occupancy_pct"] - merged["typical_pct"]
    return merged


def weather_effect(df, min_rows=30):
    """Average occupancy difference (percentage points) vs the typical hour, by lot and weather.

    Only no-event hours are used, so events do not distort the weather comparison.
    """
    data = _typical_occupancy(df)
    data = data[data["special_event"] == 0]
    table = (data.groupby(["location", "weather"])
                 .agg(raw_avg_pct=("occupancy_pct", "mean"), vs_typical_pts=("vs_typical_pts", "mean"),
                      std_err=("vs_typical_pts", lambda s: s.std() / np.sqrt(len(s)) if len(s) > 1 else np.nan),
                      hours=("occupancy_pct", "size")).reset_index())
    return table[table["hours"] >= min_rows].reset_index(drop=True)


def event_effect(df):
    """Occupancy difference vs typical during the event hours, by lot (event hours vs non-event hours)."""
    data = _typical_occupancy(df)
    window = data[data["hour"].between(config.EVENT_START_HOUR, config.EVENT_END_HOUR)]
    table = (window.groupby(["location", "special_event"])
                   .agg(avg_pct=("occupancy_pct", "mean"), vs_typical_pts=("vs_typical_pts", "mean"),
                        hours=("occupancy_pct", "size")).reset_index())
    table["condition"] = table["special_event"].map({0: "No event", 1: "Special event"})
    return table.drop(columns="special_event")
