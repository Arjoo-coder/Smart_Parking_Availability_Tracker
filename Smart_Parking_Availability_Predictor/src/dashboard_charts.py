"""Matplotlib charts for the Streamlit dashboard (Phase 2).

Every function takes a DataFrame prepared by src/dashboard_data.py and
returns a Matplotlib Figure. There is no Streamlit code here, so the charts
can be tested and saved from a normal Python script as well.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

# --- one consistent colour per parking lot (colour-blind friendly) -----------
_PALETTE = ["#2a9d8f", "#e76f51", "#264653", "#7b6cc4", "#d4a017", "#4c9be8"]
LOT_COLORS = {lot_id: _PALETTE[i % len(_PALETTE)]
              for i, lot_id in enumerate(config.PARKING_LOTS)}
LOCATION_COLORS = {info["location"]: LOT_COLORS[lot_id]
                   for lot_id, info in config.PARKING_LOTS.items()}

# --- status colours (same as the badges in the app) -----------------------
STATUS_COLORS = {"Available": "#2e9e5b", "Limited": "#e0a100", "Full": "#d64545"}
FREE_COLOR = "#dde5ec"
INK = "#1f2933"
MUTED = "#6b7785"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c3ccd5",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,          # grid lines stay behind bars
    "grid.color": "#e7ecf0",
    "grid.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
})


def _color(location):
    return LOCATION_COLORS.get(location, "#4c9be8")


def _threshold_lines(ax, limited_label=True):
    """Dashed lines for the Limited / Full thresholds on a 0-100% axis."""
    limited = config.STATUS_LIMITED_FROM * 100
    full = config.STATUS_FULL_FROM * 100
    ax.axhline(limited, color=STATUS_COLORS["Limited"], linestyle="--", linewidth=1, alpha=0.8)
    ax.axhline(full, color=STATUS_COLORS["Full"], linestyle="--", linewidth=1, alpha=0.8)
    if limited_label:
        ax.text(0.995, limited + 1, "Limited", color=STATUS_COLORS["Limited"], fontsize=8,
                ha="right", transform=ax.get_yaxis_transform())
        ax.text(0.995, full + 1, "Full", color=STATUS_COLORS["Full"], fontsize=8,
                ha="right", transform=ax.get_yaxis_transform())


def _empty_figure(message):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, fontsize=12)
    return fig


# ----------------------------------------------------------------------------
# Home page
# ----------------------------------------------------------------------------
def fig_last_24_hours(df, end_timestamp):
    """Occupancy of every location (and overall) over the 24 readings up to the snapshot."""
    end = pd.Timestamp(end_timestamp)
    window = df[(df["timestamp"] > end - pd.Timedelta(hours=24)) & (df["timestamp"] <= end)]
    if window.empty:
        return _empty_figure("No readings available for this period")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for location, group in window.groupby("location"):
        ax.plot(group["timestamp"], group["occupancy_pct"], linewidth=1.4, alpha=0.8,
                color=_color(location), label=location)
    overall = window.groupby("timestamp").agg(occ=("occupied_spaces", "sum"), tot=("total_spaces", "sum"))
    ax.plot(overall.index, overall["occ"] / overall["tot"] * 100, color=INK, linewidth=2.6,
            label="All locations")
    ax.set(ylim=(0, 100), ylabel="Occupancy (%)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_title("Occupancy over the 24 hours up to the snapshot", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Availability page
# ----------------------------------------------------------------------------
def fig_snapshot_stacked(snapshot):
    """Occupied vs available spaces for each location at the snapshot."""
    data = snapshot.sort_values("total_spaces")
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(data) + 1.4))
    y = np.arange(len(data))
    colors = [STATUS_COLORS[s] for s in data["status"]]
    ax.barh(y, data["occupied_spaces"], color=colors, label="Occupied")
    ax.barh(y, data["available_spaces"], left=data["occupied_spaces"], color=FREE_COLOR,
            label="Available")
    for i, row in enumerate(data.itertuples()):
        ax.text(row.total_spaces + 6, i,
                f"{row.available_spaces} free ({row.occupancy_pct:.0f}% full)",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(data["location"])
    ax.set_xlim(0, data["total_spaces"].max() * 1.35)
    ax.set(xlabel="Parking spaces")
    ax.grid(axis="y", visible=False)
    ax.set_title("Occupied vs available spaces (bar colour = status)", loc="left")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Charts & Analytics page
# ----------------------------------------------------------------------------
def fig_hourly_demand(profile, metric="occupancy_pct"):
    """Average demand for each hour of the day, one line per location."""
    if profile.empty:
        return _empty_figure("Select at least one location")
    labels = {"occupancy_pct": "Average occupancy (%)",
              "occupied_spaces": "Average occupied spaces",
              "available_spaces": "Average available spaces"}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for location, group in profile.groupby("location"):
        ax.plot(group["hour"], group[metric], marker="o", markersize=3.5, linewidth=2,
                color=_color(location), label=location)
    if metric == "occupancy_pct":
        ax.set_ylim(0, 100)
        _threshold_lines(ax)
    ax.set_xticks(range(0, 24, 2))
    ax.set(xlabel="Hour of day", ylabel=labels[metric])
    ax.set_title("Hourly parking demand", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def fig_day_of_week(profile):
    """Grouped bars: average occupancy for each weekday and location."""
    if profile.empty:
        return _empty_figure("Select at least one location")
    locations = list(profile["location"].unique())
    width = 0.8 / len(locations)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, location in enumerate(locations):
        group = profile[profile["location"] == location].sort_values("day_of_week")
        offset = (i - (len(locations) - 1) / 2) * width
        ax.bar(group["day_of_week"] + offset, group["occupancy_pct"], width=width * 0.92,
               color=_color(location), label=location)
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_NAMES)
    ax.set(ylim=(0, 100), ylabel="Average occupancy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Occupancy by day of the week", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def fig_weekday_hour_heatmap(df, location):
    """Weekday x hour heatmap for ONE location."""
    data = df[df["location"] == location]
    if data.empty:
        return _empty_figure("No data for this location")
    pivot = data.pivot_table(index="day_of_week", columns="hour", values="occupancy_pct",
                             aggfunc="mean")
    pivot.index = [DAY_NAMES[i] for i in pivot.index]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=100, linewidths=0.4,
                linecolor="white", cbar_kws={"label": "% occupied"})
    ax.set(xlabel="Hour of day", ylabel="")
    ax.set_title(f"{location}: when is it busy?", loc="left")
    ax.grid(False)
    fig.tight_layout()
    return fig


_COMPARISON_LABELS = {
    "avg_occupancy_pct": "Average occupancy (%)",
    "peak_occupancy_pct": "Peak hourly occupancy (%)",
    "avg_available_spaces": "Average available spaces",
    "total_spaces": "Total spaces",
}


def fig_location_comparison(summary, metric="avg_occupancy_pct"):
    """Horizontal bars comparing locations on one metric."""
    data = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(data) + 1.4))
    ax.barh(data["location"], data[metric], color=[_color(loc) for loc in data["location"]])
    maximum = data[metric].max()
    for i, value in enumerate(data[metric]):
        text = f"{value:.0f}%" if metric.endswith("_pct") else f"{value:.0f}"
        ax.text(value + maximum * 0.015, i, text, va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 100 if metric.endswith("_pct") else maximum * 1.18)
    ax.set(xlabel=_COMPARISON_LABELS.get(metric, metric))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Location comparison: {_COMPARISON_LABELS.get(metric, metric).lower()}", loc="left")
    fig.tight_layout()
    return fig


def fig_actual_vs_predicted_series(window, location):
    """Actual vs predicted AVAILABLE spaces over time for one location."""
    if window.empty:
        return _empty_figure("No test data in this period")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(window["timestamp"], window["actual_available_spaces"], color=INK, linewidth=2,
            label="Actual")
    ax.plot(window["timestamp"], window["predicted_available_spaces"], color="#e76f51",
            linewidth=1.8, linestyle="--", label="Predicted")
    ax.set(ylabel="Available spaces", ylim=(0, None))
    ax.set_title(f"{location}: actual vs predicted available spaces", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_actual_vs_predicted_scatter(window, location):
    """Every hour as one dot: closer to the diagonal = better prediction."""
    if window.empty:
        return _empty_figure("No test data in this period")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(window["actual_available_spaces"], window["predicted_available_spaces"], s=14,
               alpha=0.5, color=_color(location))
    top = max(window["actual_available_spaces"].max(), window["predicted_available_spaces"].max()) * 1.05
    ax.plot([0, top], [0, top], color=INK, linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set(xlabel="Actual available spaces", ylabel="Predicted available spaces",
           xlim=(0, top), ylim=(0, top))
    ax.set_title("Hour by hour", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Predict page
# ----------------------------------------------------------------------------
def fig_forecast_day(day, hour, total_spaces, title):
    """Predicted free spaces through one day, with the selected hour marked."""
    hours = day["timestamp"].dt.hour
    free = day["predicted_available_spaces"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, free, color=LOT_COLORS.get(day["parking_lot_id"].iloc[0], "#2a9d8f"), linewidth=2.4)
    ax.fill_between(hours, free, alpha=0.12, color="#2a9d8f")

    # thresholds expressed in free spaces
    limited_free = total_spaces * (1 - config.STATUS_LIMITED_FROM)
    full_free = total_spaces * (1 - config.STATUS_FULL_FROM)
    ax.axhline(limited_free, color=STATUS_COLORS["Limited"], linestyle="--", linewidth=1)
    ax.axhline(full_free, color=STATUS_COLORS["Full"], linestyle="--", linewidth=1)
    ax.text(23, limited_free + total_spaces * 0.015, "Limited below this", color=STATUS_COLORS["Limited"],
            fontsize=8, ha="right")
    ax.text(23, full_free + total_spaces * 0.015, "Full below this", color=STATUS_COLORS["Full"],
            fontsize=8, ha="right")

    selected = int(free[hours == hour].iloc[0])
    ax.scatter([hour], [selected], s=90, color="#e76f51", zorder=5)
    ax.annotate(f"{selected} free", (hour, selected), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=9, color="#e76f51", fontweight="bold")
    ax.set(xlim=(0, 23), ylim=(0, total_spaces), xlabel="Hour of day", ylabel="Predicted free spaces")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def fig_prediction_all_lots(predictions):
    """Predicted occupancy of every location for the same date and time."""
    data = predictions.sort_values("predicted_occupancy_rate")
    fig, ax = plt.subplots(figsize=(6.5, 0.8 * len(data) + 1.6))
    pct = data["predicted_occupancy_rate"] * 100
    ax.barh(data["location"], pct, color=[STATUS_COLORS[s] for s in data["status"]])
    for i, (value, free) in enumerate(zip(pct, data["predicted_available_spaces"])):
        label = f"{value:.0f}% - {free} free"
        if value >= 30:      # long bar: write the label inside it, in white
            ax.text(value - 1.5, i, label, va="center", ha="right", fontsize=9,
                    color="white", fontweight="bold")
        else:                # short bar: write the label to its right
            ax.text(value + 1.5, i, label, va="center", ha="left", fontsize=9, color=INK)
    ax.axvline(config.STATUS_LIMITED_FROM * 100, color=STATUS_COLORS["Limited"], linestyle="--", linewidth=1)
    ax.axvline(config.STATUS_FULL_FROM * 100, color=STATUS_COLORS["Full"], linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), xlabel="Predicted occupancy (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("All locations at the same time", loc="left")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Phase 3 charts
# ----------------------------------------------------------------------------
def fig_slot_forecast(slots, selected_lots=None):
    """Predicted free spaces for upcoming hourly slots, one line per lot (with a likely range)."""
    if slots.empty:
        return _empty_figure("No forecast available")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for lot_id, group in slots.groupby("parking_lot_id"):
        group = group.sort_values("timestamp")
        color = LOT_COLORS.get(lot_id, "#4c9be8")
        ax.plot(group["timestamp"], group["predicted_available_spaces"], color=color, linewidth=2.2,
                marker="o", markersize=3, label=group["location"].iloc[0])
        if "likely_low" in group:
            ax.fill_between(group["timestamp"], group["likely_low"], group["likely_high"],
                            color=color, alpha=0.12, linewidth=0)
    ax.set(ylim=(0, None), ylabel="Predicted free spaces")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %H:%M"))
    ax.set_title("Hourly availability forecast (shaded = typical error range)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_slot_heat(slots):
    """Lots x time slots, coloured by predicted occupancy; the numbers are free spaces."""
    if slots.empty:
        return _empty_figure("No forecast available")
    pivot_rate = slots.pivot_table(index="location", columns="slot_label", values="predicted_occupancy_rate",
                                   sort=False) * 100
    pivot_free = slots.pivot_table(index="location", columns="slot_label", values="predicted_available_spaces",
                                   sort=False)
    columns = list(dict.fromkeys(slots.sort_values("timestamp")["slot_label"]))
    pivot_rate, pivot_free = pivot_rate[columns], pivot_free[columns]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(columns) + 3), 0.8 * len(pivot_rate) + 1.8))
    sns.heatmap(pivot_rate, ax=ax, cmap="YlOrRd", vmin=0, vmax=100, annot=pivot_free, fmt=".0f",
                annot_kws={"fontsize": 8}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": "% occupied (numbers = free spaces)"})
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title("Free spaces by time slot", loc="left")
    ax.grid(False)
    fig.tight_layout()
    return fig


def fig_weekday_weekend(hourly, location):
    """Average occupancy by hour: weekdays vs weekends for one location."""
    if hourly.empty:
        return _empty_figure("No data for this location")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    styles = {"Weekday": ("#264653", "-"), "Weekend": ("#e76f51", "--")}
    for day_type, group in hourly.groupby("day_type"):
        color, line = styles.get(day_type, ("#4c9be8", "-"))
        ax.plot(group["hour"], group["occupancy_pct"], color=color, linestyle=line, linewidth=2.4,
                marker="o", markersize=3.5, label=day_type)
    ax.set(ylim=(0, 100), xlabel="Hour of day", ylabel="Average occupancy (%)")
    ax.set_xticks(range(0, 24, 2))
    _threshold_lines(ax)
    ax.set_title(f"{location}: weekday vs weekend", loc="left")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def fig_weekday_weekend_bars(summary):
    """Average occupancy on weekdays vs weekends for every location."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["weekday_avg_pct"], width=0.38, color="#264653", label="Weekdays")
    ax.bar(x + 0.2, summary["weekend_avg_pct"], width=0.38, color="#e76f51", label="Weekends")
    for i, row in enumerate(summary.itertuples()):
        ax.text(i - 0.2, row.weekday_avg_pct + 1, f"{row.weekday_avg_pct:.0f}%", ha="center", fontsize=8, color=INK)
        ax.text(i + 0.2, row.weekend_avg_pct + 1, f"{row.weekend_avg_pct:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["location"], fontsize=9)
    ax.set(ylim=(0, 100), ylabel="Average occupancy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Weekday vs weekend demand", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def fig_daily_trend(daily, locations=None):
    """Daily average occupancy (faint) and 7-day rolling average (bold) per location."""
    data = daily if not locations else daily[daily["location"].isin(locations)]
    if data.empty:
        return _empty_figure("Select at least one location")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for location, group in data.groupby("location"):
        ax.plot(group["date"], group["occupancy_pct"], color=_color(location), alpha=0.25, linewidth=1)
        ax.plot(group["date"], group["rolling_avg"], color=_color(location), linewidth=2.4, label=location)
    ax.set(ylim=(0, 100), ylabel="Average occupancy (%)")
    ax.set_title("Daily occupancy trend (bold = 7-day average)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_weather_effect(effect):
    """How much busier/quieter than TYPICAL each weather type is, per location (percentage points)."""
    if effect.empty:
        return _empty_figure("Not enough data for the weather comparison")
    weathers = [w for w in config.WEATHER_OPTIONS if w in set(effect["weather"])]
    locations = list(effect["location"].unique())
    width = 0.8 / len(weathers)
    palette = {"Sunny": "#e9b44c", "Cloudy": "#9aa5b1", "Rainy": "#4c9be8", "Foggy": "#7b6cc4"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, weather in enumerate(weathers):
        sub = effect[effect["weather"] == weather].set_index("location").reindex(locations)
        offset = (i - (len(weathers) - 1) / 2) * width
        ax.bar(np.arange(len(locations)) + offset, sub["vs_typical_pts"].fillna(0), width=width * 0.92,
               yerr=1.96 * sub["std_err"].fillna(0), error_kw={"elinewidth": 1, "capsize": 2, "ecolor": MUTED},
               color=palette.get(weather, "#4c9be8"), label=weather)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels(locations, fontsize=9)
    ax.set(ylabel="Occupancy vs typical hour (pct points)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Weather effect (bars = difference from typical, whiskers = 95% range)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def fig_event_effect(effect):
    """Occupancy in the event window with and without a special event, per location."""
    if effect.empty:
        return _empty_figure("No event data")
    locations = list(effect["location"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (condition, color) in enumerate([("No event", "#9aa5b1"), ("Special event", "#e76f51")]):
        sub = effect[effect["condition"] == condition].set_index("location").reindex(locations)
        ax.bar(np.arange(len(locations)) + (i - 0.5) * 0.38, sub["avg_pct"].fillna(0), width=0.36, color=color, label=condition)
    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels(locations, fontsize=9)
    ax.set(ylim=(0, 100), ylabel="Average occupancy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Special events ({config.EVENT_START_HOUR}:00-{config.EVENT_END_HOUR}:00 only)", loc="left")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


def fig_model_metric_bars(results, best_name):
    """Test-set MAE (spaces), RMSE and R2 side by side; the selected model is highlighted."""
    panels = [("test_mae_spaces", "MAE in parking spaces (lower is better)", True),
              ("test_rmse", "RMSE, occupancy rate (lower is better)", True),
              ("test_r2", "R² (higher is better)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (column, title, ascending) in zip(axes, panels):
        data = results.sort_values(column, ascending=ascending)
        colors = ["#2a9d8f" if m == best_name else "#aab4c3" for m in data["model"]]
        ax.barh(data["model"], data[column], color=colors)
        ax.invert_yaxis()
        for y, value in enumerate(data[column]):
            ax.text(value, y, f" {value:.3f}" if column != "test_mae_spaces" else f" {value:.1f}", va="center", fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.margins(x=0.2)
        ax.grid(axis="y", visible=False)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    return fig


def fig_model_actual_vs_predicted(window, model_name, location):
    """Actual vs predicted free spaces for one model over a slice of the unseen test days."""
    if window.empty:
        return _empty_figure("No test data in this period")
    window = window.sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(window["timestamp"], window["actual_available_spaces"], color=INK, linewidth=2, label="Actual")
    ax.plot(window["timestamp"], window["predicted_available_spaces"], color="#e76f51", linewidth=1.6,
            linestyle="--", label=f"Predicted ({model_name})")
    ax.set(ylabel="Free spaces", ylim=(0, None))
    ax.set_title(f"{location}: {model_name} on unseen test days", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_feature_importance(table, title):
    """Horizontal bars of importance values (`feature`, `importance`, optional `std`)."""
    if table is None or table.empty:
        return _empty_figure("No importance data")
    data = table.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(data) + 1.4))
    err = data["std"].fillna(0) if "std" in data else None
    ax.barh(data["feature"], data["importance"], xerr=err, color="#2a9d8f",
            error_kw={"elinewidth": 1, "ecolor": MUTED, "capsize": 2})
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    return fig
