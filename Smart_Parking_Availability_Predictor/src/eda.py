"""Exploratory Data Analysis (EDA).

Creates a text summary and several charts that show the patterns hidden in
the (simulated) parking data. All charts are saved to outputs/charts/.
"""

import matplotlib
matplotlib.use("Agg")  # draw charts to files; no pop-up windows needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, filename):
    """Save a figure into outputs/charts/ and close it."""
    fig.text(0.99, 0.002, "Simulated data", ha="right", fontsize=8, color="gray")
    path = config.CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def load_processed(path=config.PROCESSED_DATA_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["occupancy_pct"] = df["occupancy_rate"] * 100     # easier to read than 0-1
    return df


def write_summary(df):
    """Write a plain-text summary of the dataset."""
    lines = [
        "EDA SUMMARY  (SIMULATED DATA - not real parking measurements)",
        "=" * 64,
        f"Rows: {len(df):,}    Columns: {df.shape[1]}",
        f"Time range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}",
        f"Parking lots: {df['parking_lot_id'].nunique()}",
        f"Missing values left: {int(df.isna().sum().sum())}",
        "",
        "Average occupancy per lot (%):",
        df.groupby("location")["occupancy_pct"].agg(["mean", "min", "max"]).round(1).to_string(),
        "",
        "Busiest hour (highest average occupancy) per lot:",
    ]
    hourly = df.groupby(["location", "hour"])["occupancy_pct"].mean()
    for location in df["location"].unique():
        peak_hour = hourly[location].idxmax()
        lines.append(f"  {location}: {peak_hour}:00  ({hourly[location].max():.1f}%)")

    lines += [
        "",
        "Weekday vs weekend average occupancy (%):",
        df.groupby(["location", "is_weekend"])["occupancy_pct"].mean().round(1)
          .unstack().rename(columns={0: "weekday", 1: "weekend"}).to_string(),
        "",
        "Numeric column statistics:",
        df[["total_spaces", "occupied_spaces", "available_spaces", "occupancy_pct"]]
          .describe().round(2).to_string(),
    ]

    path = config.RESULTS_DIR / "eda_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eda] Summary saved: {path}")


def chart_distribution(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="occupancy_pct", hue="location", bins=30,
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set(title="How full are the lots? (distribution of occupancy)",
           xlabel="Occupancy (%)", ylabel="Density")
    return _save(fig, "eda_01_occupancy_distribution.png")


def chart_hourly_pattern(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (flag, label) in zip(axes, [(0, "Weekdays"), (1, "Weekends")]):
        sns.lineplot(data=df[df["is_weekend"] == flag], x="hour", y="occupancy_pct",
                     hue="location", errorbar=None, marker="o", ax=ax)
        ax.set(title=f"Average occupancy by hour - {label}",
               xlabel="Hour of day", ylabel="Occupancy (%)")
        ax.set_xticks(range(0, 24, 2))
    return _save(fig, "eda_02_hourly_pattern.png")


def chart_heatmaps(df):
    lots = list(config.PARKING_LOTS.items())
    rows = (len(lots) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
    axes = axes.ravel()
    for ax, (lot_id, info) in zip(axes, lots):
        pivot = (df[df["parking_lot_id"] == lot_id]
                 .pivot_table(index="day_of_week", columns="hour",
                              values="occupancy_pct", aggfunc="mean"))
        pivot.index = [DAY_NAMES[i] for i in pivot.index]
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=100,
                    cbar_kws={"label": "% occupied"})
        ax.set(title=info["location"], xlabel="Hour of day", ylabel="")
    for ax in axes[len(lots):]:
        ax.axis("off")
    fig.suptitle("Average occupancy by weekday and hour", fontsize=14)
    return _save(fig, "eda_03_heatmap_hour_by_day.png")


def chart_weather(df):
    order = [w for w in config.WEATHER_OPTIONS + ["Unknown"] if w in df["weather"].unique()]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="weather", y="occupancy_pct", hue="location",
                order=order, errorbar=None, ax=ax)
    ax.set(title="Average occupancy by weather", xlabel="Weather", ylabel="Occupancy (%)")
    return _save(fig, "eda_04_weather_effect.png")


def chart_special_event(df):
    # Compare only the event hours so we compare like with like.
    evening = df[df["hour"].between(config.EVENT_START_HOUR, config.EVENT_END_HOUR)].copy()
    evening["event"] = evening["special_event"].map({0: "No event", 1: "Special event"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=evening, x="location", y="occupancy_pct", hue="event",
                errorbar=None, ax=ax)
    ax.set(title=f"Effect of special events ({config.EVENT_START_HOUR}:00-"
                 f"{config.EVENT_END_HOUR}:00 only)",
           xlabel="", ylabel="Occupancy (%)")
    return _save(fig, "eda_05_special_event_effect.png")


def chart_correlation(df):
    columns = ["occupancy_rate", "hour", "day_of_week", "is_weekend",
               "is_rush_hour", "special_event", "total_spaces"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title("Correlation between numeric columns")
    return _save(fig, "eda_06_correlation_heatmap.png")


def chart_daily_trend(df):
    daily = (df.assign(date=df["timestamp"].dt.floor("D"))
               .groupby(["date", "location"])["occupancy_pct"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=daily, x="date", y="occupancy_pct", hue="location", ax=ax)
    ax.set(title="Daily average occupancy over time", xlabel="Date", ylabel="Occupancy (%)")
    fig.autofmt_xdate()
    return _save(fig, "eda_07_daily_trend.png")


def run_eda(path=config.PROCESSED_DATA_PATH):
    """Run the full EDA: text summary + all charts."""
    config.ensure_directories()
    df = load_processed(path)
    write_summary(df)
    for chart_function in (chart_distribution, chart_hourly_pattern, chart_heatmaps,
                           chart_weather, chart_special_event, chart_correlation,
                           chart_daily_trend):
        saved = chart_function(df)
        print(f"[eda] Chart saved: {saved.name}")


if __name__ == "__main__":
    run_eda()
