"""Page - Historical Analysis: peak hours, weekday vs weekend, trends, weather and events."""

import pandas as pd
import streamlit as st

from app import ui
from src import analysis as an
from src import dashboard_charts as charts
from src import dashboard_data as dd


def _caveat():
    if ui.is_simulated():
        st.caption("These patterns describe the **simulated** data, so they show what the simulation "
                   "contains (for example the weather effects were written into it on purpose).")
    else:
        st.caption("These patterns describe the supplied data. Correlation is not proof of cause: "
                   "weather, time of day and events often change together.")


def _tab_peaks(ctx):
    st.markdown("**When is each location busiest?** The *peak window* is every hour within 90% of the busiest hour.")
    peaks = an.peak_hours(ctx.df)
    for row in peaks.itertuples():
        st.markdown(f"- **{row.location}**: busiest at **{row.peak_hour:02d}:00** "
                    f"({row.peak_occupancy_pct:.0f}% full on average). Peak window {row.peak_window}; "
                    f"quietest at {row.quietest_hour:02d}:00 ({row.quietest_occupancy_pct:.0f}%).")
    table = pd.DataFrame({
        "Location": peaks["location"],
        "Peak hour": peaks["peak_hour"].map(lambda h: f"{h:02d}:00"),
        "Peak occupancy %": peaks["peak_occupancy_pct"].round(1),
        "Weekday peak window": peaks["weekday_peak_window"],
        "Weekend peak window": peaks["weekend_peak_window"],
    })
    st.dataframe(table, hide_index=True)
    names = sorted(ctx.df["location"].unique())
    chosen = st.multiselect("Locations in the chart", names, default=names, key="hist_peak_locs")
    if chosen:
        ui.show_fig(charts.fig_hourly_demand(dd.hourly_profile(ctx.df, chosen)))
    else:
        st.info("Select at least one location to see the chart.")


def _tab_weekday_weekend(ctx):
    summary = an.weekday_weekend_summary(ctx.df)
    for row in summary.itertuples():
        word = "busier" if row.difference_pts > 0 else "quieter"
        st.markdown(f"- **{row.location}** is **{abs(row.difference_pts):.0f} points {word}** at weekends "
                    f"({row.weekday_avg_pct:.0f}% on weekdays vs {row.weekend_avg_pct:.0f}% at weekends).")
    ui.show_fig(charts.fig_weekday_weekend_bars(summary))
    location = st.selectbox("Hour-by-hour comparison for", sorted(ctx.df["location"].unique()), key="hist_wd_loc")
    ui.show_fig(charts.fig_weekday_weekend(an.weekday_weekend_hourly(ctx.df, location), location))
    st.dataframe(pd.DataFrame({
        "Location": summary["location"],
        "Weekday avg %": summary["weekday_avg_pct"].round(1),
        "Weekend avg %": summary["weekend_avg_pct"].round(1),
        "Weekday peak %": summary["weekday_peak_pct"].round(1),
        "Weekend peak %": summary["weekend_peak_pct"].round(1),
        "Busier on": summary["busier_on"]}), hide_index=True)


def _tab_trends(ctx):
    names = sorted(ctx.df["location"].unique())
    chosen = st.multiselect("Locations", names, default=names, key="hist_trend_locs")
    daily = an.daily_trend(ctx.df)
    if chosen:
        ui.show_fig(charts.fig_daily_trend(daily, chosen))
    else:
        st.info("Select at least one location to see the chart.")
    trend = an.trend_summary(daily)
    st.markdown("**Is any location getting busier or quieter?** A trend is only reported when the straight-line "
                "slope is statistically clear and not tiny.")
    show = pd.DataFrame({
        "Location": trend["location"],
        "Change per week (pts)": trend["slope_pts_per_week"].round(2),
        "p-value": trend["p_value"].round(3),
        "Verdict": trend["verdict"]})
    st.dataframe(show, hide_index=True)
    days = ctx.df["timestamp"].dt.normalize().nunique()
    st.caption(f"Only {days} days of history are available, so seasonal changes cannot be seen. "
               "The weekly up-and-down is smoothed out by the 7-day average.")


def _tab_weather(ctx):
    effect = an.weather_effect(ctx.df)
    ui.show_fig(charts.fig_weather_effect(effect))
    st.markdown("Each bar compares occupancy in that weather with the **typical value for the same location, "
                "hour and weekday/weekend**, so a rainy afternoon is not mistaken for 'rain makes it busy'. "
                "Hours with special events are left out.")
    if not effect.empty:
        strongest = effect.reindex(effect["vs_typical_pts"].abs().sort_values(ascending=False).index).head(3)
        for row in strongest.itertuples():
            direction = "busier" if row.vs_typical_pts > 0 else "quieter"
            st.markdown(f"- {row.location} in **{row.weather.lower()}** weather: about "
                        f"{abs(row.vs_typical_pts):.1f} points {direction} than usual.")
        table = effect.rename(columns={"location": "Location", "weather": "Weather", "raw_avg_pct": "Average %",
                                       "vs_typical_pts": "vs typical (pts)", "hours": "Hours of data"})
        st.dataframe(table[["Location", "Weather", "Average %", "vs typical (pts)", "Hours of data"]].round(2),
                     hide_index=True)
    st.markdown("**Special events**")
    ui.show_fig(charts.fig_event_effect(an.event_effect(ctx.df)))
    if "Unknown" in set(ctx.df["weather"]):
        st.caption("Hours with missing weather ('Unknown') are not shown in the weather comparison.")


def render(ctx):
    ui.page_header("Historical Analysis", "Peak hours, weekday vs weekend, trends and weather")
    _caveat()
    tabs = st.tabs(["Peak hours", "Weekday vs weekend", "Trends by location", "Weather & events"])
    with tabs[0]:
        _tab_peaks(ctx)
    with tabs[1]:
        _tab_weekday_weekend(ctx)
    with tabs[2]:
        _tab_trends(ctx)
    with tabs[3]:
        _tab_weather(ctx)
