"""Page 4 - Charts & Analytics: demand patterns, comparisons, actual vs predicted."""

import pandas as pd
import streamlit as st

from app import ui
from src import dashboard_charts as charts
from src import dashboard_data as dd

HOURLY_METRICS = {
    "Occupancy (%)": "occupancy_pct",
    "Occupied spaces": "occupied_spaces",
    "Available spaces": "available_spaces",
}
COMPARISON_METRICS = {
    "Average available spaces": "avg_available_spaces",
    "Average occupancy (%)": "avg_occupancy_pct",
    "Peak hourly occupancy (%)": "peak_occupancy_pct",
    "Total spaces": "total_spaces",
}


def _location_filter(ctx, key):
    names = sorted(ctx.df["location"].unique())
    chosen = st.multiselect("Locations", names, default=names, key=key)
    if not chosen:
        st.info("Select at least one location to see the chart.")
    return chosen


def _tab_hourly(ctx):
    st.markdown("When is parking in demand? Each line is the **average** over all days in the dataset.")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        chosen = _location_filter(ctx, "an_hourly_locs")
    day_type = c2.selectbox("Days", ["All days", "Weekdays", "Weekends"], key="an_hourly_days")
    metric_label = c3.selectbox("Measure", list(HOURLY_METRICS), key="an_hourly_metric")
    if chosen:
        profile = dd.hourly_profile(ctx.df, chosen, day_type)
        ui.show_fig(charts.fig_hourly_demand(profile, HOURLY_METRICS[metric_label]))
        peaks = profile.loc[profile.groupby("location")["occupancy_pct"].idxmax()]
        st.caption("Busiest hour: " + " • ".join(
            f"{r.location} at {int(r.hour):02d}:00 ({r.occupancy_pct:.0f}%)" for r in peaks.itertuples()))


def _tab_weekday(ctx):
    st.markdown("Which days are busiest? Average occupancy for each day of the week.")
    chosen = _location_filter(ctx, "an_dow_locs")
    if chosen:
        ui.show_fig(charts.fig_day_of_week(dd.day_of_week_profile(ctx.df, chosen)))
        st.markdown("**Heatmap: weekday and hour combined**")
        location = st.selectbox("Location for the heatmap", chosen, key="an_heat_loc")
        ui.show_fig(charts.fig_weekday_hour_heatmap(ctx.df, location))


def _tab_comparison(ctx):
    st.markdown("How do the locations compare overall?")
    metric_label = st.selectbox("Compare by", list(COMPARISON_METRICS), key="an_cmp_metric")
    summary = dd.location_summary(ctx.df)
    col_chart, col_table = st.columns([3, 2], gap="large")
    with col_chart:
        ui.show_fig(charts.fig_location_comparison(summary, COMPARISON_METRICS[metric_label]))
    with col_table:
        table = pd.DataFrame({
            "Location": summary["location"],
            "Spaces": summary["total_spaces"],
            "Avg occupancy %": summary["avg_occupancy_pct"].round(1),
            "Avg free": summary["avg_available_spaces"].round(0).astype(int),
            "Peak hour": summary["peak_hour"].map(lambda h: f"{h:02d}:00"),
        })
        st.dataframe(table, hide_index=True)


def _tab_actual_vs_predicted(ctx):
    tp = ctx.test_predictions
    if tp is None or tp.empty:
        st.info("No test predictions found. Run `python main.py` to create "
                "`outputs/results/test_predictions.csv`.")
        return

    first, last = tp["timestamp"].min(), tp["timestamp"].max()
    st.success(
        f"**Only valid test data is shown.** The model never saw {first:%d %b %Y} to {last:%d %b %Y} "
        "during training, so comparing actual with predicted values here is a fair test. "
        "Dates before that were used for training and are deliberately not shown.")

    c1, c2, c3 = st.columns([2, 1, 1])
    location = c1.selectbox("Location", sorted(tp["location"].unique()), key="an_avp_loc")
    first_day, last_day = first.date(), last.date()
    start = c2.date_input("Start date", value=first_day, min_value=first_day, max_value=last_day,
                          key="an_avp_start")
    span = c3.selectbox("Period", [1, 3, 7, 14, 18], index=2, format_func=lambda d: f"{d} day(s)",
                        key="an_avp_days")

    window = dd.prediction_window(tp, location, start, span)
    if window.empty:
        st.info("No test data for this selection. Choose a start date inside the test period.")
        return

    metrics = dd.prediction_metrics(window)
    if metrics:
        ui.kpi_row([
            {"label": "Hours compared", "value": f"{metrics['rows']}", "sub": "one row per hour"},
            {"label": "MAE", "value": f"{metrics['mae']:.1f} spaces", "sub": "average miss"},
            {"label": "RMSE", "value": f"{metrics['rmse']:.1f} spaces", "sub": "punishes big misses"},
            {"label": "R²", "value": f"{metrics['r2']:.3f}", "sub": "1.0 = perfect"},
        ])
    col_line, col_scatter = st.columns([3, 2], gap="large")
    with col_line:
        ui.show_fig(charts.fig_actual_vs_predicted_series(window, location))
    with col_scatter:
        ui.show_fig(charts.fig_actual_vs_predicted_scatter(window, location))
    st.caption("Metrics here are for the selected location and period, in parking spaces. "
               "A short period can look better or worse than the full test set.")


def _tab_models(ctx):
    table = ctx.model_comparison
    if table is None:
        st.info("No model comparison found. Run `python main.py` first.")
        return
    best = ctx.metadata.get("best_model")
    st.markdown(f"All models were scored on the same unseen test days. **Best model: {best}**.")
    show = table[["model", "test_mae", "test_rmse", "test_r2", "test_mae_spaces", "cv_rmse"]].copy()
    show.columns = ["Model", "MAE", "RMSE", "R²", "MAE (spaces)", "CV RMSE (train)"]
    st.dataframe(show.round(4), hide_index=True)
    st.caption("MAE/RMSE are in occupancy-rate units (0.05 = 5 percentage points). The baseline "
               "always predicts the average, so every real model must beat it.")
    chart = charts.config.CHARTS_DIR / "model_01_comparison.png"
    if chart.exists():
        st.image(str(chart))


def render(ctx):
    ui.page_header("Charts & Analytics", "Patterns in the data and how well the model predicts them")
    tabs = st.tabs(["Hourly demand", "Day of week", "Location comparison",
                    "Actual vs predicted", "Model scores"])
    with tabs[0]:
        _tab_hourly(ctx)
    with tabs[1]:
        _tab_weekday(ctx)
    with tabs[2]:
        _tab_comparison(ctx)
    with tabs[3]:
        _tab_actual_vs_predicted(ctx)
    with tabs[4]:
        _tab_models(ctx)
