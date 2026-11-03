"""Page 3 - Predict Availability: uses the saved ML model (no hardcoded results)."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app import ui
from src import config
from src import dashboard_charts as charts
from src import dashboard_data as dd
from src.predict import predict_full_day, predict_many, predict_slots, prepare_features


def _lot_label(lot_id):
    return f"{config.PARKING_LOTS[lot_id]['location']} ({lot_id})"


def _collect_inputs(today):
    """Draw the input widgets and return (lot_id, chosen_date, hour, weather, event_today)."""
    st.subheader("1. Choose the situation")
    lot_id = st.selectbox("Parking location", list(config.PARKING_LOTS),
                          format_func=_lot_label, key="pred_lot")

    first_day = pd.Timestamp(config.START_DATE).date()
    last_day = today + timedelta(days=dd.MAX_FORECAST_DAYS)
    mode = st.radio("Pick the day by", ["Calendar date", "Day of week"], horizontal=True,
                    key="pred_mode")
    if mode == "Calendar date":
        default_day = min(max(today, first_day), last_day)
        chosen_date = st.date_input("Date", value=default_day, min_value=first_day,
                                    max_value=last_day, key="pred_date")
    else:
        weekday_name = st.selectbox("Day of week", dd.DAY_NAMES_FULL, index=today.weekday(),
                                    key="pred_weekday")
        chosen_date = dd.next_date_for_weekday(dd.DAY_NAMES_FULL.index(weekday_name), today).date()
        st.caption(f"Uses the next {weekday_name}: **{chosen_date:%d %b %Y}**")

    hour = st.selectbox("Time of day", list(range(24)), index=9,
                        format_func=lambda h: f"{h:02d}:00", key="pred_hour")
    st.caption(f"Day of week: **{pd.Timestamp(chosen_date):%A}**")

    weather = st.selectbox("Weather", config.WEATHER_OPTIONS, key="pred_weather")

    if dd.lot_supports_events(lot_id):
        event_today = st.checkbox(
            f"Special event near this location today "
            f"(affects {config.EVENT_START_HOUR}:00-{config.EVENT_END_HOUR}:00)", key="pred_event")
    else:
        event_today = False
        st.caption("No special events are modelled for this location.")

    st.caption("The prediction updates automatically when you change an input.")
    return lot_id, chosen_date, hour, weather, event_today


def _slot_section(lot_id, when, weather, event_today, model):
    """Hourly forecast for several upcoming time slots (one or more locations)."""
    st.subheader("4. Upcoming time slots")
    st.caption("An hourly forecast starting at the time you chose. The same weather and event "
               "assumptions are used for every hour.")
    c1, c2 = st.columns([1, 2])
    hours = c1.selectbox("Number of hourly slots", [3, 6, 12, 24, 48], index=2, key="pred_slot_hours")
    names = {lot: config.PARKING_LOTS[lot]["location"] for lot in config.PARKING_LOTS}
    chosen = c2.multiselect("Locations", list(config.PARKING_LOTS), default=list(config.PARKING_LOTS),
                            format_func=lambda lot: names[lot], key="pred_slot_lots")
    if not chosen:
        st.info("Select at least one location to see the forecast.")
        return
    try:
        slots = predict_slots(chosen, when, hours, weather, event_today, model=model)
    except ValueError as error:
        st.error(str(error))
        return
    left, right = st.columns(2, gap="large")
    with left:
        ui.show_fig(charts.fig_slot_forecast(slots))
    with right:
        ui.show_fig(charts.fig_slot_heat(slots))
    pivot = slots.pivot_table(index="slot_label", columns="location", values="predicted_available_spaces",
                              sort=False).astype(int)
    st.dataframe(pivot, hide_index=False)
    ui.estimate_notice()
    st.download_button("Download the slot forecast (CSV)", data=slots.to_csv(index=False).encode("utf-8"),
                       file_name=f"slot_forecast_{when:%Y-%m-%d_%H}h.csv", mime="text/csv", key="pred_slot_download")


def render(ctx):
    ui.page_header("Predict Parking Availability",
                   "The saved machine-learning model estimates free spaces for a chosen time")

    try:
        model = ctx.get_model()
    except Exception as error:   # missing file, or a scikit-learn version mismatch
        st.error(f"The trained model could not be loaded: {error}")
        st.info("Train it on this computer (this also fixes library-version mismatches):")
        st.code("python main.py", language="bash")
        return

    today = date.today()
    inputs_col, results_col = st.columns([1, 2], gap="large")

    with inputs_col:
        lot_id, chosen_date, hour, weather, event_today = _collect_inputs(today)

    when = pd.Timestamp(chosen_date) + pd.Timedelta(hours=hour)
    errors, warnings = dd.validate_prediction_inputs(lot_id, when, weather, event_today, today)

    with results_col:
        st.subheader("2. Prediction")
        for message in errors:
            st.error(message)
        if errors:
            st.info("Fix the input above to see a prediction.")
            return
        for message in warnings:
            st.warning(message)

        request = {"parking_lot_id": lot_id, "timestamp": when, "weather": weather,
                   "event_today": event_today}
        try:
            result = predict_many([request], model=model).iloc[0]
            day = predict_full_day(lot_id, chosen_date, weather, event_today, model=model)
            all_lots = predict_many(
                [{"parking_lot_id": other, "timestamp": when, "weather": weather,
                  "event_today": bool(event_today and dd.lot_supports_events(other))}
                 for other in config.PARKING_LOTS], model=model)
        except Exception as error:
            st.error(f"The prediction could not be made: {error}")
            return

        status = result["status"]
        st.markdown(f"**{result['location']}** • {when:%A, %d %b %Y, %H:%M} • "
                    f"{ui.badge(status)}", unsafe_allow_html=True)
        ui.kpi_row([
            {"label": "Predicted available", "value": f"{result['predicted_available_spaces']}",
             "sub": f"of {result['total_spaces']} spaces", "accent": ui.STATUS_ACCENT[status]},
            {"label": "Predicted occupied", "value": f"{result['predicted_occupied_spaces']}",
             "sub": "spaces in use"},
            {"label": "Predicted occupancy", "value": f"{result['predicted_occupancy_rate'] * 100:.0f}%",
             "sub": ui.STATUS_DESCRIPTIONS[status]},
        ])
        st.progress(float(min(max(result["predicted_occupancy_rate"], 0.0), 1.0)))
        if "likely_low" in result.index:
            st.caption(f"Typical range for this location: about **{result['likely_low']}-{result['likely_high']}** "
                       "free spaces (how far off forecasts usually are).")
        ui.estimate_notice()

        notes = []
        error_spaces = ctx.per_lot_error.get(result["location"])
        if error_spaces is not None:
            notes.append(f"On unseen test days this model was off by about **{error_spaces:.0f} spaces** "
                         "on average for this location, so treat the number as an estimate.")
        history = dd.historical_reference(ctx.df, lot_id, when.hour, when.dayofweek >= 5)
        if history:
            notes.append(f"For comparison, in the historical data this location averaged "
                         f"**{history['avg_occupancy_pct']:.0f}% occupancy** at {when.hour:02d}:00 on "
                         f"{'weekends' if when.dayofweek >= 5 else 'weekdays'} "
                         f"({history['avg_available_spaces']:.0f} spaces free).")
        for note in notes:
            st.caption(note)

        # Compare with the dataset only when that is a FAIR test (hold-out period).
        recorded = dd.recorded_reading(ctx.df, ctx.test_predictions, lot_id, when)
        if recorded and recorded["kind"] == "test":
            st.success(
                f"**Reality check:** this hour is in the unseen test period. The dataset recorded "
                f"**{recorded['actual_available']} free spaces** ({recorded['weather']}, "
                f"{'event' if recorded['special_event'] else 'no event'}); the model, given those recorded "
                f"conditions, predicted **{recorded['model_available']}**. Your chosen weather and event "
                "may differ from what was recorded.")
        elif recorded:
            st.info("This hour was part of the **training data**, so comparing it with the prediction "
                    "would be unfair to judge the model. See **Charts & Analytics > Actual vs predicted** "
                    "for the proper test-period comparison.")

    # ---- charts below, full width -------------------------------------------
    st.subheader("3. Forecast around this time")
    left, right = st.columns([3, 2], gap="large")
    with left:
        title = f"{result['location']} on {when:%a %d %b %Y}"
        ui.show_fig(charts.fig_forecast_day(day, when.hour, int(result["total_spaces"]), title))
    with right:
        ui.show_fig(charts.fig_prediction_all_lots(all_lots))

    _slot_section(lot_id, when, weather, event_today, model)

    st.download_button("Download the 24-hour forecast (CSV)",
                       data=day.to_csv(index=False).encode("utf-8"),
                       file_name=f"forecast_{lot_id}_{chosen_date:%Y-%m-%d}.csv",
                       mime="text/csv", key="pred_download")

    with st.expander("How was this prediction made?"):
        st.markdown(
            "The dashboard sends your choices through the **same feature engineering and "
            "saved scikit-learn pipeline that were used in training** (`src/features.py` "
            "and `models/best_model.joblib`). Nothing is typed in by hand.\n\n"
            "These are the exact input values the model received:")
        ok, message = dd.model_feature_check(model)
        (st.success if ok else st.error)(message)
        _, features = prepare_features([request])
        feature_table = pd.DataFrame({"Feature": list(features.columns),
                                      "Value": [str(v) for v in features.iloc[0]]})
        st.dataframe(feature_table, hide_index=True)
        best = ctx.metadata.get("best_model", "the saved model")
        st.caption(f"Model: {best}. Trained on the project dataset only; the model knows weekday, hour, "
                   "weather and special events, but nothing about real-time traffic or holidays.")
