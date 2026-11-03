"""Page - Smart Recommendations: where should I park, and when should I arrive?"""

import html
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app import ui
from src import config
from src import dashboard_charts as charts
from src import dashboard_data as dd
from src import recommend as rec
from src.predict import lot_supports_events

NO_DESTINATION = "No specific destination"
WEATHER_CHOICES = ["Not sure"] + config.WEATHER_OPTIONS


def _validation_messages(lots, when, weather, event_today, today):
    """Run the shared input checks for every chosen lot; return (errors, warnings) without repeats."""
    errors, warnings = [], []
    for lot in lots:
        flag = bool(event_today) and lot_supports_events(lot)
        e, w = dd.validate_prediction_inputs(lot, when, weather or "Sunny", flag, today)
        errors += [m for m in e if m not in errors]
        warnings += [m for m in w if m not in warnings]
    return errors, warnings


def render(ctx):
    ui.page_header("Smart Parking Recommendations",
                   "Which location is likely to have the most free spaces when you arrive?")

    try:
        model = ctx.get_model()
    except Exception as error:
        st.error(f"The trained model could not be loaded: {error}")
        st.info("Train it on this computer first:")
        st.code("python main.py", language="bash")
        return

    distances, distance_note = rec.load_distances()
    all_names = {info["location"]: lot for lot, info in config.PARKING_LOTS.items()}
    today = date.today()

    inputs_col, results_col = st.columns([1, 2], gap="large")

    # ---- inputs -----------------------------------------------------------------
    with inputs_col:
        st.subheader("1. Your trip")
        destination = st.selectbox("Destination", [NO_DESTINATION] + list(all_names), key="rec_destination",
                                   help="The place you are going to. The dataset has no map data, so this is "
                                        "used to highlight that location, not to measure distance.")
        destination = None if destination == NO_DESTINATION else destination

        chosen_names = st.multiselect("Parking locations you would accept", list(all_names),
                                      default=list(all_names), key="rec_locations")
        arrival_date = st.date_input("Arrival date", value=today, min_value=today,
                                     max_value=today + timedelta(days=dd.MAX_FORECAST_DAYS), key="rec_date")
        next_hour = (pd.Timestamp.now().floor(pd.Timedelta(hours=1)) + pd.Timedelta(hours=1)).hour
        hour = st.selectbox("Arrival time", list(range(24)), index=next_hour,
                            format_func=lambda h: f"{h:02d}:00", key="rec_hour")
        weather_label = st.selectbox("Weather", WEATHER_CHOICES, key="rec_weather",
                                     help="Pick 'Not sure' if you do not know. The model then averages over weather.")
        weather = None if weather_label == "Not sure" else weather_label

        chosen_lots = [all_names[n] for n in chosen_names]
        if any(lot_supports_events(lot) for lot in chosen_lots):
            event_today = st.checkbox(f"A special event is happening nearby today "
                                      f"(affects {config.EVENT_START_HOUR}:00-{config.EVENT_END_HOUR}:00)",
                                      key="rec_event")
        else:
            event_today = False

        rank_label = st.radio("Rank locations by", list(rec.RANK_OPTIONS.values()), key="rec_rank",
                              help="'Most free spaces' favours big lots; 'share of free spaces' is fairer to small ones.")
        rank_by = [k for k, v in rec.RANK_OPTIONS.items() if v == rank_label][0]

        weight = 0.3
        if distances is not None:
            weight = st.slider("How much should distance count?", 0.0, 1.0, 0.3, 0.1, key="rec_distance_weight",
                               help="0 = availability only, 1 = nearest location only")
        st.caption(distance_note)

    when = pd.Timestamp(arrival_date) + pd.Timedelta(hours=hour)

    # ---- results ----------------------------------------------------------------
    with results_col:
        st.subheader("2. Recommendation")
        if not chosen_lots:
            st.info("Select at least one parking location to compare.")
            return
        errors, warnings = _validation_messages(chosen_lots, when, weather, event_today, today)
        for message in errors:
            st.error(message)
        if errors:
            return
        for message in warnings:
            st.warning(message)
        if when < pd.Timestamp.now().floor(pd.Timedelta(hours=1)):
            st.warning("The chosen time is in the past. The prediction is still made, but it is a what-if.")

        try:
            result = rec.recommend(chosen_lots, when, weather, event_today, model=model,
                                   destination=destination, distances=distances,
                                   distance_weight=weight, rank_by=rank_by)
        except ValueError as error:
            st.error(str(error))
            return
        except Exception as error:
            st.error(f"The recommendation could not be made: {error}")
            return

        (st.success if result["best"] else st.warning)(result["headline"])
        for note in result["notes"]:
            st.caption(note)

        table = result["table"]
        cards = []
        for row in table.itertuples():
            star = " ⭐" if row.recommended else ""
            rng = (f" &bull; typically {row.likely_low}-{row.likely_high}"
                   if hasattr(row, "likely_low") else "")
            dist = f" &bull; {row.distance_km:.1f} km" if result["uses_distance"] else ""
            cards.append(
                f'<div class="loc-row"><div><div class="loc-name">#{row.rank} {html.escape(row.location)}{star}</div>'
                f'<div class="muted">{row.predicted_available_spaces} free of {row.total_spaces} &bull; '
                f'{row.predicted_occupancy_rate * 100:.0f}% full{rng}{dist}</div>'
                f'<div class="muted">{html.escape(row.reason)}</div></div>'
                f'<div>{ui.badge(row.status)}</div></div>')
        st.markdown("".join(cards), unsafe_allow_html=True)
        ui.estimate_notice()

    # ---- details -----------------------------------------------------------------
    st.subheader("3. Compare the options")
    left, right = st.columns([2, 3], gap="large")
    with left:
        ui.show_fig(charts.fig_prediction_all_lots(table))
    with right:
        show = pd.DataFrame({
            "Rank": table["rank"],
            "Location": table["location"],
            "Status": [f"{ui.STATUS_ICONS[s]} {s}" for s in table["status"]],
            "Predicted free": table["predicted_available_spaces"],
            "Capacity": table["total_spaces"],
            "Predicted occupancy %": (table["predicted_occupancy_rate"] * 100).round(0).astype(int),
        })
        if "likely_low" in table:
            show["Typical range (free)"] = [f"{a}-{b}" for a, b in zip(table["likely_low"], table["likely_high"])]
        if result["uses_distance"]:
            show["Distance (km)"] = table["distance_km"].round(1)
        st.dataframe(show, hide_index=True)
        st.caption("'Typical range' is how far off the model usually is for that location (measured on past "
                   "forecasts of the training period). Real availability can fall outside it.")

    # ---- best time to arrive -----------------------------------------------------
    st.subheader("4. When is the best time to arrive?")
    focus_name = destination or (result["best"]["location"] if result["best"] else table.iloc[0]["location"])
    focus_lot = all_names[focus_name]
    c1, c2 = st.columns([1, 2])
    span = c1.selectbox("Look ahead", [3, 6, 12, 24], index=2, format_func=lambda h: f"{h} hours",
                        key="rec_span")
    c2.markdown(f"Looking at **{html.escape(focus_name)}** from {when:%a %d %b, %H:00}"
                f"{' (your destination)' if destination else ' (the recommended location)'}.")
    try:
        timing = rec.best_time_to_arrive(focus_lot, when, span, weather,
                                         bool(event_today) and lot_supports_events(focus_lot), model=model)
    except ValueError as error:
        st.error(str(error))
        return
    best, worst = timing["best"], timing["worst"]
    ui.kpi_row([
        {"label": "Most free spaces", "value": f"{best['predicted_available_spaces']}",
         "sub": f"around {best['slot_label']}", "accent": "#2e9e5b"},
        {"label": "Fewest free spaces", "value": f"{worst['predicted_available_spaces']}",
         "sub": f"around {worst['slot_label']}", "accent": "#d64545"},
        {"label": "First time not Full",
         "value": timing["first_available"]["slot_label"] if timing["first_available"] else "none",
         "sub": "in this window"},
    ])
    ui.show_fig(charts.fig_slot_forecast(timing["slots"]))
    st.caption("Shown for the same weather and event assumptions as above. Hours are forecasts, not guarantees.")

    with st.expander("How are locations ranked?"):
        st.markdown(
            "1. The saved model predicts free spaces for every selected location at your arrival time "
            "(same features as in training).\n"
            "2. Locations predicted **Full** are put last and never recommended.\n"
            f"3. The rest are ranked by **{rank_label.lower()}**"
            + (", combined with distance using your slider.\n" if result["uses_distance"]
               else ". **Distance is not used**, because no distance data was supplied.\n")
            + "4. A destination only highlights that location and adds advice about it.")
