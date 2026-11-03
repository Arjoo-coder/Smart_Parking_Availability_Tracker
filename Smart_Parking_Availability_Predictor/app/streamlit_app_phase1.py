"""Streamlit web app - Smart Parking Availability Predictor.

Run from the project root folder:
    streamlit run app/streamlit_app.py

The app only USES the model trained by `python main.py`; it does not train anything.
"""

import json
import sys
from datetime import date
from pathlib import Path

# Make `import src...` work when Streamlit starts this file from another folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.predict import load_model, predict_full_day, predict_occupancy

st.set_page_config(page_title="Smart Parking Availability Predictor",
                   page_icon="\U0001F17F️", layout="wide")


@st.cache_resource
def get_model():
    """Load the trained model once and reuse it (fast reruns)."""
    return load_model()


def lot_label(lot_id):
    return f"{lot_id} - {config.PARKING_LOTS[lot_id]['location']}"


# ---------------------------------------------------------------------------
# Header + simulated-data warning (always visible)
# ---------------------------------------------------------------------------
st.title("\U0001F17F️ Smart Parking Availability Predictor")
st.warning("**SIMULATED DATA.** The model was trained on artificial, rule-generated "
           "parking data - not real measurements. Predictions are for learning and "
           "demonstration only.")

try:
    model = get_model()
except Exception as error:  # missing file, or a scikit-learn version mismatch
    st.error(f"Could not load the trained model: {error}")
    st.info("Train it on this computer first (this also fixes version mismatches):")
    st.code("python main.py", language="bash")
    st.stop()

tab_predict, tab_results, tab_insights = st.tabs(
    ["\U0001F52E Predict", "\U0001F4CA Model results", "\U0001F5BC️ Data insights"])

# ---------------------------------------------------------------------------
# Tab 1: prediction
# ---------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Choose the situation")
        lot_id = st.selectbox("Parking lot", list(config.PARKING_LOTS), format_func=lot_label)
        chosen_date = st.date_input("Date", value=date.today())
        hour = st.slider("Hour of day (0-23)", min_value=0, max_value=23, value=9)
        weather = st.selectbox("Weather", config.WEATHER_OPTIONS)
        event_today = st.checkbox(
            f"Special event nearby today (active {config.EVENT_START_HOUR}:00-"
            f"{config.EVENT_END_HOUR}:00)")

    timestamp = pd.Timestamp(chosen_date) + pd.Timedelta(hours=hour)
    result = predict_occupancy(lot_id, timestamp, weather, event_today, model=model)

    with right:
        st.subheader(f"{result['location']} at {hour:02d}:00")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted free spaces",
                  f"{result['predicted_available_spaces']} / {result['total_spaces']}")
        c2.metric("Predicted occupancy", f"{result['predicted_occupancy_rate'] * 100:.0f}%")
        c3.metric("Status", result["status"])
        st.progress(float(result["predicted_occupancy_rate"]))

        # Whole-day forecast for the same settings
        day = predict_full_day(lot_id, chosen_date, weather, event_today, model=model)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(day["timestamp"].dt.hour, day["predicted_available_spaces"],
                color="#2a9d8f", linewidth=2)
        ax.fill_between(day["timestamp"].dt.hour, day["predicted_available_spaces"],
                        alpha=0.15, color="#2a9d8f")
        ax.axvline(hour, color="#e76f51", linestyle="--", label="Selected hour")
        ax.set(title=f"Predicted free spaces through {chosen_date:%A, %d %b %Y}",
               xlabel="Hour of day", ylabel="Free spaces",
               xlim=(0, 23), ylim=(0, result["total_spaces"]))
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close(fig)

    st.caption("The model uses: parking lot, hour, weekday/weekend, weather and special "
               "events. It does not know about real-time traffic, holidays or road closures.")

# ---------------------------------------------------------------------------
# Tab 2: model results
# ---------------------------------------------------------------------------
with tab_results:
    if config.MODEL_COMPARISON_PATH.exists():
        results = pd.read_csv(config.MODEL_COMPARISON_PATH)
        meta = {}
        if config.MODEL_METADATA_PATH.exists():
            meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        st.subheader(f"Best model: {meta.get('best_model', results.iloc[0]['model'])}")
        st.write("Scores are measured on the most recent days, which the models never "
                 "saw during training. MAE/RMSE are in occupancy-rate units "
                 "(0.05 = 5 percentage points); `test_mae_spaces` is in parking spaces.")
        st.dataframe(results.round(4), hide_index=True)
        for filename in ["model_01_comparison.png", "model_02_actual_vs_predicted.png",
                         "model_05_forecast_example.png"]:
            path = config.CHARTS_DIR / filename
            if path.exists():
                st.image(str(path))
        with st.expander("Why MAE, RMSE and R²?"):
            explanation = config.RESULTS_DIR / "metrics_explained.txt"
            if explanation.exists():
                st.text(explanation.read_text(encoding="utf-8"))
    else:
        st.info("No results yet. Run `python main.py` first.")

# ---------------------------------------------------------------------------
# Tab 3: EDA charts
# ---------------------------------------------------------------------------
with tab_insights:
    st.write("Patterns found in the **simulated** dataset (exploratory data analysis).")
    charts = sorted(config.CHARTS_DIR.glob("eda_*.png"))
    if charts:
        for chart in charts:
            st.image(str(chart), caption=chart.stem.replace("_", " "))
    else:
        st.info("No charts yet. Run `python main.py` first.")
