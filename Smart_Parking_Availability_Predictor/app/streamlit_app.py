"""Smart Parking Availability Predictor - Streamlit dashboard (Phase 2 + 3).

Run from the project root folder:
    streamlit run app/streamlit_app.py

The dashboard only READS the artifacts created by `python main.py`
(dataset, trained model, evaluation results). It never retrains anything.
The previous single-page Phase 1 app is kept as app/streamlit_app_phase1.py.
"""

import sys
from pathlib import Path

# Make `import src...` and `import app...` work from any starting folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app import ui
from app.context import AppContext
from app.views import (about, analytics, availability, evaluation, history, home, prediction,
                       recommendations)
from src import config
from src import dashboard_data as dd
from src.predict import load_model

st.set_page_config(page_title="Smart Parking Dashboard", page_icon="\U0001F17F️",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

PAGES = {
    "\U0001F3E0 Home": home,
    "\U0001F697 Parking Availability": availability,
    "\u2B50 Smart Recommendations": recommendations,
    "\U0001F52E Predict Availability": prediction,
    "\U0001F4CA Charts & Analytics": analytics,
    "\U0001F4C8 Historical Analysis": history,
    "\U0001F9EA Model Evaluation": evaluation,
    "\u2139\uFE0F About": about,
}


# ---------------------------------------------------------------------------
# Cached loading: the files are read once, not on every click.
# (Streamlit reruns this script on every interaction.)
# The "stamp" holds the files' last-modified times, so if you run
# `python main.py` again while the app is open, the new data/model is loaded.
# ---------------------------------------------------------------------------
def artifact_stamp():
    files = [config.PROCESSED_DATA_PATH, config.MODEL_PATH, config.MODEL_COMPARISON_PATH,
             config.MODEL_METADATA_PATH, config.RESULTS_DIR / "test_predictions.csv",
             config.RESULTS_DIR / "per_lot_test_error.csv", config.ALL_TEST_PREDICTIONS_PATH,
             config.FEATURE_IMPORTANCE_PATH, config.ERROR_BANDS_PATH, config.DATA_SOURCE_PATH,
             config.DISTANCE_FILE]
    return tuple(f.stat().st_mtime if f.exists() else 0.0 for f in files)


@st.cache_data(show_spinner="Loading data...")
def load_data(stamp):
    return dd.load_dashboard_data()


@st.cache_data(show_spinner=False)
def load_results(stamp):
    return (dd.load_test_predictions(), dd.load_model_comparison(),
            dd.load_model_metadata(), dd.load_per_lot_error(),
            dd.load_all_test_predictions(), dd.load_feature_importance(), dd.load_data_source())


@st.cache_resource(show_spinner="Loading model...")
def cached_model(stamp):
    return load_model()


def show_setup_help(error):
    """Friendly message when Phase 1 outputs are missing."""
    st.title("Smart Parking Dashboard")
    st.error(f"The dashboard could not start: {error}")
    st.info("The dashboard needs the files created by the Phase 1 pipeline. "
            "Open a terminal in the project folder and run:")
    st.code("python main.py", language="bash")
    st.caption("Then refresh this page.")


def main():
    try:
        stamp = artifact_stamp()
        df = load_data(stamp)
    except FileNotFoundError as error:
        show_setup_help(error)
        return
    except Exception as error:   # unreadable / corrupted CSV
        show_setup_help(f"the dataset could not be read ({error})")
        return

    (test_predictions, model_comparison, metadata, per_lot_error,
     all_predictions, importance, data_source) = load_results(stamp)
    ui.set_data_kind(data_source.get("is_simulated", True))

    # ---- sidebar: navigation + snapshot picker --------------------------------
    with st.sidebar:
        st.markdown("## \U0001F17F️ Smart Parking")
        page_name = st.radio("Navigation", list(PAGES), label_visibility="collapsed", key="nav_page")
        st.divider()

        choices = dd.snapshot_choices(df)
        snapshot_label = st.selectbox(
            "Snapshot shown on Home & Availability", list(choices), key="snapshot_choice",
            help="The dataset has no live feed, so 'current' means the moment you "
                 "pick here. The default is the most recent reading in the dataset.")
        snapshot_ts = choices[snapshot_label]
        st.caption(f"Showing: {snapshot_ts:%a %d %b %Y, %H:%M}")
        st.divider()
        st.markdown(ui.data_pill(), unsafe_allow_html=True)
        st.caption(f"{len(df):,} hourly records • {df['parking_lot_id'].nunique()} locations")
        if metadata.get("best_model"):
            st.caption(f"Model: {metadata['best_model']}")

    ctx = AppContext(
        df=df,
        snapshot_ts=snapshot_ts,
        snapshot=dd.snapshot_at(df, snapshot_ts),
        previous_snapshot=dd.previous_snapshot(df, snapshot_ts),
        test_predictions=test_predictions,
        model_comparison=model_comparison,
        metadata=metadata,
        per_lot_error=per_lot_error,
        get_model=lambda: cached_model(stamp),
        is_simulated=bool(data_source.get("is_simulated", True)),
        all_predictions=all_predictions,
        importance=importance,
    )

    # ---- keep model, dataset and dashboard consistent --------------------------------
    try:
        ctx.issues = dd.consistency_issues(df, metadata, cached_model(stamp), test_predictions)
    except Exception:            # a missing/unreadable model is reported by the pages that need it
        ctx.issues = []
    for level, message in ctx.issues:
        (st.error if level == "error" else st.warning)(message)

    try:
        PAGES[page_name].render(ctx)
    except Exception as error:   # never show a raw traceback in the demo
        st.error(f"Something went wrong on this page: {error}")
        st.info("Try another page or snapshot. If it keeps happening, run `python main.py` "
                "to rebuild the data and model, then refresh.")
        with st.expander("Technical details"):
            st.exception(error)


main()
