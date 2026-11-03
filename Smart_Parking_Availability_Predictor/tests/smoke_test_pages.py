"""Smoke test: run EVERY dashboard page with many different inputs.

    python tests/smoke_test_pages.py

It uses tests/fake_streamlit.py instead of the real Streamlit, so it needs no
browser and no Streamlit installation. It fails if any page raises an error,
shows an error box, prints "nan", or draws no chart where one is expected.
Run `python main.py` first (the pages need the Phase 1 data and model).
"""

import datetime as dt
import itertools
import re
import runpy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in (PROJECT_ROOT, PROJECT_ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import fake_streamlit

fake = fake_streamlit.install()      # must happen BEFORE the app is imported

from src import config
from src import dashboard_data as dd

APP_FILE = PROJECT_ROOT / "app" / "streamlit_app.py"
PAGES = ["\U0001F3E0 Home", "\U0001F697 Parking Availability", "\U0001F52E Predict Availability",
         "\U0001F4CA Charts & Analytics", "ℹ️ About"]
HOME, AVAILABILITY, PREDICT, ANALYTICS, ABOUT = PAGES
RECOMMEND = "\u2B50 Smart Recommendations"
HISTORY = "\U0001F4C8 Historical Analysis"
EVALUATION = "\U0001F9EA Model Evaluation"
READY = (config.PROCESSED_DATA_PATH.exists() and config.MODEL_PATH.exists()
         and (config.RESULTS_DIR / "test_predictions.csv").exists())


def run_app(page, expect_errors=False, **overrides):
    """Run the whole app script once, like Streamlit does after a click."""
    fake.reset()
    fake.current_name = re.sub(r"\W+", "_", page).strip("_")[:24]
    fake.overrides = {"nav_page": page, **overrides}
    runpy.run_path(str(APP_FILE), run_name="__main__")
    if not expect_errors:
        assert not fake.errors, f"{page} {overrides}: app showed errors: {fake.errors}"
    joined = "\n".join(fake.texts)
    assert not re.search(r"\bnan\b", joined, re.IGNORECASE), f"{page} {overrides}: 'nan' on screen"
    assert "None" not in re.sub(r"<[^>]+>", "", joined).split(), f"{page} {overrides}: 'None' on screen"
    return fake


@unittest.skipUnless(READY, "run `python main.py` first")
class TestEveryPage(unittest.TestCase):
    def test_all_pages_open(self):
        expected_figures = {HOME: 1, AVAILABILITY: 1, PREDICT: 2, ANALYTICS: 6, ABOUT: 0}
        for page in PAGES:
            with self.subTest(page=page):
                st = run_app(page)
                self.assertGreaterEqual(st.figures, expected_figures[page])

    def test_every_snapshot_on_home_and_availability(self):
        import pandas as pd
        choices = dd.snapshot_choices(dd.load_dashboard_data())
        for label in choices:
            for page in (HOME, AVAILABILITY):
                with self.subTest(snapshot=label, page=page):
                    run_app(page, snapshot_choice=label)

    def test_availability_filters_and_sorting(self):
        from app.views.availability import SORT_OPTIONS
        for sort in SORT_OPTIONS:
            run_app(AVAILABILITY, av_sort=sort)
        for status in ("Available", "Limited", "Full"):
            run_app(AVAILABILITY, av_status_filter=[status])
        st = run_app(AVAILABILITY, av_status_filter=[])      # nothing selected: friendly message
        self.assertTrue(any("No location matches" in m for m in st.infos))

    def test_predict_every_lot_weather_hour_and_event(self):
        today = dt.date.today()
        for lot, weather, hour, event in itertools.product(
                config.PARKING_LOTS, config.WEATHER_OPTIONS, (0, 9, 17, 23), (False, True)):
            with self.subTest(lot=lot, weather=weather, hour=hour, event=event):
                run_app(PREDICT, pred_lot=lot, pred_weather=weather, pred_hour=hour,
                        pred_event=event, pred_date=today + dt.timedelta(days=3))

    def test_predict_by_day_of_week(self):
        for name in dd.DAY_NAMES_FULL:
            with self.subTest(day=name):
                run_app(PREDICT, pred_mode="Day of week", pred_weekday=name)

    def test_predict_date_limits(self):
        today = dt.date.today()
        first = pd_date(config.START_DATE)
        run_app(PREDICT, pred_date=first)                                   # first training day
        run_app(PREDICT, pred_date=today + dt.timedelta(days=dd.MAX_FORECAST_DAYS))   # last allowed
        st = run_app(PREDICT, pred_date=today + dt.timedelta(days=30))
        self.assertTrue(any("what-if" in w for w in st.warnings))           # far-future weather
        # Streamlit's date picker would not allow this date; if it ever got through, the
        # page-level guard must show a friendly message instead of crashing.
        st = run_app(PREDICT, expect_errors=True,
                     pred_date=today + dt.timedelta(days=dd.MAX_FORECAST_DAYS + 1))
        self.assertTrue(any("Something went wrong" in e for e in st.errors))

    def test_predict_compares_with_recorded_data_only_when_fair(self):
        tp = dd.load_test_predictions()
        in_test = (tp["timestamp"].min() + dt.timedelta(days=2)).date()
        st = run_app(PREDICT, pred_date=in_test, pred_hour=12, pred_lot="LOT_A")
        self.assertTrue(any("Reality check" in t for t in st.texts))          # unseen test hour
        st = run_app(PREDICT, pred_date=dt.date(2026, 2, 10), pred_hour=12)
        self.assertFalse(any("Reality check" in t for t in st.texts))
        self.assertTrue(any("training data" in m for m in st.infos))          # training hour: no fair test
        st = run_app(PREDICT, pred_date=dt.date.today() + dt.timedelta(days=2))
        self.assertFalse(any("Reality check" in t for t in st.texts))         # real forecast: nothing to compare

    def test_predict_event_outside_event_hours_warns(self):
        st = run_app(PREDICT, pred_lot="LOT_B", pred_event=True, pred_hour=9)
        self.assertTrue(any("Events only affect" in w for w in st.warnings))

    def test_analytics_controls(self):
        names = sorted(dd.load_dashboard_data()["location"].unique())
        for day_type, metric in itertools.product(("All days", "Weekdays", "Weekends"),
                                                  ("Occupancy (%)", "Occupied spaces", "Available spaces")):
            run_app(ANALYTICS, an_hourly_days=day_type, an_hourly_metric=metric)
        for metric in ("Average occupancy (%)", "Peak hourly occupancy (%)",
                       "Average available spaces", "Total spaces"):
            run_app(ANALYTICS, an_cmp_metric=metric)
        st = run_app(ANALYTICS, an_hourly_locs=[], an_dow_locs=[])           # nothing selected
        self.assertTrue(any("Select at least one location" in m for m in st.infos))
        run_app(ANALYTICS, an_hourly_locs=names[:1], an_dow_locs=names[:2])

    def test_actual_vs_predicted_controls(self):
        tp = dd.load_test_predictions()
        first, last = tp["timestamp"].min().date(), tp["timestamp"].max().date()
        for location, span in itertools.product(sorted(tp["location"].unique()), (1, 7, 18)):
            run_app(ANALYTICS, an_avp_loc=location, an_avp_days=span, an_avp_start=first)
        run_app(ANALYTICS, an_avp_start=last, an_avp_days=14)               # only one day left
        # a start date before the test period (training days) can never be chosen
        st = run_app(ANALYTICS, expect_errors=True, an_avp_start=first - dt.timedelta(days=1))
        self.assertTrue(any("Something went wrong" in e for e in st.errors))


def pd_date(text):
    return dt.date.fromisoformat(text)


@unittest.skipUnless(READY and config.ALL_TEST_PREDICTIONS_PATH.exists(), "run `python main.py` first")
class TestPhase3Pages(unittest.TestCase):
    def test_navigation_lists_every_page(self):
        st = run_app(HOME)
        # the radio received all eight pages and the three new ones can be opened
        for page in (RECOMMEND, HISTORY, EVALUATION):
            run_app(page)
        import app.streamlit_app as application
        self.assertEqual(len(application.PAGES), 8)

    def test_new_pages_open_and_draw_charts(self):
        for page, minimum in ((RECOMMEND, 2), (HISTORY, 6), (EVALUATION, 3)):
            with self.subTest(page=page):
                self.assertGreaterEqual(run_app(page).figures, minimum)

    # ---- recommendations -------------------------------------------------------
    def test_recommendations_every_combination(self):
        names = [info["location"] for info in config.PARKING_LOTS.values()]
        today = dt.date.today()
        for destination, weather, rank in itertools.product(
                ["No specific destination"] + names, ("Not sure", "Rainy"),
                ("Most free spaces", "Highest share of free spaces")):
            with self.subTest(destination=destination, weather=weather, rank=rank):
                st = run_app(RECOMMEND, rec_destination=destination, rec_weather=weather, rec_rank=rank,
                             rec_date=today + dt.timedelta(days=2))
                self.assertTrue(any("estimates" in t for t in st.texts))
                self.assertTrue(any("availability only" in t for t in st.texts))   # distance is not used

    def test_recommendations_headline_and_subsets(self):
        st = run_app(RECOMMEND, rec_hour=12, rec_date=dt.date.today() + dt.timedelta(days=1))
        self.assertTrue(any(t.startswith("Best choice:") or "Full" in t for t in st.texts))
        run_app(RECOMMEND, rec_locations=["Central Hospital"])
        run_app(RECOMMEND, rec_locations=["Central Hospital", "City Center Mall"], rec_event=True, rec_hour=19)
        for hour, span in itertools.product((0, 11, 23), (3, 6, 12, 24)):
            run_app(RECOMMEND, rec_hour=hour, rec_span=span)

    def test_recommendations_without_locations_shows_a_message(self):
        st = run_app(RECOMMEND, rec_locations=[])
        self.assertTrue(any("at least one parking location" in m for m in st.infos))

    def test_recommendations_date_guard(self):
        st = run_app(RECOMMEND, expect_errors=True, rec_date=dt.date.today() - dt.timedelta(days=1))
        self.assertTrue(any("Something went wrong" in e for e in st.errors))      # past dates cannot be picked

    def test_recommendations_use_distance_file_when_present(self):
        path = config.DISTANCE_FILE
        self.assertFalse(path.exists(), "a distance file already exists; remove it before testing")
        rows = ["destination,parking_lot_id,distance_km"]
        for lot, km in {"LOT_A": 0.4, "LOT_B": 6.0, "LOT_C": 0.2, "LOT_D": 3.0}.items():
            rows.append(f"Central Hospital,{lot},{km}")
        path.write_text("\n".join(rows), encoding="utf-8")
        try:
            st = run_app(RECOMMEND, rec_destination="Central Hospital", rec_distance_weight=0.8)
            joined = "\n".join(st.texts)
            self.assertIn("km away", joined)
            self.assertIn("Distances from your distance file are used", joined)
            self.assertNotIn("does not consider how far", joined)
            run_app(RECOMMEND, rec_destination="No specific destination")           # distance but no destination
        finally:
            path.unlink()
        st = run_app(RECOMMEND)                                                     # back to availability only
        self.assertNotIn("km away", "\n".join(st.texts))

    # ---- time slots on the Predict page ---------------------------------------------
    def test_predict_time_slots(self):
        for hours in (3, 6, 12, 24, 48):
            with self.subTest(hours=hours):
                st = run_app(PREDICT, pred_slot_hours=hours)
                self.assertGreaterEqual(st.figures, 4)             # day chart, all-lots chart, 2 slot charts
                self.assertTrue(any("Upcoming time slots" in t for t in st.texts))
        run_app(PREDICT, pred_slot_lots=["LOT_B"])
        st = run_app(PREDICT, pred_slot_lots=[])
        self.assertTrue(any("at least one location" in m for m in st.infos))
        st = run_app(PREDICT)
        self.assertTrue(any("not guaranteed" in t or "estimates" in t for t in st.texts))

    # ---- history -----------------------------------------------------------------
    def test_history_controls(self):
        names = sorted(dd.load_dashboard_data()["location"].unique())
        for location in names:
            run_app(HISTORY, hist_wd_loc=location)
        run_app(HISTORY, hist_peak_locs=names[:1], hist_trend_locs=names[:2])
        st = run_app(HISTORY, hist_peak_locs=[], hist_trend_locs=[])
        self.assertGreaterEqual(sum("at least one location" in m for m in st.infos), 2)

    # ---- evaluation ----------------------------------------------------------------
    def test_evaluation_controls(self):
        allp = dd.load_all_test_predictions()
        first = allp["timestamp"].min().date()
        for model, location in itertools.product(sorted(allp["model"].unique()), sorted(allp["location"].unique())):
            with self.subTest(model=model, location=location):
                run_app(EVALUATION, ev_model=model, ev_location=location, ev_start=first)
        for days in (1, 3, 7, 18):
            run_app(EVALUATION, ev_days=days)
        for tree in ("Decision Tree", "Random Forest", "Gradient Boosting"):
            run_app(EVALUATION, ev_tree_model=tree)
        st = run_app(EVALUATION, expect_errors=True, ev_start=first - dt.timedelta(days=1))
        self.assertTrue(any("Something went wrong" in e for e in st.errors))        # training days cannot be picked

    def test_evaluation_shows_what_matters(self):
        joined = "\n".join(run_app(EVALUATION).texts)
        for needle in ("MAE", "RMSE", "Unseen data only", "Selection rule", "train/test separation checks passed",
                       "time of day"):
            self.assertIn(needle, joined)

    # ---- robustness: missing artifacts and inconsistent files ---------------------------
    def test_missing_phase3_files_give_messages_not_crashes(self):
        originals = (dd.load_all_test_predictions, dd.load_feature_importance, dd.load_leakage_checks)
        dd.load_all_test_predictions = lambda: None
        dd.load_feature_importance = lambda: None
        dd.load_leakage_checks = lambda: []
        try:
            st = run_app(EVALUATION)
            self.assertTrue(any("No test predictions found" in m for m in st.infos))
            self.assertTrue(any("No feature importance found" in m for m in st.infos))
            self.assertTrue(any("No leakage report found" in m for m in st.infos))
            for page in PAGES + [RECOMMEND, HISTORY]:
                run_app(page)                                          # every other page still works
        finally:
            dd.load_all_test_predictions, dd.load_feature_importance, dd.load_leakage_checks = originals

    def test_inconsistent_metadata_is_shown_on_every_page(self):
        original = dd.load_model_metadata
        dd.load_model_metadata = lambda: {}
        try:
            for page in (HOME, RECOMMEND, EVALUATION):
                st = run_app(page)
                self.assertTrue(any("No model metadata" in w for w in st.warnings), page)
        finally:
            dd.load_model_metadata = original

    def test_changed_dataset_triggers_an_error_banner(self):
        original = dd.load_dashboard_data
        dd.load_dashboard_data = lambda *a, **k: original(*a, **k).iloc[:-4].reset_index(drop=True)
        try:
            st = run_app(HOME, expect_errors=True, snapshot_choice="Weekday lunchtime (12:00)")
            self.assertTrue(any("dataset has changed" in e for e in st.errors))
        finally:
            dd.load_dashboard_data = original

    def test_genuine_data_changes_the_labels(self):
        original = dd.load_data_source
        dd.load_data_source = lambda: {"is_simulated": False, "source_file": "my.csv"}
        try:
            st = run_app(HOME, expect_errors=True)
            joined = "\n".join(st.texts)
            self.assertIn("REAL DATA", joined)
            self.assertIn("Supplied data.", joined)
            self.assertNotIn("Simulated data.", joined)
            st = run_app(ABOUT, expect_errors=True)
            self.assertIn("Your parking data", "\n".join(st.texts))
        finally:
            dd.load_data_source = original
        st = run_app(HOME)
        self.assertIn("SIMULATED DATA", "\n".join(st.texts))



@unittest.skipUnless(READY, "run `python main.py` first")
class TestMissingFiles(unittest.TestCase):
    def test_friendly_message_when_model_is_missing(self):
        """Model file missing -> the Predict page explains how to fix it; other pages still work."""
        import src.predict as predict_module
        original = predict_module.load_model

        def missing(*a, **k):
            raise FileNotFoundError("Trained model not found: demo")
        predict_module.load_model = missing
        try:
            st = run_app(PREDICT, expect_errors=True)
            self.assertTrue(any("could not be loaded" in e for e in st.errors))
            self.assertTrue(any("python main.py" in t for t in st.texts))
            run_app(HOME)                    # the rest of the dashboard is unaffected
            run_app(ANALYTICS)
        finally:
            predict_module.load_model = original

    def test_friendly_message_when_data_is_missing(self):
        """No data file -> a help message, not a crash."""
        import app.streamlit_app  # noqa: F401  (loads the module once so we can patch it)
        original = dd.load_dashboard_data

        def missing(*a, **k):
            raise FileNotFoundError("Processed data not found: demo")
        dd.load_dashboard_data = missing
        try:
            st = run_app(HOME, expect_errors=True)
            self.assertTrue(any("could not start" in e for e in st.errors))
            self.assertTrue(any("python main.py" in t for t in st.texts))
        finally:
            dd.load_dashboard_data = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
