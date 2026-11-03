"""Tests for the Phase 2 dashboard logic (no Streamlit needed).

    python -m unittest discover -s tests -v

Run `python main.py` first - these tests use the saved Phase 1 data and model.
"""

import datetime as dt
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src import dashboard_charts as charts
from src import dashboard_data as dd
from src.features import FEATURE_COLUMNS
from src.predict import load_model, predict_many, prepare_features, status_label

READY = (config.PROCESSED_DATA_PATH.exists() and config.MODEL_PATH.exists()
         and (config.RESULTS_DIR / "test_predictions.csv").exists())


@unittest.skipUnless(READY, "run `python main.py` first")
class TestSnapshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = dd.load_dashboard_data()

    def test_snapshot_has_one_row_per_lot_and_consistent_numbers(self):
        for label, ts in dd.snapshot_choices(self.df).items():
            snap = dd.snapshot_at(self.df, ts)
            self.assertEqual(len(snap), len(config.PARKING_LOTS), label)
            self.assertTrue((snap["occupied_spaces"] + snap["available_spaces"]
                             == snap["total_spaces"]).all(), label)
            self.assertTrue(snap["status"].isin(["Available", "Limited", "Full"]).all(), label)

    def test_overall_summary_adds_up(self):
        snap = dd.snapshot_at(self.df, self.df["timestamp"].max())
        s = dd.overall_summary(snap)
        self.assertEqual(s["total_spaces"], sum(l["total_spaces"] for l in config.PARKING_LOTS.values()))
        self.assertEqual(s["occupied_spaces"] + s["available_spaces"], s["total_spaces"])
        self.assertAlmostEqual(s["occupancy_pct"], s["occupied_spaces"] / s["total_spaces"] * 100)
        self.assertEqual(s["num_locations"], len(config.PARKING_LOTS))

    def test_snapshot_is_deterministic_not_random(self):
        ts = self.df["timestamp"].max()
        pd.testing.assert_frame_equal(dd.snapshot_at(self.df, ts), dd.snapshot_at(self.df, ts))

    def test_status_thresholds(self):
        self.assertEqual(status_label(0.0), "Available")
        self.assertEqual(status_label(config.STATUS_LIMITED_FROM - 0.001), "Available")
        self.assertEqual(status_label(config.STATUS_LIMITED_FROM), "Limited")
        self.assertEqual(status_label(config.STATUS_FULL_FROM - 0.001), "Limited")
        self.assertEqual(status_label(config.STATUS_FULL_FROM), "Full")
        self.assertEqual(status_label(1.0), "Full")

    def test_unknown_snapshot_time_raises(self):
        with self.assertRaises(ValueError):
            dd.snapshot_at(self.df, "2030-01-01 00:00")


@unittest.skipUnless(READY, "run `python main.py` first")
class TestAggregations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = dd.load_dashboard_data()

    def test_hourly_profile_shape_and_range(self):
        profile = dd.hourly_profile(self.df)
        self.assertEqual(len(profile), 24 * len(config.PARKING_LOTS))
        self.assertTrue(profile["occupancy_pct"].between(0, 100).all())

    def test_weekday_and_weekend_filters_differ(self):
        weekdays = dd.hourly_profile(self.df, ["Downtown Office Plaza"], "Weekdays")
        weekends = dd.hourly_profile(self.df, ["Downtown Office Plaza"], "Weekends")
        self.assertGreater(weekdays["occupancy_pct"].max(), weekends["occupancy_pct"].max() + 30)

    def test_day_of_week_profile(self):
        profile = dd.day_of_week_profile(self.df)
        self.assertEqual(len(profile), 7 * len(config.PARKING_LOTS))
        self.assertEqual(set(profile["day_of_week"]), set(range(7)))

    def test_location_summary(self):
        summary = dd.location_summary(self.df)
        self.assertEqual(len(summary), len(config.PARKING_LOTS))
        self.assertTrue(summary["peak_hour"].between(0, 23).all())
        self.assertTrue((summary["peak_occupancy_pct"] >= summary["avg_occupancy_pct"]).all())


@unittest.skipUnless(READY, "run `python main.py` first")
class TestPredictionPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.df = dd.load_dashboard_data()
        cls.test_predictions = dd.load_test_predictions()

    def test_dashboard_uses_the_training_feature_columns(self):
        ok, message = dd.model_feature_check(self.model)
        self.assertTrue(ok, message)
        request = {"parking_lot_id": "LOT_A", "timestamp": "2026-11-04 10:00",
                   "weather": "Cloudy", "event_today": False}
        _, features = prepare_features([request])
        self.assertEqual(list(features.columns), FEATURE_COLUMNS)

    def test_prediction_equals_the_raw_saved_pipeline(self):
        """The dashboard must give exactly what the saved model gives for the same features."""
        requests = [{"parking_lot_id": lot, "timestamp": f"2026-11-0{d} {h:02d}:00",
                     "weather": w, "event_today": lot != "LOT_C"}
                    for lot in config.PARKING_LOTS for d in (4, 7) for h in (3, 12, 18)
                    for w in config.WEATHER_OPTIONS]
        _, features = prepare_features(requests)
        raw = np.clip(self.model.predict(features), 0, 1)
        shown = predict_many(requests, model=self.model)
        np.testing.assert_allclose(shown["predicted_occupancy_rate"], raw, atol=1e-4)

    def test_predictions_are_deterministic_and_consistent(self):
        request = [{"parking_lot_id": "LOT_B", "timestamp": "2026-11-07 15:00",
                    "weather": "Rainy", "event_today": True}]
        first = predict_many(request, model=self.model)
        second = predict_many(request, model=self.model)
        pd.testing.assert_frame_equal(first, second)
        row = first.iloc[0]
        self.assertEqual(row["predicted_occupied_spaces"] + row["predicted_available_spaces"],
                         row["total_spaces"])
        self.assertTrue(0 <= row["predicted_occupancy_rate"] <= 1)

    def test_inputs_change_the_prediction(self):
        """Guards against hardcoded output: different inputs must give different answers."""
        def free(lot, when, weather="Sunny", event=False):
            return predict_many([{"parking_lot_id": lot, "timestamp": when, "weather": weather,
                                  "event_today": event}], model=self.model).iloc[0]["predicted_available_spaces"]
        self.assertLess(free("LOT_A", "2026-11-04 12:00"), free("LOT_A", "2026-11-04 03:00"))
        self.assertLess(free("LOT_A", "2026-11-04 12:00"), free("LOT_A", "2026-11-08 12:00"))
        self.assertLess(free("LOT_B", "2026-11-07 19:00", event=True),
                        free("LOT_B", "2026-11-07 19:00", event=False))
        self.assertNotEqual(free("LOT_A", "2026-11-04 12:00"), free("LOT_D", "2026-11-04 12:00"))


@unittest.skipUnless(READY, "run `python main.py` first")
class TestActualVsPredicted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tp = dd.load_test_predictions()
        cls.df = dd.load_dashboard_data()
        cls.meta = dd.load_model_metadata()

    def test_only_the_held_out_period_is_available(self):
        test_start = pd.Timestamp(self.meta["split"]["test_start"])
        self.assertGreaterEqual(self.tp["timestamp"].min(), test_start)

    def test_window_and_metrics(self):
        start = self.tp["timestamp"].min()
        window = dd.prediction_window(self.tp, "City Center Mall", start, 3)
        self.assertEqual(len(window), 72)
        m = dd.prediction_metrics(window)
        self.assertEqual(m["rows"], 72)
        manual_mae = (window["actual_available_spaces"] - window["predicted_available_spaces"]).abs().mean()
        self.assertAlmostEqual(m["mae"], manual_mae)
        self.assertGreaterEqual(m["rmse"], m["mae"])
        self.assertLess(m["r2"], 1.0)

    def test_too_few_rows_gives_no_metrics(self):
        self.assertIsNone(dd.prediction_metrics(self.tp.head(1)))

    def test_recorded_reading_distinguishes_test_from_training(self):
        test_hour = self.tp["timestamp"].min() + pd.Timedelta(hours=12)
        self.assertEqual(dd.recorded_reading(self.df, self.tp, "LOT_A", test_hour)["kind"], "test")
        self.assertEqual(dd.recorded_reading(self.df, self.tp, "LOT_A", "2026-01-10 12:00")["kind"], "training")
        self.assertIsNone(dd.recorded_reading(self.df, self.tp, "LOT_A", "2027-01-10 12:00"))


class TestInputValidation(unittest.TestCase):
    TODAY = dt.date(2026, 11, 1)

    def check(self, **overrides):
        args = dict(lot_id="LOT_A", when="2026-11-05 10:00", weather="Sunny",
                    event_today=False, today=self.TODAY)
        args.update(overrides)
        return dd.validate_prediction_inputs(args["lot_id"], args["when"], args["weather"],
                                             args["event_today"], args["today"])

    def test_normal_input_has_no_messages(self):
        self.assertEqual(self.check(), ([], []))

    def test_bad_lot_weather_and_date_are_errors(self):
        self.assertTrue(self.check(lot_id="LOT_Z")[0])
        self.assertTrue(self.check(weather="Snowy")[0])
        self.assertTrue(self.check(when="not a date")[0])

    def test_event_rules(self):
        self.assertTrue(self.check(lot_id="LOT_C", event_today=True)[0])            # hospital: no events
        errors, warnings = self.check(lot_id="LOT_B", event_today=True, when="2026-11-05 09:00")
        self.assertFalse(errors)
        self.assertTrue(warnings)                                                    # outside event hours
        self.assertEqual(self.check(lot_id="LOT_B", event_today=True, when="2026-11-05 19:00"), ([], []))

    def test_far_future_dates(self):
        _, warnings = self.check(when="2026-12-20 10:00")
        self.assertTrue(any("what-if" in w for w in warnings))
        errors, _ = self.check(when=pd.Timestamp(self.TODAY) + pd.Timedelta(days=dd.MAX_FORECAST_DAYS + 1))
        self.assertTrue(errors)

    def test_next_date_for_weekday(self):
        for weekday in range(7):
            result = dd.next_date_for_weekday(weekday, self.TODAY)
            self.assertEqual(result.dayofweek, weekday)
            self.assertTrue(0 <= (result - pd.Timestamp(self.TODAY)).days <= 6)


@unittest.skipUnless(READY, "run `python main.py` first")
class TestCharts(unittest.TestCase):
    """Every chart builds without errors, for normal and empty input."""

    @classmethod
    def setUpClass(cls):
        cls.df = dd.load_dashboard_data()
        cls.tp = dd.load_test_predictions()
        cls.model = load_model()

    def done(self, fig):
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_all_charts(self):
        ts = self.df["timestamp"].max()
        snap = dd.snapshot_at(self.df, ts)
        self.done(charts.fig_last_24_hours(self.df, ts))
        self.done(charts.fig_last_24_hours(self.df, "2025-01-01"))           # nothing there -> message
        self.done(charts.fig_snapshot_stacked(snap))
        for metric in ("occupancy_pct", "occupied_spaces", "available_spaces"):
            self.done(charts.fig_hourly_demand(dd.hourly_profile(self.df), metric))
        self.done(charts.fig_hourly_demand(dd.hourly_profile(self.df, ["nowhere"])))   # empty
        self.done(charts.fig_day_of_week(dd.day_of_week_profile(self.df)))
        self.done(charts.fig_day_of_week(dd.day_of_week_profile(self.df, ["nowhere"])))
        for location in self.df["location"].unique():
            self.done(charts.fig_weekday_hour_heatmap(self.df, location))
        for metric in ("avg_occupancy_pct", "peak_occupancy_pct", "avg_available_spaces", "total_spaces"):
            self.done(charts.fig_location_comparison(dd.location_summary(self.df), metric))
        window = dd.prediction_window(self.tp, "Central Hospital", self.tp["timestamp"].min(), 2)
        self.done(charts.fig_actual_vs_predicted_series(window, "Central Hospital"))
        self.done(charts.fig_actual_vs_predicted_scatter(window, "Central Hospital"))
        self.done(charts.fig_actual_vs_predicted_series(window.iloc[0:0], "Central Hospital"))

    def test_prediction_charts(self):
        from src.predict import predict_full_day
        day = predict_full_day("LOT_B", "2026-11-07", "Rainy", True, model=self.model)
        for hour in (0, 12, 23):
            self.done(charts.fig_forecast_day(day, hour, 500, "title"))
        everyone = predict_many([{"parking_lot_id": lot, "timestamp": "2026-11-07 15:00",
                                  "weather": "Sunny", "event_today": False}
                                 for lot in config.PARKING_LOTS], model=self.model)
        self.done(charts.fig_prediction_all_lots(everyone))


if __name__ == "__main__":
    unittest.main()
