"""Tests for the Phase 3 features: robustness, time slots, recommendations, analysis, data checks.

    python -m unittest discover -s tests -v

Run `python main.py` first - most tests use the saved dataset and model.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import analysis, config, data_checks, recommend
from src import dashboard_data as dd
from src import predict as P
from src.features import FEATURE_COLUMNS

READY = (config.PROCESSED_DATA_PATH.exists() and config.MODEL_PATH.exists()
         and config.ALL_TEST_PREDICTIONS_PATH.exists() and config.FEATURE_IMPORTANCE_PATH.exists())


class StubModel:
    """A fake model that returns whatever numbers we give it (to test the safety limits)."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float) if np.isscalar(self.value) else np.asarray(self.value, float)


# ----------------------------------------------------------------------------
class TestInputHandling(unittest.TestCase):
    def test_missing_weather_and_event_get_safe_defaults(self):
        clean, notes = P.normalize_request({"parking_lot_id": " lot_a ", "timestamp": "2026-11-04 10:30"})
        self.assertEqual((clean["parking_lot_id"], clean["weather"], clean["event_today"]),
                         ("LOT_A", "Unknown", False))
        self.assertTrue(notes)                                   # the user is told about the default
        for blank in (None, float("nan"), "", "  ", "Not sure"):
            self.assertEqual(P.normalize_request({"parking_lot_id": "LOT_A", "timestamp": "2026-11-04",
                                                  "weather": blank})[0]["weather"], "Unknown")

    def test_text_values_are_understood(self):
        clean, _ = P.normalize_request({"parking_lot_id": "LOT_B", "timestamp": "2026-11-04 10:00",
                                        "weather": "rainy", "event_today": "Yes"})
        self.assertEqual((clean["weather"], clean["event_today"]), ("Rainy", True))

    def test_invalid_values_raise_clear_errors(self):
        bad = [({"parking_lot_id": None, "timestamp": "2026-11-04"}, "No parking location"),
               ({"parking_lot_id": "LOT_Z", "timestamp": "2026-11-04"}, "Unknown parking lot"),
               ({"parking_lot_id": "LOT_A", "timestamp": None}, "No date and time"),
               ({"parking_lot_id": "LOT_A", "timestamp": "garbage"}, "could not be understood"),
               ({"parking_lot_id": "LOT_A", "timestamp": "2026-11-04", "weather": "Snowy"}, "Unknown weather"),
               ({"parking_lot_id": "LOT_A", "timestamp": "2026-11-04", "event_today": "maybe"}, "event_today")]
        for request, fragment in bad:
            with self.assertRaises(ValueError, msg=str(request)) as caught:
                P.normalize_request(request)
            self.assertIn(fragment, str(caught.exception))

    def test_empty_request_list(self):
        for empty in ([], None):
            with self.assertRaises(ValueError):
                P.prepare_features(empty)


@unittest.skipUnless(READY, "run `python main.py` first")
class TestCapacityLimits(unittest.TestCase):
    """Predicted free spaces can never be below 0 or above the lot's capacity."""

    def requests(self):
        return [{"parking_lot_id": lot, "timestamp": "2026-11-04 12:00", "weather": "Sunny"}
                for lot in config.PARKING_LOTS]

    def test_extreme_model_outputs_are_clipped(self):
        for value in (-5.0, -0.01, 0.0, 1.0, 1.7, 1e9):
            result = P.predict_many(self.requests(), model=StubModel(value))
            total = result["total_spaces"]
            self.assertTrue((result["predicted_available_spaces"] >= 0).all(), value)
            self.assertTrue((result["predicted_available_spaces"] <= total).all(), value)
            self.assertTrue((result["predicted_occupied_spaces"] >= 0).all(), value)
            self.assertTrue((result["predicted_occupied_spaces"] <= total).all(), value)
            self.assertTrue(((result["predicted_available_spaces"] + result["predicted_occupied_spaces"]) == total).all())
            self.assertTrue(result["predicted_occupancy_rate"].between(0, 1).all())
            if "likely_low" in result:
                self.assertTrue((result["likely_low"] >= 0).all() and (result["likely_high"] <= total).all())
                self.assertTrue((result["likely_low"] <= result["predicted_available_spaces"]).all())
                self.assertTrue((result["predicted_available_spaces"] <= result["likely_high"]).all())

    def test_status_matches_extremes(self):
        self.assertTrue((P.predict_many(self.requests(), model=StubModel(-3))["status"] == "Available").all())
        self.assertTrue((P.predict_many(self.requests(), model=StubModel(2))["status"] == "Full").all())

    def test_non_numeric_model_output_is_rejected(self):
        for bad in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                P.predict_many(self.requests(), model=StubModel(bad))

    def test_real_model_always_within_capacity_for_every_hour(self):
        model = P.load_model()
        for weather in config.WEATHER_OPTIONS + ["Unknown"]:
            slots = P.predict_slots(list(config.PARKING_LOTS), "2026-11-07 00:00", 48, weather, True, model=model)
            self.assertTrue((slots["predicted_available_spaces"].between(0, slots["total_spaces"])).all())


@unittest.skipUnless(READY, "run `python main.py` first")
class TestTimeSlots(unittest.TestCase):
    def test_shape_order_and_labels(self):
        slots = P.predict_slots(["LOT_A", "LOT_B"], "2026-11-07 15:40", 6, "Rainy")
        self.assertEqual(len(slots), 12)
        self.assertEqual(slots["timestamp"].min(), pd.Timestamp("2026-11-07 15:00"))     # rounded down
        self.assertEqual(slots["timestamp"].max(), pd.Timestamp("2026-11-07 20:00"))
        self.assertEqual(sorted(slots["slot"].unique()), list(range(6)))
        self.assertTrue(slots["slot_label"].str.contains(":00").all())

    def test_missing_weather_is_allowed(self):
        slots = P.predict_slots("LOT_A", "2026-11-07 09:00", 3, None)
        self.assertEqual(len(slots), 3)

    def test_event_only_applies_to_lots_with_events(self):
        # LOT_C (hospital) has no events: the flag must be ignored, not raise
        with_event = P.predict_slots("LOT_C", "2026-11-07 18:00", 3, "Sunny", True)
        without = P.predict_slots("LOT_C", "2026-11-07 18:00", 3, "Sunny", False)
        pd.testing.assert_frame_equal(with_event, without)
        busy = P.predict_slots("LOT_B", "2026-11-07 18:00", 3, "Sunny", True)
        calm = P.predict_slots("LOT_B", "2026-11-07 18:00", 3, "Sunny", False)
        self.assertLess(busy["predicted_available_spaces"].sum(), calm["predicted_available_spaces"].sum())

    def test_invalid_arguments(self):
        for kwargs in ({"lot_ids": [], "start": "2026-11-07", "hours": 3},
                       {"lot_ids": "LOT_A", "start": "2026-11-07", "hours": 0},
                       {"lot_ids": "LOT_A", "start": "2026-11-07", "hours": config.MAX_SLOT_HOURS + 1},
                       {"lot_ids": "LOT_A", "start": "2026-11-07", "hours": "abc"},
                       {"lot_ids": "LOT_A", "start": None, "hours": 3},
                       {"lot_ids": "LOT_A", "start": "not a time", "hours": 3},
                       {"lot_ids": "LOT_Q", "start": "2026-11-07", "hours": 3}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                P.predict_slots(**kwargs)

    def test_slots_match_single_predictions(self):
        slots = P.predict_slots("LOT_D", "2026-11-04 07:00", 4, "Cloudy")
        for row in slots.itertuples():
            single = P.predict_occupancy("LOT_D", row.timestamp, "Cloudy")
            self.assertEqual(row.predicted_available_spaces, single["predicted_available_spaces"])


# ----------------------------------------------------------------------------
@unittest.skipUnless(READY, "run `python main.py` first")
class TestRecommendations(unittest.TestCase):
    ALL = list(config.PARKING_LOTS)
    WHEN = "2026-11-04 12:00"

    def test_recommends_the_lot_with_most_predicted_free_spaces(self):
        result = recommend.recommend(self.ALL, self.WHEN, "Sunny")
        table = result["table"]
        open_lots = table[~table["is_full"]]
        self.assertEqual(table.iloc[0]["predicted_available_spaces"], open_lots["predicted_available_spaces"].max())
        self.assertEqual(result["best"]["location"], table.iloc[0]["location"])
        self.assertTrue(table["recommended"].sum() == 1)
        self.assertEqual(list(table["rank"]), list(range(1, len(table) + 1)))

    def test_states_that_distance_is_not_used(self):
        result = recommend.recommend(self.ALL, self.WHEN, "Sunny")
        self.assertFalse(result["uses_distance"])
        self.assertTrue(any("not consider how far" in n or "no travel-distance" in n for n in result["notes"]))
        self.assertTrue(any("estimates" in n for n in result["notes"]))

    def test_each_option_shows_free_spaces_and_occupancy(self):
        table = recommend.recommend(self.ALL, self.WHEN, "Sunny")["table"]
        for column in ("predicted_available_spaces", "predicted_occupancy_rate", "status", "reason"):
            self.assertIn(column, table.columns)
        self.assertTrue(table["predicted_available_spaces"].between(0, table["total_spaces"]).all())

    def test_preferred_locations_limit_the_choice(self):
        result = recommend.recommend(["LOT_A", "LOT_D"], self.WHEN, "Sunny")
        self.assertEqual(set(result["table"]["parking_lot_id"]), {"LOT_A", "LOT_D"})
        result = recommend.recommend(["lot_a", "LOT_A"], self.WHEN, "Sunny")          # duplicates removed
        self.assertEqual(len(result["table"]), 1)

    def test_destination_advice(self):
        result = recommend.recommend(self.ALL, self.WHEN, "Sunny", destination="Central Hospital")
        self.assertTrue(result["notes"][0].startswith("Your destination (Central Hospital)"))
        outside = recommend.recommend(["LOT_A"], self.WHEN, "Sunny", destination="Central Hospital")
        self.assertTrue(any("not among the selected" in n for n in outside["notes"]))

    def test_all_full_gives_no_recommendation(self):
        result = recommend.recommend(self.ALL, self.WHEN, "Sunny", model=StubModel(1.0))
        self.assertIsNone(result["best"])
        self.assertIn("Full", result["headline"])
        self.assertFalse(result["table"]["recommended"].any())

    def test_full_lot_is_never_ranked_above_an_open_one(self):
        values = {}                                    # LOT_A..D rows are predicted in the order given
        model = StubModel([0.99, 0.2, 0.5, 0.97])
        table = recommend.recommend(self.ALL, self.WHEN, "Sunny", model=model)["table"]
        self.assertEqual(list(table["status"].head(2)), ["Available", "Available"])
        self.assertTrue(table.iloc[-2:]["is_full"].all())

    def test_share_ranking_prefers_small_lots(self):
        # 40% full everywhere -> by spaces the biggest lot wins, by share it is a tie broken by id
        model = StubModel(0.4)
        by_spaces = recommend.recommend(self.ALL, self.WHEN, "Sunny", model=model)["table"].iloc[0]["parking_lot_id"]
        self.assertEqual(by_spaces, "LOT_B")                                          # 500 spaces
        model = StubModel([0.9, 0.9, 0.2, 0.9])                                       # hospital has the best share
        by_share = recommend.recommend(self.ALL, self.WHEN, "Sunny", model=model, rank_by="free_share")
        self.assertEqual(by_share["table"].iloc[0]["parking_lot_id"], "LOT_C")

    def test_invalid_arguments(self):
        for kwargs in ({"lot_ids": []}, {"lot_ids": ["LOT_Q"]}, {"lot_ids": ["LOT_A"], "rank_by": "nope"},
                       {"lot_ids": ["LOT_A"], "distance_weight": 1.5}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                recommend.recommend(when=self.WHEN, **kwargs)

    def test_missing_weather_is_handled(self):
        result = recommend.recommend(self.ALL, self.WHEN, None)
        self.assertEqual(len(result["table"]), len(self.ALL))


@unittest.skipUnless(READY, "run `python main.py` first")
class TestDistanceData(unittest.TestCase):
    def write(self, text):
        folder = tempfile.mkdtemp()
        path = Path(folder) / "d.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_no_file_means_no_distance(self):
        table, note = recommend.load_distances(Path("/definitely/not/here.csv"))
        self.assertIsNone(table)
        self.assertIn("no travel-distance", note)

    def test_bad_files_fall_back_gracefully(self):
        self.assertIsNone(recommend.load_distances(self.write("a,b\n1,2\n"))[0])                       # wrong columns
        self.assertIsNone(recommend.load_distances(self.write("destination,parking_lot_id,distance_km\n"))[0])
        junk = self.write("destination,parking_lot_id,distance_km\nNowhere,LOT_A,1\nCentral Hospital,LOT_Z,1\n"
                          "Central Hospital,LOT_A,abc\nCentral Hospital,LOT_A,-2\n")
        self.assertIsNone(recommend.load_distances(junk)[0])

    def test_distance_changes_the_ranking(self):
        rows = ["destination,parking_lot_id,distance_km"]
        km = {"LOT_A": 0.4, "LOT_B": 6.0, "LOT_C": 0.2, "LOT_D": 3.0}
        rows += [f"Central Hospital,{lot},{d}" for lot, d in km.items()] + ["Bad Row,LOT_A,1"]
        distances, note = recommend.load_distances(self.write("\n".join(rows)))
        self.assertIn("1 unusable row", note)
        model = StubModel([0.5, 0.3, 0.5, 0.5])        # LOT_B (far) has by far the most free spaces
        args = dict(lot_ids=list(config.PARKING_LOTS), when="2026-11-04 12:00", weather="Sunny", model=model,
                    destination="Central Hospital", distances=distances)
        availability_only = recommend.recommend(distance_weight=0.0, **args)
        self.assertEqual(availability_only["table"].iloc[0]["parking_lot_id"], "LOT_B")
        self.assertTrue(availability_only["uses_distance"])
        nearest = recommend.recommend(distance_weight=1.0, **args)
        self.assertEqual(nearest["table"].iloc[0]["parking_lot_id"], "LOT_C")
        self.assertIn("km away", nearest["table"].iloc[0]["reason"])
        self.assertNotIn("not consider how far", " ".join(nearest["notes"]))

    def test_incomplete_distances_are_ignored_with_a_message(self):
        distances, _ = recommend.load_distances(self.write(
            "destination,parking_lot_id,distance_km\nCentral Hospital,LOT_A,1\nCentral Hospital,LOT_C,2\n"))
        result = recommend.recommend(list(config.PARKING_LOTS), "2026-11-04 12:00", "Sunny",
                                     destination="Central Hospital", distances=distances)
        self.assertFalse(result["uses_distance"])
        self.assertTrue(any("ignored" in n for n in result["notes"]))

    def test_best_time_to_arrive(self):
        result = recommend.best_time_to_arrive("LOT_B", "2026-11-07 12:00", 12, "Sunny")
        slots = result["slots"]
        self.assertEqual(result["best"]["predicted_available_spaces"], slots["predicted_available_spaces"].max())
        self.assertEqual(result["worst"]["predicted_available_spaces"], slots["predicted_available_spaces"].min())
        full = recommend.best_time_to_arrive("LOT_B", "2026-11-07 12:00", 3, "Sunny")["first_available"]
        self.assertIsNotNone(full)


# ----------------------------------------------------------------------------
@unittest.skipUnless(READY, "run `python main.py` first")
class TestHistoricalAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = dd.load_dashboard_data()

    def test_hour_ranges(self):
        self.assertEqual(analysis.hour_ranges([9, 10, 11, 14]), "09:00-12:00, 14:00-15:00")
        self.assertEqual(analysis.hour_ranges([23]), "23:00-00:00")
        self.assertEqual(analysis.hour_ranges([]), "-")

    def test_peak_hours_match_a_manual_calculation(self):
        peaks = analysis.peak_hours(self.df).set_index("location")
        for location, group in self.df.groupby("location"):
            by_hour = group.groupby("hour")["occupancy_pct"].mean()
            self.assertEqual(peaks.loc[location, "peak_hour"], by_hour.idxmax())
            self.assertAlmostEqual(peaks.loc[location, "peak_occupancy_pct"], by_hour.max())
            self.assertGreaterEqual(peaks.loc[location, "peak_occupancy_pct"], peaks.loc[location, "quietest_occupancy_pct"])
        # sanity against how the simulation was written: the mall peaks in the afternoon/evening
        self.assertIn(peaks.loc["City Center Mall", "peak_hour"], range(14, 20))

    def test_weekday_vs_weekend_directions(self):
        summary = analysis.weekday_weekend_summary(self.df).set_index("location")
        self.assertEqual(summary.loc["Downtown Office Plaza", "busier_on"], "Weekdays")
        self.assertEqual(summary.loc["City Center Mall", "busier_on"], "Weekends")
        self.assertTrue(np.allclose(summary["difference_pts"], summary["weekend_avg_pct"] - summary["weekday_avg_pct"]))

    def test_trends_do_not_invent_changes(self):
        trend = analysis.trend_summary(analysis.daily_trend(self.df))
        self.assertEqual(set(trend["verdict"]), {"No clear trend"})      # the simulation has no trend

        rising = self.df.copy()
        days = (rising["timestamp"].dt.normalize() - rising["timestamp"].min().normalize()).dt.days
        rising["occupancy_pct"] = rising["occupancy_pct"] + days * 0.5   # +3.5 points per week, clearly visible
        verdicts = analysis.trend_summary(analysis.daily_trend(rising)).set_index("location")["verdict"]
        self.assertEqual(set(verdicts), {"Getting busier"})
        short = analysis.trend_summary(analysis.daily_trend(self.df[self.df["timestamp"] < "2026-01-08"]))
        self.assertTrue(short["verdict"].str.contains("Not enough").all())

    def test_weather_effect_is_compared_with_the_typical_hour(self):
        effect = analysis.weather_effect(self.df)
        self.assertFalse((effect["weather"] == "Unknown").any() and (effect["hours"] < 30).any())
        rainy_mall = effect[(effect["location"] == "City Center Mall") & (effect["weather"] == "Rainy")]["vs_typical_pts"].iloc[0]
        rainy_station = effect[(effect["location"] == "Railway Station") & (effect["weather"] == "Rainy")]["vs_typical_pts"].iloc[0]
        self.assertLess(rainy_mall, -2)          # the simulation makes rain keep shoppers away
        self.assertGreater(rainy_station, 2)     # ... and pushes commuters to the station

    def test_event_effect(self):
        effect = analysis.event_effect(self.df)
        event = effect[effect["condition"] == "Special event"]
        self.assertTrue((event["vs_typical_pts"] > 5).all())
        self.assertNotIn("Central Hospital", set(event["location"]))      # no events there


# ----------------------------------------------------------------------------
class TestDataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.features import add_time_features
        from src.data_generator import generate_dataset
        from src.train import time_based_split
        cls.full = add_time_features(generate_dataset())
        cls.train, cls.test, _ = time_based_split(cls.full)

    def failed(self, train, test, full):
        return [c["check"] for c in data_checks.check_split(train, test, full) if not c["ok"]]

    def test_proper_split_passes_everything(self):
        self.assertEqual(self.failed(self.train, self.test, self.full), [])

    def test_overlap_is_detected(self):
        leaky_test = pd.concat([self.test, self.train.tail(100)])
        failed = self.failed(self.train, leaky_test, self.full)
        self.assertTrue(any("strictly after" in c for c in failed))
        self.assertTrue(any("No row appears in both" in c for c in failed))
        with self.assertRaises(ValueError):
            data_checks.assert_clean_split(data_checks.check_split(self.train, leaky_test, self.full))

    def test_random_split_is_detected(self):
        shuffled = self.full.sample(frac=1, random_state=1)
        n = int(len(shuffled) * 0.8)
        failed = self.failed(shuffled.iloc[:n], shuffled.iloc[n:], self.full)
        self.assertTrue(any("strictly after" in c for c in failed))

    def test_near_copy_of_target_is_detected(self):
        train, test, full = self.train.copy(), self.test.copy(), self.full.copy()
        import src.features as features
        original = list(features.NUMERIC_FEATURES)
        features.NUMERIC_FEATURES.append("is_weekend")          # harmless column, then make it a copy of the target
        try:
            for frame in (train, test, full):
                frame["is_weekend"] = frame["occupancy_rate"]
            self.assertTrue(any("near-copy" in c for c in self.failed(train, test, full)))
        finally:
            features.NUMERIC_FEATURES[:] = original

    def test_fingerprint_changes_when_data_changes(self):
        base = data_checks.dataset_fingerprint(self.full)
        self.assertEqual(base, data_checks.dataset_fingerprint(self.full.copy()))
        self.assertNotEqual(base, data_checks.dataset_fingerprint(self.full.iloc[:-1]))
        edited = self.full.copy()
        edited.loc[edited.index[0], "occupied_spaces"] += 1
        self.assertNotEqual(base, data_checks.dataset_fingerprint(edited))

    def test_schema_validation_of_supplied_data(self):
        good = self.full[data_checks.REQUIRED_RAW_COLUMNS].copy()
        self.assertEqual(data_checks.validate_raw_schema(good), [])
        self.assertIn("Missing required column", data_checks.validate_raw_schema(good.drop(columns=["weather"]))[0])
        self.assertTrue(any("500" in p for p in data_checks.validate_raw_schema(good.head(100))))
        unknown = good.copy()
        unknown["parking_lot_id"] = "NEW_LOT"
        self.assertTrue(any("Unknown parking_lot_id" in p for p in data_checks.validate_raw_schema(unknown)))
        broken = good.copy()
        broken["occupied_spaces"] = broken["total_spaces"] + 50
        self.assertTrue(any("above total_spaces" in p for p in data_checks.validate_raw_schema(broken)))
        bad_dates = good.copy()
        bad_dates["timestamp"] = "not a date"
        self.assertTrue(any("timestamps" in p for p in data_checks.validate_raw_schema(bad_dates)))


@unittest.skipUnless(READY, "run `python main.py` first")
class TestConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = dd.load_dashboard_data()
        cls.meta = dd.load_model_metadata()
        cls.model = P.load_model()
        cls.tp = dd.load_test_predictions()

    def test_saved_artifacts_agree(self):
        self.assertEqual(dd.consistency_issues(self.df, self.meta, self.model, self.tp), [])
        self.assertEqual(self.meta["dataset_fingerprint"], data_checks.dataset_fingerprint(self.df))
        self.assertTrue(self.meta["leakage_checks_passed"])
        self.assertEqual(self.meta["features"], FEATURE_COLUMNS)

    def test_changed_dataset_is_reported(self):
        issues = dd.consistency_issues(self.df.iloc[:-4], self.meta, self.model, self.tp)
        self.assertTrue(any(level == "error" and "dataset has changed" in msg for level, msg in issues))

    def test_wrong_capacity_is_reported(self):
        broken = self.df.copy()
        broken.loc[broken["parking_lot_id"] == "LOT_A", "total_spaces"] += 10
        issues = dd.consistency_issues(broken, self.meta, self.model, self.tp)
        self.assertTrue(any("total_spaces" in msg for _, msg in issues))

    def test_missing_or_old_metadata_is_reported(self):
        self.assertTrue(dd.consistency_issues(self.df, {}, self.model, self.tp))
        old = {k: v for k, v in self.meta.items() if k != "dataset_fingerprint"}
        self.assertTrue(any("fingerprints" in msg for _, msg in dd.consistency_issues(self.df, old, self.model, self.tp)))

    def test_training_days_in_test_predictions_are_reported(self):
        tampered = pd.concat([self.tp, self.tp.head(3).assign(timestamp=pd.Timestamp("2026-01-05"))])
        issues = dd.consistency_issues(self.df, self.meta, self.model, tampered)
        self.assertTrue(any("training period" in msg for _, msg in issues))

    def test_saved_test_predictions_are_all_after_training(self):
        allp = dd.load_all_test_predictions()
        self.assertGreater(allp["timestamp"].min(), pd.Timestamp(self.meta["split"]["train_end"]))
        self.assertEqual(set(allp["model"]), set(dd.load_model_comparison()["model"]))

    def test_selected_model_is_chosen_without_the_test_set(self):
        table = dd.load_model_comparison()
        real = table[~table["model"].str.startswith("Baseline")]
        self.assertEqual(self.meta["best_model"], real.sort_values("cv_rmse").iloc[0]["model"])
        self.assertEqual(int(table["selected"].sum()), 1)

    def test_error_bands_were_built_without_the_test_days(self):
        bands = json.loads(config.ERROR_BANDS_PATH.read_text())
        self.assertTrue(all(0.6 <= v <= 0.95 for v in bands["share_of_test_errors_within_p80"].values()))


@unittest.skipUnless(READY, "run `python main.py` first")
class TestFeatureImportance(unittest.TestCase):
    def test_importance_table(self):
        table = dd.load_feature_importance()
        self.assertEqual(set(table["kind"]), {"impurity", "permutation", "permutation_grouped"})
        self.assertEqual(set(table.loc[table["kind"] == "impurity", "model"]),
                         {"Decision Tree", "Random Forest", "Gradient Boosting"})
        for model, group in table[table["kind"] == "impurity"].groupby("model"):
            self.assertAlmostEqual(group["importance"].sum(), 1.0, places=3, msg=model)
        grouped = table[table["kind"] == "permutation_grouped"]
        self.assertEqual(set(grouped["feature"]), {"Time of day", "Day of week", "Parking location",
                                                   "Weather", "Special event"})
        self.assertEqual(grouped.sort_values("importance").iloc[0]["feature"] in {"Weather", "Special event"}, True)


class TestPhase3Charts(unittest.TestCase):
    @unittest.skipUnless(READY, "run `python main.py` first")
    def test_new_charts_build_for_normal_and_empty_input(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src import dashboard_charts as c
        df = dd.load_dashboard_data()
        slots = P.predict_slots(list(config.PARKING_LOTS), "2026-11-07 15:00", 8, "Rainy")
        empty = slots.iloc[0:0]
        made = [c.fig_slot_forecast(slots), c.fig_slot_forecast(empty), c.fig_slot_heat(slots), c.fig_slot_heat(empty),
                c.fig_weekday_weekend(analysis.weekday_weekend_hourly(df, "City Center Mall"), "City Center Mall"),
                c.fig_weekday_weekend(analysis.weekday_weekend_hourly(df, "nowhere"), "nowhere"),
                c.fig_weekday_weekend_bars(analysis.weekday_weekend_summary(df)),
                c.fig_daily_trend(analysis.daily_trend(df)), c.fig_daily_trend(analysis.daily_trend(df), ["nowhere"]),
                c.fig_weather_effect(analysis.weather_effect(df)), c.fig_weather_effect(pd.DataFrame()),
                c.fig_event_effect(analysis.event_effect(df)), c.fig_event_effect(pd.DataFrame()),
                c.fig_model_metric_bars(dd.load_model_comparison(), "Random Forest"),
                c.fig_feature_importance(dd.load_feature_importance().query("kind == 'permutation_grouped'"), "t"),
                c.fig_feature_importance(None, "t")]
        allp = dd.load_all_test_predictions()
        window = allp[(allp["model"] == "Random Forest") & (allp["location"] == "City Center Mall")].head(72)
        made += [c.fig_model_actual_vs_predicted(window, "Random Forest", "City Center Mall"),
                 c.fig_model_actual_vs_predicted(window.iloc[0:0], "Random Forest", "City Center Mall")]
        for fig in made:
            self.assertIsInstance(fig, matplotlib.figure.Figure)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
