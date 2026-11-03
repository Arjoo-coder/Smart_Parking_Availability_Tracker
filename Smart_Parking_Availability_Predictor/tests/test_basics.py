"""Basic sanity tests. Run from the project root with:

    python -m unittest discover -s tests -v

(Run `python main.py` first - a few tests need the trained model and data.)
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_generator import generate_dataset, inject_data_issues
from src.features import FEATURE_COLUMNS, LEAKY_COLUMNS, add_time_features
from src.preprocessing import clean_data
from src.train import time_based_split


class TestSimulatedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_dataset()

    def test_has_enough_rows_and_columns(self):
        self.assertGreaterEqual(len(self.df), 2000)
        required = {"timestamp", "parking_lot_id", "location", "total_spaces",
                    "occupied_spaces", "available_spaces", "hour", "day_of_week",
                    "is_weekend", "weather", "special_event"}
        self.assertTrue(required.issubset(self.df.columns))

    def test_numbers_are_consistent(self):
        d = self.df
        self.assertTrue((d["occupied_spaces"] + d["available_spaces"] == d["total_spaces"]).all())
        self.assertTrue(d["occupied_spaces"].between(0, d["total_spaces"].max()).all())
        self.assertTrue((d["occupied_spaces"] <= d["total_spaces"]).all())

    def test_generation_is_reproducible(self):
        pd.testing.assert_frame_equal(self.df, generate_dataset())


class TestCleaning(unittest.TestCase):
    def test_cleaning_removes_injected_problems(self):
        raw = inject_data_issues(generate_dataset())
        self.assertGreater(len(raw), len(generate_dataset()))      # duplicates were added
        self.assertGreater(raw["weather"].isna().sum(), 0)         # missing values were added
        clean, report = clean_data(raw)
        self.assertEqual(report["duplicate_rows_removed"], 30)
        self.assertEqual(int(clean["weather"].isna().sum()), 0)
        self.assertEqual(len(clean), len(generate_dataset()))
        self.assertFalse(clean.duplicated(subset=["timestamp", "parking_lot_id"]).any())


class TestNoDataLeakage(unittest.TestCase):
    def test_features_do_not_contain_the_answer(self):
        for column in LEAKY_COLUMNS:
            self.assertNotIn(column, FEATURE_COLUMNS)
        self.assertNotIn("timestamp", FEATURE_COLUMNS)

    def test_time_split_has_no_overlap(self):
        df = add_time_features(generate_dataset())
        train, test, cutoff = time_based_split(df)
        self.assertLess(train["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(len(train) + len(test), len(df))
        self.assertAlmostEqual(len(test) / len(df), config.TEST_FRACTION, delta=0.01)


@unittest.skipUnless(config.MODEL_PATH.exists(), "trained model not found - run main.py first")
class TestPrediction(unittest.TestCase):
    def test_single_prediction_is_valid(self):
        from src.predict import predict_occupancy
        r = predict_occupancy("LOT_A", "2026-11-04 11:00", "Sunny", False)
        self.assertTrue(0 <= r["predicted_occupancy_rate"] <= 1)
        self.assertEqual(r["predicted_occupied_spaces"] + r["predicted_available_spaces"],
                         r["total_spaces"])

    def test_full_day_has_24_rows(self):
        from src.predict import predict_full_day
        day = predict_full_day("LOT_B", "2026-11-07", "Rainy", True)
        self.assertEqual(len(day), 24)

    def test_office_is_busier_on_weekday_noon_than_sunday_noon(self):
        from src.predict import predict_occupancy
        weekday = predict_occupancy("LOT_A", "2026-11-04 12:00")      # Wednesday
        sunday = predict_occupancy("LOT_A", "2026-11-08 12:00")       # Sunday
        self.assertGreater(weekday["predicted_occupancy_rate"], sunday["predicted_occupancy_rate"])

    def test_bad_input_is_rejected(self):
        from src.predict import predict_occupancy
        with self.assertRaises(ValueError):
            predict_occupancy("LOT_Z", "2026-11-04 12:00")
        with self.assertRaises(ValueError):
            predict_occupancy("LOT_A", "2026-11-04 12:00", weather="Snowy")


if __name__ == "__main__":
    unittest.main()
