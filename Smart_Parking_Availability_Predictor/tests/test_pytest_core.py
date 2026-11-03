"""pytest suite for the final checks.

    pip install -r requirements.txt
    python -m pytest                 # runs every tests/test_*.py file (about 30 s)

The first group of tests runs the WHOLE pipeline (generate -> clean -> train -> predict)
in a temporary folder with a smaller dataset, so it never touches your real data or model.
No expected accuracy numbers are hardcoded: the tests only check relationships that must
always hold (for example "the trained model beats the baseline", "free spaces are within capacity").
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data_checks, data_generator, preprocessing, recommend, train
from src import dashboard_data as dd
from src import predict as P
from src.features import FEATURE_COLUMNS, LEAKY_COLUMNS

SMALL_DAYS = 28            # a 4-week dataset keeps the pipeline test quick (4 lots x 24 h x 28 d = 2,688 rows)


class StubModel:
    """Returns whatever numbers it is given, to test the safety limits."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


# ============================================================================
# 1. The complete workflow in a temporary folder
# ============================================================================
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Generate data, clean it, train models and load the result - all inside a temp folder."""
    base = tmp_path_factory.mktemp("pipeline")
    paths = {
        "DATA_RAW_DIR": base / "data" / "raw", "DATA_PROCESSED_DIR": base / "data" / "processed",
        "MODELS_DIR": base / "models", "CHARTS_DIR": base / "outputs" / "charts",
        "RESULTS_DIR": base / "outputs" / "results",
    }
    paths.update({
        "RAW_DATA_PATH": paths["DATA_RAW_DIR"] / "parking_data_simulated.csv",
        "PROCESSED_DATA_PATH": paths["DATA_PROCESSED_DIR"] / "parking_data_processed.csv",
        "DATA_SOURCE_PATH": paths["DATA_PROCESSED_DIR"] / "data_source.json",
        "MODEL_PATH": paths["MODELS_DIR"] / "best_model.joblib",
        "MODEL_COMPARISON_PATH": paths["RESULTS_DIR"] / "model_comparison.csv",
        "MODEL_METADATA_PATH": paths["RESULTS_DIR"] / "model_metadata.json",
        "ALL_TEST_PREDICTIONS_PATH": paths["RESULTS_DIR"] / "test_predictions_all_models.csv",
        "FEATURE_IMPORTANCE_PATH": paths["RESULTS_DIR"] / "feature_importance.csv",
        "ERROR_BANDS_PATH": paths["RESULTS_DIR"] / "error_bands.json",
        "DISTANCE_FILE": paths["DATA_RAW_DIR"] / "lot_distances.csv",
        "NUM_DAYS": SMALL_DAYS,
    })
    with pytest.MonkeyPatch.context() as mp:
        for name, value in paths.items():
            mp.setattr(config, name, value)
        raw = data_generator.generate_and_save()
        processed = preprocessing.run_preprocessing(is_simulated=True)
        results, best_name = train.run_training(tune=False)
        model = joblib.load(config.MODEL_PATH)
        yield {"raw": raw, "processed": processed, "results": results, "best": best_name,
               "model": model, "paths": paths}


def test_dataset_generation(run):
    raw = run["raw"]
    assert len(raw) >= 2000
    required = {"timestamp", "parking_lot_id", "location", "total_spaces", "occupied_spaces",
                "available_spaces", "hour", "day_of_week", "is_weekend", "weather", "special_event"}
    assert required <= set(raw.columns)
    assert run["paths"]["RAW_DATA_PATH"].name == "parking_data_simulated.csv"      # labelled as simulated
    assert run["raw"]["parking_lot_id"].nunique() == len(config.PARKING_LOTS)


def test_dataset_is_consistent_and_reproducible(run):
    clean = data_generator.generate_dataset()            # NUM_DAYS is patched only inside the fixture
    again = pd.DataFrame(run["processed"])
    assert (again["occupied_spaces"] + again["available_spaces"] == again["total_spaces"]).all()
    assert again["occupied_spaces"].between(0, again["total_spaces"]).all()
    assert again["occupancy_rate"].between(0, 1).all()
    assert clean.shape[1] >= 11


def test_cleaning_removed_the_injected_problems(run):
    processed = run["processed"]
    assert processed["weather"].isna().sum() == 0
    assert not processed.duplicated(subset=["timestamp", "parking_lot_id"]).any()
    assert len(processed) < len(run["raw"])              # the 30 duplicates were dropped
    for column in FEATURE_COLUMNS:
        assert column in processed.columns
        assert processed[column].notna().all()


def test_training_wrote_every_artifact(run):
    for name in ("MODEL_PATH", "MODEL_COMPARISON_PATH", "MODEL_METADATA_PATH", "ALL_TEST_PREDICTIONS_PATH",
                 "FEATURE_IMPORTANCE_PATH", "ERROR_BANDS_PATH", "PROCESSED_DATA_PATH"):
        assert run["paths"][name].exists(), name
    results_dir = run["paths"]["RESULTS_DIR"]
    for filename in ("test_predictions.csv", "leakage_checks.json", "per_lot_test_error.csv", "metrics_explained.txt"):
        assert (results_dir / filename).exists(), filename
    assert len(list(run["paths"]["CHARTS_DIR"].glob("model_*.png"))) >= 4


def test_training_is_honest(run):
    results = run["results"].set_index("model")
    baseline = results.loc[train.BASELINE_NAME]
    best = results.loc[run["best"]]
    assert run["best"] != train.BASELINE_NAME
    assert best["test_rmse"] < baseline["test_rmse"]                 # the model must beat "always predict the average"
    assert best["test_mae"] < baseline["test_mae"]
    assert best["test_r2"] > baseline["test_r2"]
    assert np.isfinite(results[["test_mae", "test_rmse", "test_r2", "cv_rmse"]].to_numpy()).all()
    assert (results["test_rmse"] >= results["test_mae"] - 1e-12).all()     # RMSE is never below MAE
    assert len(results) >= 4                                          # at least three real models + baseline


def test_model_chosen_without_the_test_set(run):
    meta = json.loads(run["paths"]["MODEL_METADATA_PATH"].read_text())
    real = run["results"][run["results"]["model"] != train.BASELINE_NAME]
    assert meta["best_model"] == real.sort_values("cv_rmse").iloc[0]["model"]
    assert meta["data_is_simulated"] is True
    assert meta["leakage_checks_passed"] is True
    assert meta["features"] == FEATURE_COLUMNS


def test_no_leakage_in_the_split(run):
    train_df, test_df, _ = train.time_based_split(run["processed"])
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
    assert abs(len(test_df) / len(run["processed"]) - config.TEST_FRACTION) < 0.02
    assert not set(LEAKY_COLUMNS) & set(FEATURE_COLUMNS)
    checks = json.loads((run["paths"]["RESULTS_DIR"] / "leakage_checks.json").read_text())
    assert checks and all(c["ok"] for c in checks)
    saved = pd.read_csv(run["paths"]["ALL_TEST_PREDICTIONS_PATH"], parse_dates=["timestamp"])
    assert saved["timestamp"].min() > train_df["timestamp"].max()     # only unseen days were saved for the plots


def test_saved_model_dataset_and_metadata_agree(run):
    df = dd.load_dashboard_data(run["paths"]["PROCESSED_DATA_PATH"])
    meta = json.loads(run["paths"]["MODEL_METADATA_PATH"].read_text())
    assert dd.consistency_issues(df, meta, run["model"], dd.load_test_predictions()) == []


def test_trained_model_predicts_within_capacity_for_every_situation(run):
    requests = [{"parking_lot_id": lot, "timestamp": pd.Timestamp("2026-11-07") + pd.Timedelta(hours=h),
                 "weather": weather, "event_today": True}
                for lot in config.PARKING_LOTS for h in range(24) for weather in config.WEATHER_OPTIONS + [None]]
    result = P.predict_many(requests, model=run["model"])
    assert len(result) == len(requests)
    assert result["predicted_available_spaces"].between(0, result["total_spaces"]).all()
    assert result["predicted_occupancy_rate"].between(0, 1).all()
    assert (result["predicted_available_spaces"] + result["predicted_occupied_spaces"] == result["total_spaces"]).all()
    assert set(result["status"]) <= {"Available", "Limited", "Full"}


def test_the_model_learned_the_daily_pattern(run):
    """A trained model must not give the same answer for every hour (that would be a dead model)."""
    day = P.predict_full_day("LOT_A", "2026-11-04", "Sunny", False, model=run["model"])
    assert day["predicted_available_spaces"].nunique() > 5
    noon = day.loc[day["timestamp"].dt.hour == 12, "predicted_available_spaces"].iloc[0]
    night = day.loc[day["timestamp"].dt.hour == 3, "predicted_available_spaces"].iloc[0]
    assert noon < night                                  # the office lot is fuller at noon than at 03:00 on a weekday


def test_sample_prediction_workflow(run):
    """The workflow shown in the README: choose inputs -> predict -> read the result."""
    result = P.predict_occupancy("LOT_B", "2026-11-07 15:00", "Rainy", False, model=run["model"])
    for key in ("timestamp", "parking_lot_id", "location", "total_spaces", "predicted_occupancy_rate",
                "predicted_occupied_spaces", "predicted_available_spaces", "status"):
        assert key in result
    assert result["location"] == config.PARKING_LOTS["LOT_B"]["location"]
    assert result["total_spaces"] == config.PARKING_LOTS["LOT_B"]["total_spaces"]
    slots = P.predict_slots(["LOT_A", "LOT_B"], "2026-11-07 15:00", 6, "Rainy", model=run["model"])
    assert len(slots) == 12 and slots["slot"].max() == 5
    ranked = recommend.recommend(list(config.PARKING_LOTS), "2026-11-07 15:00", "Rainy", model=run["model"])
    assert ranked["table"]["rank"].tolist() == list(range(1, len(config.PARKING_LOTS) + 1))
    assert not ranked["uses_distance"]


def test_supplied_data_is_marked_as_not_simulated(run, tmp_path, monkeypatch):
    """--data path: a valid file is accepted and the project then records that it is NOT simulated."""
    import main
    good = tmp_path / "mine.csv"
    run["raw"].to_csv(good, index=False)
    monkeypatch.setattr(main, "SUPPLIED_DATA_PATH", tmp_path / "copy.csv")
    accepted = main.use_supplied_data(good)
    assert accepted.exists()
    out = tmp_path / "processed.csv"
    monkeypatch.setattr(config, "DATA_SOURCE_PATH", tmp_path / "source.json")
    preprocessing.run_preprocessing(raw_path=accepted, out_path=out, is_simulated=False)
    assert json.loads((tmp_path / "source.json").read_text())["is_simulated"] is False
    assert dd.load_data_source()["is_simulated"] is False


# ============================================================================
# 2. Prediction boundaries and invalid input (no trained model needed)
# ============================================================================
@pytest.mark.parametrize("value", [-100.0, -0.5, -1e-9, 0.0, 0.5, 1.0, 1.0000001, 3.0, 1e12])
def test_extreme_model_output_never_leaves_the_capacity(value):
    requests = [{"parking_lot_id": lot, "timestamp": "2026-11-04 12:00"} for lot in config.PARKING_LOTS]
    result = P.predict_many(requests, model=StubModel(value))
    assert result["predicted_available_spaces"].between(0, result["total_spaces"]).all()
    assert result["predicted_occupied_spaces"].between(0, result["total_spaces"]).all()
    assert (result["predicted_available_spaces"] + result["predicted_occupied_spaces"] == result["total_spaces"]).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_numeric_model_output_is_rejected(bad):
    with pytest.raises(ValueError):
        P.predict_many([{"parking_lot_id": "LOT_A", "timestamp": "2026-11-04 12:00"}], model=StubModel(bad))


@pytest.mark.parametrize("rate,expected", [(0.0, "Available"), (0.69, "Available"), (0.70, "Limited"),
                                           (0.94, "Limited"), (0.95, "Full"), (1.0, "Full")])
def test_status_labels(rate, expected):
    assert P.status_label(rate) == expected


@pytest.mark.parametrize("request_,message", [
    ({"parking_lot_id": None, "timestamp": "2026-11-04"}, "No parking location"),
    ({"parking_lot_id": "", "timestamp": "2026-11-04"}, "No parking location"),
    ({"parking_lot_id": "LOT_Z", "timestamp": "2026-11-04"}, "Unknown parking lot"),
    ({"parking_lot_id": "LOT_A", "timestamp": None}, "No date and time"),
    ({"parking_lot_id": "LOT_A", "timestamp": "not a date"}, "could not be understood"),
    ({"parking_lot_id": "LOT_A", "timestamp": "2026-11-04", "weather": "Snowy"}, "Unknown weather"),
    ({"parking_lot_id": "LOT_A", "timestamp": "2026-11-04", "event_today": "perhaps"}, "event_today"),
])
def test_invalid_requests_give_clear_errors(request_, message):
    with pytest.raises(ValueError, match=message):
        P.normalize_request(request_)


@pytest.mark.parametrize("weather", [None, "", "  ", float("nan"), "Not sure", "unknown"])
def test_missing_weather_becomes_unknown(weather):
    clean, _ = P.normalize_request({"parking_lot_id": "LOT_A", "timestamp": "2026-11-04 10:00", "weather": weather})
    assert clean["weather"] == "Unknown"


@pytest.mark.parametrize("hours", [0, -1, config.MAX_SLOT_HOURS + 1, "x", None])
def test_invalid_slot_counts(hours):
    with pytest.raises(ValueError):
        P.predict_slots("LOT_A", "2026-11-07 10:00", hours, model=StubModel(0.5))


def test_slot_forecast_with_stub_model():
    slots = P.predict_slots(list(config.PARKING_LOTS), "2026-11-07 10:40", 5, None, model=StubModel(0.5))
    assert len(slots) == 5 * len(config.PARKING_LOTS)
    assert slots["timestamp"].min() == pd.Timestamp("2026-11-07 10:00")
    assert (slots["predicted_occupancy_rate"] == 0.5).all()


def test_recommendation_logic_with_stub_model():
    when = "2026-11-04 12:00"
    everything_full = recommend.recommend(list(config.PARKING_LOTS), when, "Sunny", model=StubModel(1.0))
    assert everything_full["best"] is None and not everything_full["table"]["recommended"].any()
    with pytest.raises(ValueError):
        recommend.recommend([], when)
    with pytest.raises(ValueError):
        recommend.recommend(["LOT_Q"], when)
    note = " ".join(recommend.recommend(["LOT_A"], when, "Sunny", model=StubModel(0.2))["notes"])
    assert "no travel-distance" in note and "estimates" in note        # honest about what is NOT known


def test_leakage_checks_catch_a_bad_split():
    full = preprocessing.add_time_features(data_generator.generate_dataset())
    shuffled = full.sample(frac=1, random_state=0)
    cut = int(len(shuffled) * 0.8)
    failed = [c["check"] for c in data_checks.check_split(shuffled.iloc[:cut], shuffled.iloc[cut:], full) if not c["ok"]]
    assert any("strictly after" in c for c in failed)


def test_supplied_data_validation_rejects_unusable_files(tmp_path):
    import main
    df = data_generator.generate_dataset()
    cases = {"missing_column.csv": df.drop(columns=["weather"]), "too_small.csv": df.head(50)}
    for name, frame in cases.items():
        path = tmp_path / name
        frame.to_csv(path, index=False)
        with pytest.raises(SystemExit):
            main.use_supplied_data(path)
    with pytest.raises(SystemExit):
        main.use_supplied_data(tmp_path / "does_not_exist.csv")


# ============================================================================
# 3. The project package itself: required files, documented commands, no secrets
# ============================================================================
REQUIRED_FILES = [
    "README.md", "requirements.txt", ".gitignore", "pytest.ini", "main.py", "generate_dataset.py",
    "train_model.py", ".streamlit/config.toml", "app/streamlit_app.py", "app/ui.py", "app/context.py",
    "app/views/home.py", "app/views/availability.py", "app/views/recommendations.py", "app/views/prediction.py",
    "app/views/analytics.py", "app/views/history.py", "app/views/evaluation.py", "app/views/about.py",
    "src/config.py", "src/data_generator.py", "src/preprocessing.py", "src/features.py", "src/eda.py",
    "src/train.py", "src/evaluation.py", "src/predict.py", "src/recommend.py", "src/analysis.py",
    "src/data_checks.py", "src/dashboard_data.py", "src/dashboard_charts.py", "data/README.md",
]
SAVED_FILES = [
    "data/raw/parking_data_simulated.csv", "data/processed/parking_data_processed.csv",
    "data/processed/data_source.json", "models/best_model.joblib", "outputs/results/model_comparison.csv",
    "outputs/results/model_metadata.json", "outputs/results/test_predictions.csv",
    "outputs/results/feature_importance.csv", "outputs/results/error_bands.json",
]


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_source_and_config_files_exist(relative):
    assert (ROOT / relative).is_file(), f"missing {relative}"


@pytest.mark.parametrize("relative", SAVED_FILES)
def test_shipped_dataset_model_and_results_exist(relative):
    assert (ROOT / relative).is_file(), f"missing {relative} (run: python main.py)"


def test_requirements_cover_every_imported_library():
    text = (ROOT / "requirements.txt").read_text().lower()
    for package in ("pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "seaborn", "streamlit", "joblib", "pytest"):
        assert package in text, package


def test_every_documented_command_points_at_a_real_file():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scripts = set(re.findall(r"python (\S+\.py)", readme))
    scripts |= set(re.findall(r"streamlit run (\S+\.py)", readme))
    assert {"main.py", "generate_dataset.py", "train_model.py"} <= scripts
    for script in scripts:
        assert (ROOT / script).is_file(), f"README mentions {script} but it does not exist"
    for module in set(re.findall(r"python -m src\.(\w+)", readme)):
        assert (ROOT / "src" / f"{module}.py").is_file(), module


def test_readme_is_honest_about_the_data():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "simulated" in readme
    assert "no live" in readme or "not live" in readme or "no real-time" in readme
    assert "gps" in readme and ("no gps" in readme or "not use gps" in readme or "without gps" in readme)
    for section in ("problem statement", "objectives", "proposed solution", "features", "technology stack",
                    "folder structure", "dataset description", "evaluation metrics", "installation",
                    "sample prediction workflow", "limitations", "future scope"):
        assert section in readme, f"README is missing the section '{section}'"


def test_no_secrets_or_paid_services_are_needed():
    forbidden = ("api_key", "apikey", "secret_key", "openai", "googlemaps", "mapbox", "twilio", "requests.get(", "urlopen(")
    for folder in ("src", "app"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for word in forbidden:
                assert word not in text, f"{path.relative_to(ROOT)} contains '{word}'"
    assert not (ROOT / ".streamlit" / "secrets.toml").exists()


def test_gitignore_covers_environments_and_caches():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    for entry in ("venv/", ".venv/", "__pycache__/", "*.pyc", ".pytest_cache/"):
        assert entry in lines, entry


def test_wrapper_scripts_expose_main():
    import generate_dataset
    import train_model
    assert callable(generate_dataset.main) and callable(train_model.main)
