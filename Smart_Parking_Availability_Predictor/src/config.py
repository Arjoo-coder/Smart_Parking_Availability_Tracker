"""Central configuration for the Smart Parking Availability Predictor.

Everything that is shared between several scripts (folder paths, the random
seed, the parking-lot definitions ...) lives here, so there is exactly one
place to change it.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Folder and file paths (all relative to the project root folder)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

# NOTE: the word "simulated" is in the file name on purpose - this is NOT real data.
RAW_DATA_PATH = DATA_RAW_DIR / "parking_data_simulated.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "parking_data_processed.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
DATA_SOURCE_PATH = DATA_PROCESSED_DIR / "data_source.json"     # {"is_simulated": true/false, ...}

MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
MODEL_METADATA_PATH = RESULTS_DIR / "model_metadata.json"

# Phase 3 artifacts (all written by train.py)
ALL_TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions_all_models.csv"   # every model, test days only
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
ERROR_BANDS_PATH = RESULTS_DIR / "error_bands.json"                           # typical error per lot

# OPTIONAL user-supplied file with travel distances (not part of the simulated data).
# Columns: destination, parking_lot_id, distance_km   - see README "Distance data".
DISTANCE_FILE = DATA_RAW_DIR / "lot_distances.csv"

# ----------------------------------------------------------------------------
# Simulation settings
# ----------------------------------------------------------------------------
RANDOM_SEED = 42          # same seed -> same dataset every time (reproducible)
START_DATE = "2026-01-01"
NUM_DAYS = 90             # 90 days x 24 hours x 4 lots = 8,640 hourly records

# ----------------------------------------------------------------------------
# Modelling settings
# ----------------------------------------------------------------------------
TEST_FRACTION = 0.20      # the LAST 20% of the time range is used for testing
N_CV_SPLITS = 5           # time-series cross-validation folds (training data only)
TUNE_MODELS = True        # small hyper-parameter search (on training data only); main.py --no-tune turns it off

# Forecasting
MAX_SLOT_HOURS = 48       # most hourly slots the app will forecast in one go

# ----------------------------------------------------------------------------
# Domain settings
# ----------------------------------------------------------------------------
WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy", "Foggy"]

# A "special event" is only active in the evening on event days.
# special_event = 1 only for rows between these hours (inclusive) on event days.
EVENT_START_HOUR = 17
EVENT_END_HOUR = 22

# Occupancy thresholds for the status label shown in the app:
#   occupancy below STATUS_LIMITED_FROM   -> "Available"
#   from STATUS_LIMITED_FROM up to 95%    -> "Limited"
#   from STATUS_FULL_FROM upwards         -> "Full"
STATUS_LIMITED_FROM = 0.70
STATUS_FULL_FROM = 0.95

# The four simulated parking lots.
#   type        -> decides the daily pattern (see data_generator.py)
#   event_prob  -> chance that a given day has a special event near this lot
#   event_boost -> how much an event increases occupancy (0.20 = +20 points)
PARKING_LOTS = {
    "LOT_A": {"location": "Downtown Office Plaza", "total_spaces": 300,
              "type": "office", "event_prob": 0.06, "event_boost": 0.20},
    "LOT_B": {"location": "City Center Mall", "total_spaces": 500,
              "type": "mall", "event_prob": 0.12, "event_boost": 0.25},
    "LOT_C": {"location": "Central Hospital", "total_spaces": 250,
              "type": "hospital", "event_prob": 0.00, "event_boost": 0.00},
    "LOT_D": {"location": "Railway Station", "total_spaces": 200,
              "type": "station", "event_prob": 0.05, "event_boost": 0.15},
}


def ensure_directories():
    """Create all output folders if they do not exist yet."""
    for folder in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, CHARTS_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
