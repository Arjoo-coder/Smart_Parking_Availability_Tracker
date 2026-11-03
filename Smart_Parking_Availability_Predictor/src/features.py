"""Feature engineering - shared by training AND prediction.

Keeping this in one file guarantees that the model sees exactly the same kind
of features when it is trained and when it is used in the app.

Data-leakage guard
------------------
The target is `occupancy_rate`. The columns occupied_spaces, available_spaces
and occupancy_rate are all different views of the SAME number, so they must
NEVER be used as inputs. A model that sees them would simply copy the answer
and show fake, perfect accuracy. The assertion below makes sure this can't
happen by accident.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "occupancy_rate"

# Columns that would leak the answer -> never allowed as inputs.
LEAKY_COLUMNS = ["occupied_spaces", "available_spaces", "occupancy_rate"]

# Text columns -> converted to 0/1 columns (one-hot encoding)
CATEGORICAL_FEATURES = ["parking_lot_id", "weather"]

# Number columns
NUMERIC_FEATURES = [
    "hour", "day_of_week", "is_weekend", "special_event", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Safety check: fail loudly if a leaky column ever sneaks into the feature list.
assert not set(LEAKY_COLUMNS) & set(FEATURE_COLUMNS), "Data leakage: leaky column in features!"


def add_time_features(df):
    """Create time-based features from the `timestamp` column.

    Cyclic features (sin/cos): hour 23 and hour 0 are close together in real
    life, but far apart as plain numbers (23 vs 0). Sine/cosine "wrap" the
    hour (and the weekday) around a circle so the model understands that.
    """
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek            # Monday=0 ... Sunday=6
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def get_xy(df):
    """Split a processed DataFrame into inputs X and target y."""
    return df[FEATURE_COLUMNS], df[TARGET]


def build_preprocessor(scale_numeric=False):
    """Build the column transformer used inside every model pipeline.

    - numeric columns: scaled (only for Linear Regression, which needs it)
    - categorical columns: one-hot encoded
    The encoder/scaler are fitted inside the pipeline on TRAINING data only,
    so no information from the test set leaks in.
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", numeric_step, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
