"""A small container that carries the loaded data to every dashboard page."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd


@dataclass
class AppContext:
    df: pd.DataFrame                              # processed dataset (all lots, all hours)
    snapshot_ts: pd.Timestamp                     # the moment shown on Home / Availability
    snapshot: pd.DataFrame                        # one row per lot at snapshot_ts
    previous_snapshot: Optional[pd.DataFrame]     # the hour before (for the +/- arrows)
    test_predictions: Optional[pd.DataFrame]      # hold-out predictions from train.py
    model_comparison: Optional[pd.DataFrame]      # metrics table from train.py
    metadata: dict                                # best model name, split dates ...
    per_lot_error: dict                           # {location: average error in spaces}
    get_model: Callable                           # loads the saved model (cached)
    # ---- Phase 3 (all optional, so older code and tests keep working) -------------
    is_simulated: bool = True                     # from data/processed/data_source.json
    all_predictions: Optional[pd.DataFrame] = None   # every model's predictions on the test days
    importance: Optional[pd.DataFrame] = None     # feature importance table from train.py
    issues: Optional[List[Tuple[str, str]]] = None   # consistency problems [(level, message)]
