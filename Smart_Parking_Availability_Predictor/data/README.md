# Data folder

**All data in this project is SIMULATED (synthetic). It is not real parking data.**

| Folder | Content |
|---|---|
| `raw/parking_data_simulated.csv` | Output of `src/data_generator.py`, including a few deliberate problems (about 1% missing weather values and 30 duplicated rows) so the cleaning step has real work to do. |
| `processed/parking_data_processed.csv` | Cleaned data plus engineered features (cyclic hour/weekday, rush-hour flag ...). This is what the models train on. |

## Columns

| Column | Meaning |
|---|---|
| `timestamp` | Hourly reading time |
| `parking_lot_id`, `location` | Which of the 4 simulated lots (`LOT_A` ... `LOT_D`) |
| `total_spaces` | Capacity of the lot |
| `occupied_spaces` | Cars parked at that hour |
| `available_spaces` | `total_spaces - occupied_spaces` |
| `occupancy_rate` | `occupied_spaces / total_spaces` (0-1) - the value the models predict |
| `hour`, `day_of_week`, `is_weekend` | Calendar features (Monday = 0 ... Sunday = 6) |
| `weather` | `Sunny`, `Cloudy`, `Rainy`, `Foggy` (`Unknown` = missing in the raw data) |
| `special_event` | 1 only during 17:00-22:00 on event days near that lot |

Recreate the data any time with `python main.py` (same seed = same data).

## Phase 3 additions

| File | Meaning |
|---|---|
| `raw/parking_data_supplied.csv` | Created only when you run `python main.py --data your_file.csv` (genuine data). Not present by default. |
| `raw/lot_distances.csv` | OPTIONAL. Travel distances so recommendations can consider distance: columns `destination, parking_lot_id, distance_km`. Not present by default, so recommendations use availability only. |
| `processed/data_source.json` | Says whether the processed data is simulated (`"is_simulated": true`) or supplied. Read by the model metadata and the dashboard labels. |

The simulated file contains **no** distances, coordinates or travel times.
