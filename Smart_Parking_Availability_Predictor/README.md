# Smart Parking Availability Predictor

A beginner-friendly machine learning project that predicts **how full a parking lot will be**, **how many spaces will be free**, and **which lot is the best choice**, using the time of day, weekday, weather and special events. It ends in an interactive Streamlit dashboard.

> **ABOUT THE DATA: SIMULATED UNLESS YOU SUPPLY YOUR OWN.**
> By default every number in this project comes from a **simulation** (daily patterns + weather + events + random noise), not from real parking sensors. Scores show that the machine-learning pipeline works; they are **not** a claim about accuracy on a real car park. Genuine data is used only if you supply it with `python main.py --data your_file.csv`. The dashboard labels the data source on every page.
>
> **Predictions are estimates, not real-time availability.** There is **no live sensor feed, no GPS or map integration, and no paid API**. The project needs no internet connection and no API keys or secrets once the libraries are installed.

**Status:** all phases complete (data and models, dashboard, recommendations and evaluation, final testing and packaging).

---

## Contents
1. [Project overview and problem statement](#project-overview-and-problem-statement)
2. [Objectives and proposed solution](#objectives-and-proposed-solution)
3. [Features and technology stack](#features-and-technology-stack)
4. [Installation and how to run](#installation-and-how-to-run)
5. [Sample prediction workflow](#sample-prediction-workflow)
6. [Dataset description](#dataset-description)
7. [ML algorithms and evaluation metrics](#ml-algorithms-and-evaluation-metrics)
8. [Folder structure](#folder-structure)
9. [The dashboard](#the-dashboard)
10. [Recommendations and distance data](#recommendations-and-distance-data)
11. [Using genuine parking data](#using-genuine-parking-data)
12. [Avoiding leakage and fake accuracy](#avoiding-data-leakage-and-fake-accuracy)
13. [Robustness](#robustness)
14. [Tests](#tests)
15. [Limitations](#limitations)
16. [Future scope](#future-scope)
17. [Troubleshooting](#troubleshooting)

## Project overview and problem statement
Drivers lose time and fuel circling full car parks, and operators have little warning of when a lot will fill up. Most people only learn that a lot is full **after** they arrive.

This project builds a small end-to-end system that answers, **before** the trip: *"How many spaces will this lot probably have at 18:00 on Saturday, and which nearby lot is the better choice?"*

## Objectives and proposed solution
**Objectives**
- Build a complete, reproducible ML pipeline: data, cleaning, features, exploration, several models, honest evaluation, saved model.
- Predict future **occupancy rate** and convert it into free spaces and an Available / Limited / Full status.
- Recommend a parking location and a good arrival time.
- Provide a dashboard a non-programmer can use.
- Avoid data leakage and avoid claiming accuracy the project has not earned.

**Proposed solution.** A regression model learns the occupancy rate of each lot from calendar, weather and event features. Because the target is a *rate* (0 to 1), one model serves lots of different sizes; free spaces = `total_spaces x (1 - rate)`, always limited to between 0 and the lot's capacity. Several algorithms are compared with time-ordered cross-validation and the best is saved with Joblib. The Streamlit app loads that saved model and runs exactly the same preprocessing at prediction time.

## Features and technology stack
| Need | Where in the dashboard |
|---|---|
| "How full are the lots?" | Home, Parking Availability |
| "Where should I park?" (optional destination and preferred lots) | Smart Recommendations |
| "How many spaces will be free at 18:00 on Saturday? And over the next 12 hours?" | Predict Availability (single time and hourly slots) |
| "When is it busiest? Weekday vs weekend? Does rain matter?" | Historical Analysis |
| "Is the model any good? Can the scores be trusted?" | Model Evaluation |

Other features: input validation with clear messages, capacity limits, automatic leakage checks, simulated/genuine data labelling, optional distance file, command-line prediction, and an automated test suite.

**Technology stack:** Python 3.9+, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib, pytest. Data are plain CSV files. Everything is free and open source.

## Installation and how to run
Requires Python 3.9 or newer. Run all commands from the project folder.

```bash
cd Smart_Parking_Availability_Predictor

python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
```

**Quick start (the ZIP already contains data and a trained model):**

```bash
python train_model.py                   # re-train on your computer (recommended, about 1 minute)
streamlit run app/streamlit_app.py      # open the dashboard in your browser
```

Re-training matters because a saved model made with one scikit-learn version may not load in another. If you skip it and the dashboard reports a model problem, just run `python train_model.py`.

**All commands**

| Command | What it does |
|---|---|
| `python generate_dataset.py` | Regenerate the SIMULATED dataset and clean it. Only needed if you want fresh data. |
| `python train_model.py` | Tune and compare the models on the saved dataset, save the best one and all reports. Add `--no-tune` for a faster run. |
| `python main.py` | The whole pipeline in one go: simulate, clean, explore, train, save. |
| `streamlit run app/streamlit_app.py` | Start the dashboard. (The entry file is `app/streamlit_app.py`, not `app.py`.) |
| `python -m pytest` | Run the automated tests. |

```bash
python main.py --train-only             # reuse the saved dataset, only re-train
python main.py --no-tune                # skip the hyper-parameter search (faster)
python main.py --skip-eda               # skip the exploratory charts
python main.py --data my_parking.csv    # use GENUINE data instead of the simulation
python tests/smoke_test_pages.py        # opens every dashboard page with many inputs
```

## Sample prediction workflow
**A. In the dashboard.** Open **Predict Availability** and choose a location, date, time, weather and whether an event is on. Press the predict control. You get the predicted free spaces, occupancy %, a status badge and a typical-error range, plus a 24-hour chart. Use **Smart Recommendations** to compare several lots at once.

**B. From the command line.**

```bash
python -m src.predict --lot LOT_B --datetime "2026-11-07 15:00" --weather Rainy
python -m src.predict --lot LOT_B --datetime "2026-11-07 15:00" --hours 6
```

Sample inputs and the **format** of the answer (the values in `<...>` come from the trained model, so they are not listed here):

```
Parking lot : LOT_B - City Center Mall
Date & time : 2026-11-07 15:00:00
Predicted   : <free> of 500 spaces free (<percent>% occupied) -> <Available | Limited | Full>
Typical range: about <low>-<high> free
(An ESTIMATE from a model trained on the project's data - not a guarantee of real-time availability.)
```

The `--hours` form prints one row per hour with the columns `slot_label, predicted_available_spaces, total_spaces, predicted_occupancy_rate, status`.

**C. From Python.**

```python
from src.predict import predict_occupancy

result = predict_occupancy("LOT_B", "2026-11-07 15:00", weather="Rainy", event_today=False)
# result is a dictionary with the keys:
# timestamp, parking_lot_id, location, total_spaces, predicted_occupancy_rate,
# predicted_occupied_spaces, predicted_available_spaces, status, ...
```

**Input rules**

| Input | Allowed values |
|---|---|
| lot | `LOT_A` Downtown Office Plaza (300 spaces), `LOT_B` City Center Mall (500), `LOT_C` Central Hospital (250), `LOT_D` Railway Station (200) |
| date and time | any readable date/time; more than 14 days ahead the app warns that it is a what-if |
| weather | `Sunny`, `Cloudy`, `Rainy`, `Foggy`, or leave empty (treated as `Unknown`) |
| event | true/false; only has an effect 17:00-22:00 at lots that host events |
| hourly slots | 1 to 48 |

Status meaning: **Available** under 70% full, **Limited** 70% to under 95%, **Full** 95% or more.

## Dataset description
**The dataset is SIMULATED.** It is produced by `src/data_generator.py` from written rules and a fixed random seed, so it is reproducible. It is stored in `data/raw/parking_data_simulated.csv`.

- 4 lots x 24 hours x 90 days = 8,640 hourly records (the raw file also contains 30 deliberately duplicated rows and about 1% missing weather values, so the cleaning step has real work to do).
- Each lot has its own daily curve (office: weekdays 9-5; mall: weekend afternoons; hospital: steady, late-morning peak; station: morning commuter peak), plus weather effects, event boosts, day-to-day variation and random noise.

| Column | Meaning |
|---|---|
| `timestamp` | Date and hour |
| `parking_lot_id`, `location` | Lot id and name |
| `total_spaces`, `occupied_spaces`, `available_spaces` | Always `occupied + available = total` |
| `occupancy_rate` | `occupied / total` (the prediction target) |
| `hour`, `day_of_week`, `is_weekend` | Calendar information |
| `weather` | Sunny, Cloudy, Rainy, Foggy (some missing in the raw file) |
| `special_event` | 1 during an event evening at lots that have events |

Cleaning (`src/preprocessing.py`): duplicates removed, missing weather set to `Unknown`, impossible values dropped, totals re-checked. `data/processed/data_source.json` records whether the data is simulated; every page reads it.

## ML algorithms and evaluation metrics
The target is the **occupancy rate** (0 to 1).

| Model | What it is | Role |
|---|---|---|
| Baseline | Always predicts the average | A real model must beat it |
| Linear Regression | Straight-line relationship | Simple reference; cannot capture the hill-shaped daily pattern |
| Decision Tree | A flowchart of yes/no questions | Easy to explain; tuned |
| Random Forest | Many trees voting together | Robust to noise; tuned |
| Gradient Boosting | Trees built one after another, each fixing the last one's errors | Often the most accurate; tuned |

All models sit in one scikit-learn `Pipeline` (encoding/scaling + model), so training and the dashboard preprocess identically.

**Model selection.** The data is split **by time**: the first 80% of the timeline trains the models and the last 20% is held back as unseen test days. Tree models are tuned with `TimeSeriesSplit` (5 folds) on the training data only. The model with the lowest cross-validation RMSE is saved. The test days never influence that choice.

**Evaluation metrics**
- **MAE**: average size of the error. Reported as occupancy rate (0.05 = 5 percentage points) and in parking spaces.
- **RMSE**: punishes large misses more; saying "plenty of space" for a full lot is worse than being slightly off.
- **R²**: share of the ups and downs explained (1 = perfect, 0 = no better than the average).

**Where the real results are.** After you train, the scores for *your* run are in `outputs/results/model_comparison.csv`, `model_metadata.json` and `metrics_explained.txt`, and are shown on the **Model Evaluation** page. They are not copied into this README because they are produced by the run, not written by hand. Because the data is simulated and rule-based, scores will be high; **real data would be messier and scores would likely be lower.**

**Model inputs (features):** `parking_lot_id`, `weather`, `special_event`, `hour`, `day_of_week`, `is_weekend`, `is_rush_hour`, and cyclic versions of hour and weekday (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`). **Never used:** `occupied_spaces`, `available_spaces`, `occupancy_rate` (they *are* the answer) and the raw `timestamp`.

## Folder structure
```
Smart_Parking_Availability_Predictor/
|-- main.py                      <- whole pipeline in one command
|-- generate_dataset.py          <- regenerate + clean the simulated dataset
|-- train_model.py               <- train, compare and save the models
|-- requirements.txt
|-- pytest.ini
|-- README.md
|-- .gitignore
|-- .streamlit/config.toml
|-- app/                         <- Streamlit dashboard
|   |-- streamlit_app.py         <- entry point, sidebar, consistency banner
|   |-- context.py, ui.py        <- shared data container, styling, status badges
|   |-- views/
|   |   |-- home.py  availability.py  recommendations.py  prediction.py
|   |   |-- analytics.py  history.py  evaluation.py  about.py
|   `-- streamlit_app_phase1.py  <- the original single-page app (kept)
|-- src/
|   |-- config.py                <- paths, lots, thresholds, settings
|   |-- data_generator.py  preprocessing.py  features.py  eda.py
|   |-- data_checks.py           <- leakage checks, dataset fingerprint, schema check
|   |-- train.py  evaluation.py  <- tuning, selection, importance, error ranges, charts
|   |-- predict.py               <- input handling, capacity limits, time slots
|   |-- recommend.py             <- recommendations, optional distance data
|   |-- analysis.py              <- peak hours, weekday/weekend, trends, weather
|   `-- dashboard_data.py  dashboard_charts.py
|-- tests/                       <- pytest tests, page smoke test, a stand-in for Streamlit
|-- data/
|   |-- raw/                     <- parking_data_simulated.csv  (+ optional lot_distances.csv)
|   `-- processed/               <- parking_data_processed.csv, data_source.json
|-- models/best_model.joblib     <- the trained model
`-- outputs/
    |-- charts/                  <- EDA and model charts
    `-- results/                 <- metrics, predictions, importance, error ranges, reports
```

## The dashboard
`streamlit run app/streamlit_app.py`. Sidebar navigation, responsive layout, clear messages when something is missing.

| Page | What it shows |
|---|---|
| **Home** | Total / occupied / available spaces, overall occupancy %, status per location, last 24 hours |
| **Parking Availability** | One card per location with free/occupied spaces, occupancy % and a status badge; filter, sort, comparison chart |
| **Smart Recommendations** | Destination (optional), acceptable locations, arrival time, weather, event. Ranked options with predicted free spaces and a reason; best time to arrive over the next 3-24 hours |
| **Predict Availability** | One location, date, time, weather, event. Predicted free spaces, a 24-hour forecast, comparison across lots, and 3-48 hourly time slots (chart, heatmap, table, CSV) |
| **Charts & Analytics** | Hourly demand, day-of-week pattern, location comparison, actual vs predicted (unseen test days), model scores |
| **Historical Analysis** | Peak hours, weekday vs weekend, trends per location, weather and event effects |
| **Model Evaluation** | Comparison table, actual-vs-predicted for any model, feature importance, the leakage checks, typical error ranges |
| **About** | How it works and the limitations |

**"Current" values.** There is no live feed, so Home and Availability show a *snapshot of the dataset* (by default the latest reading), not what is happening in a real car park right now.

**Nothing is hardcoded or random.** Every number is read from the saved files or produced by the saved model, and tests check that different inputs give different predictions.

## Recommendations and distance data
Recommendations rank the selected locations by the model's **predicted free spaces** (or by *share* of free spaces). Locations predicted **Full** are never recommended; if every location is Full, the app says so. A destination highlights that location and adds advice about it.

> **The simulated dataset contains no distances, and the project has no GPS or map integration.** Unless you supply a distance file, recommendations are based **on parking availability only, not on actual travel distance**, and the app says so on screen.

To include distance, create `data/raw/lot_distances.csv`:

```csv
destination,parking_lot_id,distance_km
Central Hospital,LOT_A,0.4
Central Hospital,LOT_B,6.0
Central Hospital,LOT_C,0.2
Central Hospital,LOT_D,3.0
```

`destination` must be one of the location names in the app. The page then shows a slider (0 = availability only, 1 = nearest only). The distances are numbers *you* type in; nothing is measured by the app.

## Using genuine parking data
```bash
python main.py --data my_parking.csv
```
The file needs the columns `timestamp, parking_lot_id, location, total_spaces, occupied_spaces, available_spaces, weather, special_event` (hourly rows; `weather` and `special_event` may have gaps). Lot ids must exist in `PARKING_LOTS` in `src/config.py`. The file is validated first and rejected with a specific message if unusable. Afterwards the model, charts and dashboard say "supplied data" instead of "simulated". Running `python main.py` without `--data` returns to the simulation.

## Avoiding data leakage and fake accuracy
- The answer columns are never inputs; an assertion stops the program if one is added by mistake.
- **Time-based split:** test days come after all training days.
- Encoders and scalers are fitted **inside the pipeline on training data only**.
- **Model choice and tuning use training data only**; the test set is used once, to report scores.
- **9 automatic checks** run before training (`outputs/results/leakage_checks.json`); training stops if one fails.
- A **baseline** and train-vs-test scores make overfitting visible.
- The saved model stores a **fingerprint** of its dataset. If dataset, features, capacities, data type or test period no longer match, the dashboard shows a banner on every page.

## Robustness
- **Missing inputs:** missing weather becomes "Unknown" (a category seen in training); a missing event flag means "no event"; a missing lot or date raises a clear error.
- **Invalid values:** unknown lots, unreadable dates, unknown weather, impossible hour counts and events at lots that have none are rejected with a message.
- **Capacity limits:** predicted free and occupied spaces are always between 0 and capacity and add up to the capacity. Non-numeric model output is rejected.
- **Missing files:** every missing data/model/report file produces a message saying what to run, never a crash.
- **Stale caches:** re-training while the dashboard is open is detected.

## Tests
```bash
python -m pytest                           # pytest suite (unit, pipeline and package checks)
python tests/smoke_test_pages.py           # opens every dashboard page with many inputs
```
The pytest suite includes a test that runs the **entire pipeline** (generate, clean, train, predict, recommend) inside a temporary folder, so it never touches your real data or model. It also checks prediction boundaries (capacity, invalid inputs, extreme model output), leakage checks, that the required files are present, that every command in this README points at a real file, and that no API keys are needed.

The page smoke test uses `tests/fake_streamlit.py`, a small stand-in that copies Streamlit's strictest rules so pages can run without a browser. It is not a replacement for looking at the real dashboard, so open `streamlit run app/streamlit_app.py` once yourself. The unit tests can also be run with `python -m unittest discover -s tests`.

## Limitations
- **The data is simulated** unless you supply your own. Scores describe the pipeline, not real-world accuracy; weather, event and daily-pattern effects in the charts are the rules written into the simulation.
- **No live data:** no sensors, cameras, booking systems, GPS, maps or traffic feeds are connected. "Current" values are a dataset snapshot.
- Only 90 days of data: no seasons, holidays, school terms or long-term trends.
- The model knows only lot, hour, weekday, weather and an event flag.
- Future **weather is a user choice**, not a forecast.
- **No travel distances** unless you supply `lot_distances.csv`.
- The same weather and event assumptions apply to every hour of a time-slot forecast.
- Typical-error ranges are averages; real availability can fall outside them.
- Only the four lots in `src/config.py` are supported out of the box.
- The real Streamlit UI could not be launched in the environment where the project was built (no internet to install it), so the pages were verified with the stand-in described above. Please open the dashboard once on your own machine.

## Future scope
- Connect real parking sensors or a ticketing/barrier system for live occupancy.
- Train on real historical data from a car park operator and report real accuracy.
- Use a real weather forecast service, public holiday calendars and event schedules as inputs.
- Add real travel distance and time with a maps service.
- Add more lots, seasons and longer history; try time-series models and prediction intervals.
- Deploy the dashboard online and add notifications ("lot X will be full in an hour").

## Troubleshooting
- *"Trained model not found" / dashboard cannot start*: run `python train_model.py` (or `python main.py`), then refresh.
- *Model fails to load or a version warning appears*: run `python train_model.py` to re-train with your installed scikit-learn.
- *Red banner: dataset has changed since the model was trained*: run `python train_model.py`.
- *`ModuleNotFoundError: src` or `app`*: run commands from the project root folder.
- *`--data` is rejected*: the message lists exactly what to fix.
- *Charts look stale after retraining*: refresh the browser page.
