"""Train, tune, compare and save the machine-learning models.

Steps
-----
1. Load the processed data and split it BY TIME: the first 80% of the timeline
   trains the models, the last 20% (the "future") tests them.
2. Run leakage checks on the split (see data_checks.py).
3. Build one pipeline (preprocessing + model) per model. Tree models get a small
   hyper-parameter search that uses ONLY the training data (time-ordered folds).
4. Score every model on the unseen test days: MAE, RMSE, R2.
5. Choose the best model by CROSS-VALIDATION error on the training data. The test
   days are never used to choose anything, so the test scores stay an honest,
   untouched estimate. (The test-best model is reported too, for comparison.)
6. Save the model, the metrics, test predictions for EVERY model, feature
   importance, "typical error" bands per lot and a dataset fingerprint.

Phase 3 changes: tuning, CV-based selection, leakage checks, importance, error bands.
"""

import json
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config, data_checks, evaluation
from src.features import FEATURE_COLUMNS, TARGET, build_preprocessor, get_xy

BASELINE_NAME = "Baseline (always predict average)"

# Small search grids (kept small on purpose so training stays quick).
# "model__" means "a setting of the model step inside the pipeline".
TUNING_GRIDS = {
    "Decision Tree": {"model__max_depth": [6, 8, 10, 12], "model__min_samples_leaf": [5, 10, 20]},
    "Random Forest": {"model__min_samples_leaf": [3, 5, 10], "model__max_depth": [None, 14]},
    "Gradient Boosting": {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 4]},
}
TREE_MODELS = ["Decision Tree", "Random Forest", "Gradient Boosting"]

# Inputs that carry the same information (hour, hour_sin, hour_cos ...) are shuffled
# TOGETHER, otherwise their importance is split between them and looks smaller than it is.
FEATURE_GROUPS = {
    "Time of day": ["hour", "hour_sin", "hour_cos", "is_rush_hour"],
    "Day of week": ["day_of_week", "dow_sin", "dow_cos", "is_weekend"],
    "Parking location": ["parking_lot_id"],
    "Weather": ["weather"],
    "Special event": ["special_event"],
}


def load_processed_data(path=None):
    path = path or config.PROCESSED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\nRun `python main.py` first.")
    return pd.read_csv(path, parse_dates=["timestamp"])


def time_based_split(df, test_fraction=config.TEST_FRACTION):
    """Train = earlier timestamps, Test = the most recent timestamps."""
    timestamps = np.sort(df["timestamp"].unique())
    cutoff = timestamps[int(len(timestamps) * (1 - test_fraction))]
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    # Safety check: the test period must start AFTER the training period ends.
    assert train["timestamp"].max() < test["timestamp"].min(), "train/test overlap!"
    return train, test, pd.Timestamp(cutoff)


def get_models():
    """The models to compare: name -> (estimator, needs_feature_scaling)."""
    seed = config.RANDOM_SEED
    return {
        BASELINE_NAME: (DummyRegressor(strategy="mean"), False),
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, min_samples_leaf=10,
                                                random_state=seed), False),
        "Random Forest": (RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                n_jobs=-1, random_state=seed), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                        max_depth=3, random_state=seed), False),
    }


def _group_of(encoded_name):
    """'cat__weather_Rainy' -> 'weather', 'num__hour' -> 'hour' (back to the original column)."""
    name = encoded_name.split("__", 1)[-1]
    for prefix in ("parking_lot_id", "weather"):
        if name.startswith(prefix + "_"):
            return prefix
    return name


def impurity_importance(pipeline, model_name):
    """Importance of every (encoded) input for a tree-based model, as a DataFrame."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    return pd.DataFrame({
        "model": model_name, "kind": "impurity",
        "feature": [n.split("__", 1)[-1] for n in names],
        "group": [_group_of(n) for n in names],
        "importance": model.feature_importances_, "std": np.nan,
    })


def grouped_permutation_importance(pipeline, X, y, n_repeats=5, seed=config.RANDOM_SEED):
    """How much worse (RMSE) does the model get when one GROUP of inputs is scrambled?

    Rows of the group's columns are shuffled together (so hour, hour_sin and hour_cos
    stay consistent with each other). The model is only asked to predict - nothing is
    fitted on the data passed in, so using the test set here does not leak anything.
    """
    covered = sorted(c for cols in FEATURE_GROUPS.values() for c in cols)
    assert covered == sorted(X.columns), "FEATURE_GROUPS must cover every input exactly once"
    rng = np.random.default_rng(seed)
    base = float(np.sqrt(np.mean((y.values - evaluation.clip_rate(pipeline.predict(X))) ** 2)))
    rows = []
    for group, columns in FEATURE_GROUPS.items():
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[columns] = X[columns].iloc[order].values
            rmse = float(np.sqrt(np.mean((y.values - evaluation.clip_rate(pipeline.predict(shuffled))) ** 2)))
            increases.append(rmse - base)
        rows.append({"feature": group, "importance": float(np.mean(increases)),
                     "std": float(np.std(increases))})
    return pd.DataFrame(rows)


def compute_error_bands(pipeline, X_train, y_train, train_df, cv):
    """Typical error per lot in PARKING SPACES, measured only on training data.

    For every time-series fold the model is re-fitted on the earlier part and asked
    to predict the later part, which mimics real forecasting. The errors of those
    "out-of-fold" predictions give an honest idea of how far off a forecast can be.
    """
    rows = []
    for fit_idx, val_idx in cv.split(X_train):
        fold_model = clone(pipeline).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        predicted = evaluation.clip_rate(fold_model.predict(X_train.iloc[val_idx]))
        validation = train_df.iloc[val_idx]
        rows.append(pd.DataFrame({
            "lot": validation["parking_lot_id"].values,
            "abs_error_spaces": np.abs(y_train.iloc[val_idx].values - predicted)
                                * validation["total_spaces"].values}))
    errors = pd.concat(rows)
    lots = {}
    for lot, group in errors.groupby("lot"):
        e = group["abs_error_spaces"]
        lots[lot] = {"mae_spaces": round(float(e.mean()), 2), "p50_spaces": round(float(e.quantile(0.5)), 2),
                     "p80_spaces": round(float(e.quantile(0.8)), 2), "p90_spaces": round(float(e.quantile(0.9)), 2),
                     "n_predictions": int(len(e))}
    return lots


def run_training(tune=None):
    """Train everything. `tune=None` uses config.TUNE_MODELS."""
    tune = config.TUNE_MODELS if tune is None else tune
    config.ensure_directories()

    # ---- 1) data + time-based split + leakage checks ----------------------
    df = load_processed_data()
    train_df, test_df, cutoff = time_based_split(df)
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)

    checks = data_checks.check_split(train_df, test_df, df)
    (config.RESULTS_DIR / "leakage_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    data_checks.assert_clean_split(checks)

    print(f"[train] Training rows: {len(train_df):,}  ({train_df['timestamp'].min().date()} "
          f"to {train_df['timestamp'].max().date()})")
    print(f"[train] Test rows:     {len(test_df):,}  ({test_df['timestamp'].min().date()} "
          f"to {test_df['timestamp'].max().date()})")
    print(f"[train] Leakage checks passed: {sum(c['ok'] for c in checks)}/{len(checks)}")
    print(f"[train] Input features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"[train] Hyper-parameter tuning: {'ON (training data only)' if tune else 'OFF'}")

    # ---- 2) train + evaluate every model ---------------------------------
    cv = TimeSeriesSplit(n_splits=config.N_CV_SPLITS)
    rows, fitted, best_params, all_predictions = [], {}, {}, []

    for name, (estimator, needs_scaling) in get_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=needs_scaling)),
            ("model", estimator),
        ])

        if tune and name in TUNING_GRIDS:
            search = GridSearchCV(pipeline, TUNING_GRIDS[name], cv=cv, refit=True,
                                  scoring="neg_root_mean_squared_error",
                                  n_jobs=1 if name == "Random Forest" else -1)
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            cv_rmse = -search.best_score_
            best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        else:
            cv_rmse = -cross_val_score(pipeline, X_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error").mean()
            pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        train_m = evaluation.compute_metrics(y_train, train_pred)
        test_m = evaluation.compute_metrics(y_test, test_pred, test_df["total_spaces"])

        rows.append({
            "model": name,
            "cv_rmse": cv_rmse,
            "train_mae": train_m["mae"], "train_rmse": train_m["rmse"], "train_r2": train_m["r2"],
            "test_mae": test_m["mae"], "test_rmse": test_m["rmse"], "test_r2": test_m["r2"],
            "test_mae_spaces": test_m["mae_spaces"],
            "r2_gap_train_minus_test": train_m["r2"] - test_m["r2"],
            "best_params": json.dumps(best_params.get(name, {})),
        })
        fitted[name] = (pipeline, test_pred)
        clipped_all = evaluation.clip_rate(test_pred)
        all_predictions.append(pd.DataFrame({
            "timestamp": test_df["timestamp"].values, "parking_lot_id": test_df["parking_lot_id"].values,
            "location": test_df["location"].values, "total_spaces": test_df["total_spaces"].values,
            "model": name, "actual_occupancy_rate": y_test.values,
            "predicted_occupancy_rate": clipped_all.round(4),
            "actual_available_spaces": test_df["available_spaces"].values,
            "predicted_available_spaces": (test_df["total_spaces"].values
                - np.clip(np.round(clipped_all * test_df["total_spaces"].values), 0,
                          test_df["total_spaces"].values)).astype(int)}))
        extra = f"  params={best_params[name]}" if name in best_params else ""
        print(f"[train] {name:<36} CV RMSE={cv_rmse:.4f}  test MAE={test_m['mae']:.4f}  "
              f"RMSE={test_m['rmse']:.4f}  R2={test_m['r2']:.4f}{extra}")

    results = pd.DataFrame(rows)

    # ---- 3) choose the best model WITHOUT looking at the test set -----------
    candidates = results[results["model"] != BASELINE_NAME]
    best_name = candidates.sort_values("cv_rmse").iloc[0]["model"]
    test_best_name = candidates.sort_values("test_rmse").iloc[0]["model"]
    best_pipeline, best_test_pred = fitted[best_name]
    best_row = results[results["model"] == best_name].iloc[0]
    results["selected"] = results["model"] == best_name
    results = results.sort_values("cv_rmse").reset_index(drop=True)
    results.to_csv(config.MODEL_COMPARISON_PATH, index=False)

    print("\n[train] Model comparison (sorted by CV RMSE, i.e. the selection rule):")
    print(results.drop(columns=["best_params"]).round(4).to_string(index=False))
    print(f"\n[train] Best model (lowest CV RMSE on training data): {best_name}")
    if test_best_name != best_name:
        print(f"[train] Note: {test_best_name} scored slightly better on the test days, but the test "
              "days are not used to choose a model.")

    # ---- 4) save model, predictions, importance, error bands ----------------
    joblib.dump(best_pipeline, config.MODEL_PATH)

    clipped = evaluation.clip_rate(best_test_pred)
    totals = test_df["total_spaces"].values
    predicted_free = (totals - np.clip(np.round(clipped * totals), 0, totals)).astype(int)
    predictions = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "parking_lot_id": test_df["parking_lot_id"].values,
        "location": test_df["location"].values,
        "total_spaces": totals,
        "actual_occupancy_rate": y_test.values,
        "predicted_occupancy_rate": clipped.round(4),
        "actual_available_spaces": test_df["available_spaces"].values,
        "predicted_available_spaces": predicted_free,
    })
    predictions.to_csv(config.RESULTS_DIR / "test_predictions.csv", index=False)
    pd.concat(all_predictions).to_csv(config.ALL_TEST_PREDICTIONS_PATH, index=False)

    per_lot = (predictions.assign(abs_error_spaces=lambda d: (d["actual_available_spaces"]
                                                              - d["predicted_available_spaces"]).abs())
               .groupby("location")["abs_error_spaces"].mean().round(2)
               .rename("mae_available_spaces").reset_index())
    per_lot.to_csv(config.RESULTS_DIR / "per_lot_test_error.csv", index=False)

    # feature importance: built-in importance for every tree model + permutation importance
    # (measured on the test set, which only READS the data; nothing is fitted on it)
    importance = [impurity_importance(fitted[name][0], name) for name in TREE_MODELS if name in fitted]
    perm = permutation_importance(best_pipeline, X_test, y_test, scoring="neg_root_mean_squared_error",
                                  n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=1)
    importance.append(pd.DataFrame({
        "model": best_name, "kind": "permutation", "feature": list(X_test.columns),
        "group": list(X_test.columns), "importance": perm.importances_mean, "std": perm.importances_std}))
    grouped = grouped_permutation_importance(best_pipeline, X_test, y_test)
    grouped.insert(0, "model", best_name)
    grouped.insert(1, "kind", "permutation_grouped")
    grouped["group"] = grouped["feature"]
    importance.append(grouped)
    pd.concat(importance, ignore_index=True).to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)

    # typical error per lot (training data only), then check it on the test days
    bands = compute_error_bands(best_pipeline, X_train, y_train, train_df.reset_index(drop=True), cv)
    test_errors = pd.DataFrame({"lot": test_df["parking_lot_id"].values,
                                "error": np.abs(predictions["actual_available_spaces"]
                                                - predictions["predicted_available_spaces"]).values})
    coverage = {lot: round(float((g["error"] <= bands[lot]["p80_spaces"]).mean()), 3)
                for lot, g in test_errors.groupby("lot")}
    (config.ERROR_BANDS_PATH).write_text(json.dumps({
        "method": "Absolute error in spaces of out-of-fold forecasts on TRAINING data "
                  "(time-series cross-validation). p80 = 80% of forecasts were this close or closer.",
        "lots": bands,
        "share_of_test_errors_within_p80": coverage,
    }, indent=2), encoding="utf-8")
    print(f"[train] Share of unseen test errors inside the 80% band (ideal ~0.80): {coverage}")

    source = {"is_simulated": True}
    if config.DATA_SOURCE_PATH.exists():
        source = json.loads(config.DATA_SOURCE_PATH.read_text(encoding="utf-8"))

    metadata = {
        "data_is_simulated": bool(source.get("is_simulated", True)),
        "data_source_file": source.get("source_file", config.RAW_DATA_PATH.name),
        "dataset_fingerprint": data_checks.dataset_fingerprint(df),
        "best_model": best_name,
        "selection_rule": "lowest time-series cross-validation RMSE on the training data "
                          "(the test days are not used to choose a model)",
        "best_model_by_test_rmse": test_best_name,
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "tuning": {"enabled": bool(tune), "best_params": best_params},
        "leakage_checks_passed": all(c["ok"] for c in checks),
        "split": {
            "type": "time-based",
            "train_end": str(train_df["timestamp"].max()),
            "test_start": str(test_df["timestamp"].min()),
            "test_end": str(test_df["timestamp"].max()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "test_metrics": {k: round(float(best_row[k]), 4)
                         for k in ["test_mae", "test_rmse", "test_r2", "test_mae_spaces"]},
        "cv_rmse_on_training_data": round(float(best_row["cv_rmse"]), 4),
        "library_versions": {"python": platform.python_version(),
                             "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    (config.RESULTS_DIR / "metrics_explained.txt").write_text(
        evaluation.METRIC_EXPLANATIONS, encoding="utf-8")

    # ---- 5) charts --------------------------------------------------------
    evaluation.plot_model_comparison(results, best_name)
    evaluation.plot_actual_vs_predicted(y_test, best_test_pred, best_name)
    evaluation.plot_residuals(y_test, best_test_pred, best_name)
    evaluation.plot_feature_importance(best_pipeline, best_name)
    evaluation.plot_forecast_example(predictions, best_name)

    print(f"[train] Model saved: {config.MODEL_PATH}")
    print(f"[train] Results saved in: {config.RESULTS_DIR}")
    print(f"[train] Charts saved in:  {config.CHARTS_DIR}")
    print("\n" + evaluation.METRIC_EXPLANATIONS)
    return results, best_name


if __name__ == "__main__":
    run_training()
