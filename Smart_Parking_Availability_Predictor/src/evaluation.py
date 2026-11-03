"""Model evaluation: metrics and result charts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

sns.set_theme(style="whitegrid")

METRIC_EXPLANATIONS = """\
WHY THESE METRICS?
------------------
We predict a number (how full a lot is), so we use regression metrics.

MAE  - Mean Absolute Error
       The average size of the mistakes, ignoring direction.
       Easy to understand: MAE = 0.05 means "off by 5 percentage points on average".
       We also report it in parking spaces, which is the unit drivers care about.

RMSE - Root Mean Squared Error
       Like MAE, but big mistakes are punished more. In parking a big miss
       (saying "plenty of space" when the lot is full) is worse than several
       small misses, so RMSE is a useful stricter check. RMSE >= MAE always;
       a large gap between them means there are some big outlier errors.

R2   - R-squared (coefficient of determination)
       How much of the up-and-down movement of occupancy the model explains.
       1.0 = perfect, 0.0 = no better than always predicting the average,
       below 0 = worse than that. It allows comparing models on one scale.

All metrics are measured on TEST data the model has never seen (the most
recent days), because scores on training data would be too optimistic.
"""


def clip_rate(predictions):
    """Occupancy can't be below 0% or above 100%."""
    return np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)


def compute_metrics(y_true, y_pred, total_spaces=None):
    """Return MAE, RMSE and R2 (and MAE in parking spaces if totals are given)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_rate(y_pred)
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if total_spaces is not None:
        # error in available spaces = total * (true rate - predicted rate)
        metrics["mae_spaces"] = float(np.mean(np.abs((y_true - y_pred) * np.asarray(total_spaces))))
    return metrics


def _save(fig, filename):
    fig.text(0.99, 0.002, "Simulated data", ha="right", fontsize=8, color="gray")
    path = config.CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


BEST_COLOR = "#2a9d8f"
OTHER_COLOR = "#aab4c3"


def plot_model_comparison(results, best_name):
    """Bar charts comparing all models on the test set."""
    panels = [("test_mae", "MAE (lower is better)"),
              ("test_rmse", "RMSE (lower is better)"),
              ("test_r2", "R² (higher is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (column, title) in zip(axes, panels):
        data = results.sort_values(column, ascending=(column != "test_r2"))
        colors = [BEST_COLOR if m == best_name else OTHER_COLOR for m in data["model"]]
        ax.barh(data["model"], data[column], color=colors)
        ax.invert_yaxis()
        for y, value in enumerate(data[column]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=9)
        ax.set_title(title)
        ax.margins(x=0.18)
    fig.suptitle(f"Model comparison on the TEST set  (best: {best_name})", fontsize=13)
    return _save(fig, "model_01_comparison.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    y_pred = clip_rate(y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(np.asarray(y_true) * 100, y_pred * 100, s=8, alpha=0.3, color=BEST_COLOR)
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Perfect prediction")
    ax.set(title=f"{model_name}: actual vs predicted (test set)",
           xlabel="Actual occupancy (%)", ylabel="Predicted occupancy (%)")
    ax.legend()
    return _save(fig, "model_02_actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, model_name):
    errors = (np.asarray(y_true) - clip_rate(y_pred)) * 100
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.histplot(errors, bins=40, kde=True, color=BEST_COLOR, ax=ax)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set(title=f"{model_name}: prediction errors (actual - predicted)",
           xlabel="Error (percentage points)", ylabel="Count")
    return _save(fig, "model_03_residuals.png")


def plot_feature_importance(pipeline, model_name):
    """Show which inputs the model relies on (tree-based models only)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    importance = (pd.Series(model.feature_importances_, index=names)
                  .sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(importance.index, importance.values, color=BEST_COLOR)
    ax.invert_yaxis()
    ax.set(title=f"{model_name}: top feature importances", xlabel="Importance")
    return _save(fig, "model_04_feature_importance.png")


def plot_forecast_example(predictions, model_name, days=7):
    """Actual vs predicted occupancy for the first week of the test period."""
    lots = list(config.PARKING_LOTS.items())
    rows = (len(lots) + 1) // 2
    start = predictions["timestamp"].min()
    week = predictions[predictions["timestamp"] < start + pd.Timedelta(days=days)]

    fig, axes = plt.subplots(rows, 2, figsize=(15, 3.8 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (lot_id, info) in zip(axes, lots):
        sub = week[week["parking_lot_id"] == lot_id]
        ax.plot(sub["timestamp"], sub["actual_occupancy_rate"] * 100,
                label="Actual", color="#264653", linewidth=1.4)
        ax.plot(sub["timestamp"], sub["predicted_occupancy_rate"] * 100,
                label="Predicted", color="#e76f51", linewidth=1.2, linestyle="--")
        ax.set(title=info["location"], ylabel="Occupancy (%)")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[len(lots):]:
        ax.axis("off")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"{model_name}: first {days} days of the test period", fontsize=13)
    return _save(fig, "model_05_forecast_example.png")
