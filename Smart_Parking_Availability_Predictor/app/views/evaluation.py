"""Page - Model Evaluation: scores, unseen-data plots, feature importance, leakage checks."""

import json

import pandas as pd
import streamlit as st

from app import ui
from src import config
from src import dashboard_charts as charts
from src import dashboard_data as dd
from src.train import BASELINE_NAME, TREE_MODELS


def _tab_comparison(ctx):
    table = ctx.model_comparison
    if table is None or table.empty:
        st.info("No model comparison found. Run `python main.py` to create it.")
        return
    best = ctx.metadata.get("best_model")
    st.markdown(f"All models were scored on the same **unseen test days**. Selected model: **{best}**.")
    show = pd.DataFrame({
        "Model": table["model"],
        "MAE (rate)": table["test_mae"].round(4),
        "RMSE (rate)": table["test_rmse"].round(4),
        "R²": table["test_r2"].round(4),
        "MAE (spaces)": table["test_mae_spaces"].round(1),
        "CV RMSE (training)": table["cv_rmse"].round(4),
        "Train R² - test R²": table["r2_gap_train_minus_test"].round(3) if "r2_gap_train_minus_test" in table else None,
        "Selected": ["✅" if m == best else "" for m in table["model"]],
    })
    st.dataframe(show, hide_index=True)
    ui.show_fig(charts.fig_model_metric_bars(table, best))

    with st.expander("What do MAE, RMSE and R² mean?", expanded=False):
        st.markdown(
            "- **MAE**: average size of the error. *(rate)* = occupancy rate (0.05 = 5 percentage points), "
            "*(spaces)* = parking spaces.\n"
            "- **RMSE**: like MAE but punishes big misses more.\n"
            "- **R²**: share of the ups and downs the model explains (1 = perfect, 0 = no better than the "
            "average).\n"
            "- **CV RMSE**: the error from time-ordered cross-validation inside the *training* data. "
            "It is used to choose the model, so the test days stay untouched.\n"
            "- **Train R² - test R²**: a large gap means overfitting.")
    rule = ctx.metadata.get("selection_rule")
    if rule:
        st.caption(f"Selection rule: {rule}.")
    by_test = ctx.metadata.get("best_model_by_test_rmse")
    if by_test and by_test != best:
        st.caption(f"{by_test} scored marginally better on the test days, but test days are not used to pick "
                   "a model, otherwise the test scores would no longer be an honest estimate.")
    tuning = ctx.metadata.get("tuning", {})
    if tuning.get("enabled") and tuning.get("best_params"):
        with st.expander("Hyper-parameters chosen by the search (training data only)"):
            st.json(tuning["best_params"])


def _tab_unseen(ctx):
    allp = ctx.all_predictions
    if allp is None or allp.empty:
        st.info("No test predictions found. Run `python main.py --train-only` to create "
                "`outputs/results/test_predictions_all_models.csv`.")
        return
    split = ctx.metadata.get("split", {})
    first, last = allp["timestamp"].min(), allp["timestamp"].max()
    st.success(f"**Unseen data only.** These plots use {first:%d %b %Y} to {last:%d %b %Y}"
               + (f", which start after training ended ({split['train_end'][:10]})." if split else "."))
    models = [m for m in allp["model"].unique() if m != BASELINE_NAME] + [BASELINE_NAME]
    default = models.index(ctx.metadata["best_model"]) if ctx.metadata.get("best_model") in models else 0
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    model_name = c1.selectbox("Model", models, index=default, key="ev_model")
    location = c2.selectbox("Location", sorted(allp["location"].unique()), key="ev_location")
    d1, d2 = first.date(), last.date()
    start = c3.date_input("Start", value=d1, min_value=d1, max_value=d2, key="ev_start")
    span = c4.selectbox("Days", [1, 3, 7, 18], index=2, key="ev_days")

    rows = allp[(allp["model"] == model_name) & (allp["location"] == location)]
    start_ts = pd.Timestamp(start)
    window = rows[(rows["timestamp"] >= start_ts) & (rows["timestamp"] < start_ts + pd.Timedelta(days=span))]
    if window.empty:
        st.info("No test data for this selection.")
        return
    metrics = dd.model_metrics_on(allp, model_name, location, start, span)
    if metrics:
        ui.kpi_row([
            {"label": "Hours compared", "value": f"{metrics['rows']}"},
            {"label": "MAE", "value": f"{metrics['mae_spaces']:.1f} spaces"},
            {"label": "RMSE", "value": f"{metrics['rmse_spaces']:.1f} spaces"},
            {"label": "R²", "value": f"{metrics['r2']:.3f}"},
        ])
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.show_fig(charts.fig_model_actual_vs_predicted(window, model_name, location))
    with right:
        ui.show_fig(charts.fig_actual_vs_predicted_scatter(window, location))
    st.caption("Dots close to the dashed diagonal are good predictions. The simple baseline and Linear "
               "Regression show what a weak model looks like.")


def _tab_importance(ctx):
    table = ctx.importance
    if table is None or table.empty:
        st.info("No feature importance found. Run `python main.py --train-only` to create it.")
        return
    best = ctx.metadata.get("best_model", "")
    grouped = table[table["kind"] == "permutation_grouped"]
    if not grouped.empty:
        st.markdown(f"**Which kinds of input matter most?** ({best}, measured on the unseen test days)")
        ui.show_fig(charts.fig_feature_importance(
            grouped, "Increase in error when the input is scrambled (higher = more important)"))
        top = grouped.sort_values("importance", ascending=False).iloc[0]
        st.caption(f"Scrambling **{top['feature'].lower()}** hurts the predictions most. Related inputs "
                   "(for example hour, hour_sin and hour_cos) are scrambled together so their importance is not "
                   "split. Bars show the error increase in occupancy-rate units; whiskers show repeat-to-repeat variation.")
    trees = [m for m in TREE_MODELS if m in set(table.loc[table["kind"] == "impurity", "model"])]
    if trees:
        st.markdown("**Built-in importance of tree-based models**")
        model_name = st.selectbox("Tree model", trees, index=trees.index(best) if best in trees else 0,
                                  key="ev_tree_model")
        impurity = table[(table["kind"] == "impurity") & (table["model"] == model_name)]
        by_group = impurity.groupby("group", as_index=False)["importance"].sum().rename(columns={"group": "feature"})
        ui.show_fig(charts.fig_feature_importance(by_group, f"{model_name}: share of splits explained"))
        st.caption("Importance is summed over related inputs (all weather columns count as 'weather'). "
                   "Linear Regression and the baseline have no tree importance, so they are not listed.")


def _tab_integrity(ctx):
    checks = dd.load_leakage_checks()
    if not checks:
        st.info("No leakage report found. Run `python main.py --train-only` to create it.")
    else:
        failed = [c for c in checks if not c["ok"]]
        (st.error if failed else st.success)(
            f"{len(checks) - len(failed)} of {len(checks)} train/test separation checks passed.")
        st.dataframe(pd.DataFrame({"Check": [c["check"] for c in checks],
                                   "Result": ["✅ pass" if c["ok"] else "❌ FAIL" for c in checks],
                                   "Detail": [c["detail"] for c in checks]}), hide_index=True)
    st.markdown("**How far off can a forecast be?** Typical error per location, in free parking spaces.")
    path = config.ERROR_BANDS_PATH
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        coverage = data.get("share_of_test_errors_within_p80", {})
        rows = [{"Location": config.PARKING_LOTS[lot]["location"] if lot in config.PARKING_LOTS else lot,
                 "Average error": v["mae_spaces"], "80% of errors below": v["p80_spaces"],
                 "90% of errors below": v["p90_spaces"],
                 "Unseen test errors inside the 80% band": f"{coverage[lot]:.0%}" if lot in coverage else "-"}
                for lot, v in data["lots"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("The bands were measured on training-period forecasts only. The last column checks them on "
                   "the unseen test days: a value near 80% means the band is honest.")
    else:
        st.info("No error bands found. Run `python main.py --train-only` to create them.")
    meta = ctx.metadata
    if meta.get("dataset_fingerprint"):
        fp = meta["dataset_fingerprint"]
        st.caption(f"Model trained on {fp['rows']:,} rows ({fp['first_timestamp'][:10]} to "
                   f"{fp['last_timestamp'][:10]}); data source: {meta.get('data_source_file', '?')} "
                   f"({'simulated' if meta.get('data_is_simulated', True) else 'supplied'}).")


def render(ctx):
    ui.page_header("Model Evaluation", "How good is the model, and can the scores be trusted?")
    tabs = st.tabs(["Model comparison", "Actual vs predicted (unseen)", "Feature importance",
                    "Integrity & error ranges"])
    with tabs[0]:
        _tab_comparison(ctx)
    with tabs[1]:
        _tab_unseen(ctx)
    with tabs[2]:
        _tab_importance(ctx)
    with tabs[3]:
        _tab_integrity(ctx)
