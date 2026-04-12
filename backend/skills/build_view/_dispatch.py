"""
build_view() — main dispatch and auto-explanation.

Routes a ViewPlan to the appropriate chart or analytics builder based on
chart_type and plan.tags, pre-aggregates data, and returns a ViewResult.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from core.models import ChartSpec, ChartType, ViewPlan, ViewResult
from ._utils import _reconcile_encoding
from ._charts import (
    _aggregate_bar,
    _aggregate_line,
    _build_area,
    _build_box,
    _build_heatmap,
    _build_histogram,
    _build_missingness,
    _build_pie,
    _build_scatter,
    _build_table,
)
from ._analytics import (
    _build_autocorrelation_test,
    _build_binning_optimizer,
    _build_cohort_change,
    _build_date_part_features,
    _build_diff_in_diff,
    _build_drift_check,
    _build_group_comparison,
    _build_hypothesis_generator,
    _build_interaction_scan,
    _build_lag_feature_search,
    _build_leakage_scan,
    _build_lgbm_classification,
    _build_lgbm_regression,
    _build_linear_regression,
    _build_logistic_regression,
    _build_matched_comparison,
    _build_missingness_mechanism,
    _build_numeric_transforms,
    _build_partial_dependence,
    _build_percentile_compare,
    _build_quantile_regression,
    _build_result_validator,
    _build_rolling_stats,
    _build_seasonality_test,
    _build_segmentation,
    _build_shap_dependence,
    _build_shap_summary,
    _build_target_encoding,
    _build_test_selector,
    _build_trend_breaks,
    _build_uplift_check,
)

logger = logging.getLogger("uvicorn.error")

# Standard chart type → builder
_BUILDERS = {
    ChartType.bar:     _aggregate_bar,
    ChartType.line:    _aggregate_line,
    ChartType.scatter: _build_scatter,
    ChartType.hist:    _build_histogram,
    ChartType.box:     _build_box,
    ChartType.table:   _build_table,
    ChartType.heatmap: _build_heatmap,
    ChartType.pie:     _build_pie,
    ChartType.area:    _build_area,
}

# Analytics tag → builder (checked before standard dispatch)
_ANALYTICS_DISPATCH = [
    ("percentile_compare",     _build_percentile_compare),
    ("segmentation",           _build_segmentation),
    ("cohort_change",          _build_cohort_change),
    ("group_comparison",       _build_group_comparison),
    ("matched_comparison",     _build_matched_comparison),
    ("diff_in_diff",           _build_diff_in_diff),
    ("uplift_check",           _build_uplift_check),
    ("regression_linear",      _build_linear_regression),
    ("regression_logistic",    _build_logistic_regression),
    ("lgbm_regression",        _build_lgbm_regression),
    ("lgbm_classification",    _build_lgbm_classification),
    ("quantile_regression",    _build_quantile_regression),
    ("seasonality_test",       _build_seasonality_test),
    ("autocorrelation_test",   _build_autocorrelation_test),
    ("lag_feature_search",     _build_lag_feature_search),
    ("rolling_stats",          _build_rolling_stats),
    ("trend_breaks",           _build_trend_breaks),
    ("numeric_transforms",     _build_numeric_transforms),
    ("interaction_scan",       _build_interaction_scan),
    ("binning_optimizer",      _build_binning_optimizer),
    ("date_part_features",     _build_date_part_features),
    ("target_encoding",        _build_target_encoding),
    ("shap_summary",           _build_shap_summary),
    ("shap_dependence",        _build_shap_dependence),
    ("partial_dependence",     _build_partial_dependence),
    ("leakage_scan",           _build_leakage_scan),
    ("drift_check",            _build_drift_check),
    ("missingness_mechanism",  _build_missingness_mechanism),
    ("hypothesis_generator",   _build_hypothesis_generator),
    ("test_selector",          _build_test_selector),
    ("result_validator",       _build_result_validator),
]


def build_view(plan: ViewPlan, df: pd.DataFrame) -> ViewResult:
    """
    Build a ViewResult from a ViewPlan and a DataFrame.

    All data is pre-aggregated: the frontend just renders.
    """
    # Analytics dispatch (tag-based, checked first)
    data: List[Dict[str, Any]] = []
    dispatched = False
    for tag, builder in _ANALYTICS_DISPATCH:
        if tag in plan.tags:
            data = builder(plan, df)
            dispatched = True
            break

    # Missingness overview (field-based, not tag-based)
    if not dispatched and ("__missing_pct__" in plan.fields_used or "missingness" in plan.tags):
        data = _build_missingness(plan, df)
        dispatched = True

    # Standard chart dispatch
    if not dispatched:
        builder = _BUILDERS.get(plan.chart_type, _build_table)
        data = builder(plan, df)

    encoding = _reconcile_encoding(plan.chart_type, plan.encoding, data)

    spec = ChartSpec(
        chart_type=plan.chart_type,
        encoding=encoding,
        options=plan.options,
        data_inline=data,
        title=plan.intent or f"{plan.chart_type.value} chart",
    )

    explanation = _auto_explanation(plan, data, df)

    if not data:
        logger.warning(
            "View data empty: chart=%s intent=%s fields=%s",
            plan.chart_type.value, plan.intent, plan.fields_used,
        )
    else:
        logger.info(
            "View built: chart=%s keys=%s intent=%s",
            plan.chart_type.value, list(data[0].keys()), plan.intent,
        )

    return ViewResult(
        plan=plan,
        spec=spec,
        data_inline=data,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Auto-explanation
# ---------------------------------------------------------------------------

_ANALYTICS_EXPLANATIONS: Dict[str, str] = {
    "percentile_compare":    "To compare percentile groups and quantify differences.",
    "segmentation":          "To summarize the metric by group and spot segment differences.",
    "cohort_change":         "To track how cohorts change over time.",
    "group_comparison":      "To compare two groups (naive difference, not causal).",
    "matched_comparison":    "To estimate treated vs control effect via matching.",
    "diff_in_diff":          "To estimate a pre/post treatment effect (naive DiD).",
    "uplift_check":          "To estimate uplift between treated and control groups.",
    "regression_linear":     "To estimate a linear relationship between the selected variables.",
    "regression_logistic":   "To estimate classification likelihood from the selected predictor.",
    "lgbm_regression":       "To fit a non-linear model and inspect feature importance.",
    "lgbm_classification":   "To fit a non-linear model and inspect feature importance.",
    "quantile_regression":   "To compare effects across outcome quantiles.",
    "seasonality_test":      "To test for seasonal signal in the time series.",
    "autocorrelation_test":  "To summarize autocorrelation structure.",
    "lag_feature_search":    "To identify predictive lag candidates.",
    "rolling_stats":         "To summarize rolling mean/volatility.",
    "trend_breaks":          "To check for regime shifts over time.",
    "numeric_transforms":    "To assess whether transformations reduce skew.",
    "interaction_scan":      "To scan for useful interaction features.",
    "binning_optimizer":     "To capture non-linear effects via binning.",
    "date_part_features":    "To test date-part feature signal.",
    "target_encoding":       "To evaluate categorical target encoding.",
    "shap_summary":          "To explain model predictions with SHAP.",
    "shap_dependence":       "To explain model predictions with SHAP.",
    "partial_dependence":    "To summarize partial dependence for a feature.",
    "leakage_scan":          "To check for near-leakage predictors.",
    "drift_check":           "To compare distributions across time splits.",
    "missingness_mechanism": "To test missingness impact on target.",
    "hypothesis_generator":  "To propose testable hypotheses.",
    "test_selector":         "To map hypotheses to recommended tests.",
    "result_validator":      "To validate results against the hypothesis.",
}


def _auto_explanation(plan: ViewPlan, data: List[Dict[str, Any]], df: pd.DataFrame) -> str:
    """Generate a simple textual explanation of what the chart shows."""
    if not data:
        return "No data available for this view."

    ct = plan.chart_type.value
    fields = ", ".join(plan.fields_used) if plan.fields_used else "selected columns"
    parts = []

    if plan.chart_type == ChartType.bar and plan.encoding.y:
        y_field = plan.encoding.y.field
        if y_field and any(y_field in d for d in data[:1]):
            top_val = data[0]
            x_field = plan.encoding.x.field if plan.encoding.x else None
            if x_field and x_field in top_val:
                parts.append(f"To compare categories and identify leaders (top: {top_val.get(x_field)}).")

    if plan.chart_type == ChartType.scatter and plan.encoding.x and plan.encoding.y:
        parts.append(f"To assess relationship between {plan.encoding.x.field} and {plan.encoding.y.field}.")

    if plan.chart_type == ChartType.line and plan.encoding.x and plan.encoding.y:
        parts.append(f"To evaluate trends over {plan.encoding.x.field}.")

    if plan.chart_type == ChartType.hist:
        parts.append("To assess distribution shape and skew.")

    # Analytics explanations
    for tag, explanation in _ANALYTICS_EXPLANATIONS.items():
        if tag in plan.tags:
            parts.append(explanation)
            break

    if not parts:
        parts.append(f"Selected {ct} chart to examine {fields}.")

    return " ".join(parts)
