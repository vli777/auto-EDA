"""
Analytics / query-tool builders.

Each function corresponds to one of the tool names in query_planner._TOOL_NAMES.
They receive a ViewPlan (whose tags carry runtime parameters) and a DataFrame,
and return pre-computed data_inline rows.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.models import ViewPlan
from core.utils import df_to_records_safe, resolve_col, smart_numeric_series
from ._utils import _coerce_time, _parse_percentile_ranges, _tag_value


# ---------------------------------------------------------------------------
# Percentile comparison
# ---------------------------------------------------------------------------

def _build_percentile_compare(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compare percentile groups for a metric, optionally across time."""
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    entity = resolve_col(_tag_value(plan.tags, "entity"), df)
    compare = _tag_value(plan.tags, "compare") or ""
    ranges = _parse_percentile_ranges(_tag_value(plan.tags, "percentiles"))

    if not metric or metric not in df.columns or not ranges:
        return []

    base = df[[metric]].copy()
    base[metric] = smart_numeric_series(base[metric])
    base = base.dropna(subset=[metric])
    if base.empty:
        return []

    total = len(base)
    values = base[metric]

    def group_stats(series: pd.Series, label: str) -> Dict[str, Any]:
        s = series.dropna()
        if s.empty:
            return {"group": label, "count": 0, "mean": None, "median": None, "pct_of_rows": 0.0}
        return {
            "group": label,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "pct_of_rows": round(100.0 * len(s) / max(1, total), 2),
        }

    if compare == "change" and temporal and temporal in df.columns:
        work_cols = [metric, temporal]
        if entity and entity in df.columns:
            work_cols.append(entity)
        work = df[work_cols].copy()
        work[metric] = smart_numeric_series(work[metric])
        work = work.dropna(subset=[metric])
        if work.empty:
            return []
        work["_time"] = _coerce_time(work[temporal])
        work = work.dropna(subset=["_time"])
        if work.empty:
            return []

        records: List[Dict[str, Any]] = []
        if entity and entity in work.columns:
            idx_min = work.groupby(entity)["_time"].idxmin()
            idx_max = work.groupby(entity)["_time"].idxmax()
            base_df = work.loc[idx_min, [entity, metric]].set_index(entity)
            last_df = work.loc[idx_max, [entity, metric]].set_index(entity)
            joined = base_df.join(last_df, lsuffix="_base", rsuffix="_last", how="inner")
            if joined.empty:
                return []
            joined["change"] = joined[f"{metric}_last"] - joined[f"{metric}_base"]
            metric_series = joined[f"{metric}_base"]
            for lo, hi in ranges:
                lo_v = float(np.percentile(metric_series, lo))
                hi_v = float(np.percentile(metric_series, hi))
                mask = (metric_series >= lo_v) & (metric_series <= hi_v)
                subset = joined.loc[mask]
                stats = group_stats(subset["change"], f"P{lo}-{hi}")
                stats["mean_baseline"] = round(float(subset[f"{metric}_base"].mean()), 4) if len(subset) else None
                stats["mean_latest"] = round(float(subset[f"{metric}_last"].mean()), 4) if len(subset) else None
                stats["mean_change"] = stats.pop("mean")
                stats["median_change"] = stats.pop("median")
                records.append(stats)
            return records

        time_vals = work["_time"].dropna().sort_values()
        if time_vals.empty:
            return []
        first_t = time_vals.iloc[0]
        last_t = time_vals.iloc[-1]
        base_t = work[work["_time"] == first_t][metric].dropna()
        last_t_series = work[work["_time"] == last_t][metric].dropna()
        if base_t.empty or last_t_series.empty:
            return []
        for lo, hi in ranges:
            lo_v = float(np.percentile(base_t, lo))
            hi_v = float(np.percentile(base_t, hi))
            base_subset = base_t[(base_t >= lo_v) & (base_t <= hi_v)]
            last_subset = last_t_series[(last_t_series >= lo_v) & (last_t_series <= hi_v)]
            stats = group_stats(last_subset - base_subset.mean(), f"P{lo}-{hi}")
            stats["mean_baseline"] = round(float(base_subset.mean()), 4) if len(base_subset) else None
            stats["mean_latest"] = round(float(last_subset.mean()), 4) if len(last_subset) else None
            stats["mean_change"] = stats.pop("mean")
            stats["median_change"] = stats.pop("median")
            records.append(stats)
        return records

    records = []
    for lo, hi in ranges:
        lo_v = float(np.percentile(values, lo))
        hi_v = float(np.percentile(values, hi))
        subset = values[(values >= lo_v) & (values <= hi_v)]
        records.append(group_stats(subset, f"P{lo}-{hi}"))
    return records


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _build_linear_regression(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    x_field = _tag_value(plan.tags, "x") or (plan.encoding.x.field if plan.encoding.x else None)
    y_field = _tag_value(plan.tags, "y") or (plan.encoding.y.field if plan.encoding.y else None)
    x_field = resolve_col(x_field, df)
    y_field = resolve_col(y_field, df)
    if not x_field or not y_field:
        return []
    if x_field not in df.columns or y_field not in df.columns:
        return []

    x = smart_numeric_series(df[x_field])
    y = smart_numeric_series(df[y_field])
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return []

    X = np.column_stack([np.ones(len(x)), x.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    y_pred = X @ beta
    ss_res = float(np.sum((y.to_numpy() - y_pred) ** 2))
    ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return [
        {"type": "coef", "term": "intercept", "value": round(float(beta[0]), 6)},
        {"type": "coef", "term": x_field, "value": round(float(beta[1]), 6)},
        {"type": "metric", "term": "r2", "value": round(r2, 6)},
        {"type": "metric", "term": "n", "value": int(len(x))},
    ]


def _build_logistic_regression(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    x_field = _tag_value(plan.tags, "x") or (plan.encoding.x.field if plan.encoding.x else None)
    y_field = _tag_value(plan.tags, "y") or (plan.encoding.y.field if plan.encoding.y else None)
    x_field = resolve_col(x_field, df)
    y_field = resolve_col(y_field, df)
    if not x_field or not y_field:
        return []
    if x_field not in df.columns or y_field not in df.columns:
        return []

    x = smart_numeric_series(df[x_field])
    y_raw = df[y_field]
    y_unique = y_raw.dropna().unique().tolist()
    if len(y_unique) != 2:
        return []
    y_map = {y_unique[0]: 0, y_unique[1]: 1}
    y = y_raw.map(y_map)

    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return []

    x_mean = float(x.mean())
    x_std = float(x.std()) or 1.0
    xz = (x - x_mean) / x_std

    w0, w1 = 0.0, 0.0
    lr = 0.2
    for _ in range(400):
        z = w0 + w1 * xz.to_numpy()
        p = _sigmoid(z)
        grad0 = float((p - y.to_numpy()).mean())
        grad1 = float(((p - y.to_numpy()) * xz.to_numpy()).mean())
        w0 -= lr * grad0
        w1 -= lr * grad1

    z = w0 + w1 * xz.to_numpy()
    p = _sigmoid(z)
    preds = (p >= 0.5).astype(int)
    acc = float((preds == y.to_numpy()).mean())

    return [
        {"type": "coef", "term": "intercept", "value": round(float(w0), 6)},
        {"type": "coef", "term": x_field, "value": round(float(w1), 6), "note": "standardized x"},
        {"type": "metric", "term": "accuracy", "value": round(acc, 6)},
        {"type": "metric", "term": "n", "value": int(len(x))},
        {"type": "metric", "term": "positive_label", "value": str(y_unique[1])[:50]},
    ]


def _build_quantile_regression(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    x_field = resolve_col(_tag_value(plan.tags, "x"), df)
    y_field = resolve_col(_tag_value(plan.tags, "y"), df)
    if not x_field or not y_field:
        return []
    if x_field not in df.columns or y_field not in df.columns:
        return []

    try:
        import statsmodels.api as sm
    except Exception:
        return []

    x = smart_numeric_series(df[x_field])
    y = smart_numeric_series(df[y_field])
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]
    if len(x) < 20:
        return []

    X = sm.add_constant(x.to_numpy())
    rows: List[Dict[str, Any]] = []
    for q in (0.1, 0.5, 0.9):
        try:
            model = sm.QuantReg(y.to_numpy(), X)
            res = model.fit(q=q)
            rows.append({"quantile": q, "term": "intercept", "value": float(res.params[0])})
            rows.append({"quantile": q, "term": x_field, "value": float(res.params[1])})
        except Exception:
            continue
    return rows


# ---------------------------------------------------------------------------
# ML models (LightGBM)
# ---------------------------------------------------------------------------

def _build_lgbm_regression(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
    except Exception:
        return []

    work = df.copy()
    y = smart_numeric_series(work[target])
    work = work.drop(columns=[target])
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    if not numeric_cols:
        return []
    X = work[numeric_cols].copy().apply(smart_numeric_series)
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    if len(y) < 30:
        return []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = lgb.LGBMRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    importances = model.feature_importances_
    rows = [
        {"metric": "rmse", "value": round(float(rmse), 6)},
        {"metric": "r2", "value": round(float(r2), 6)},
        {"metric": "n_train", "value": int(len(y_train))},
        {"metric": "n_test", "value": int(len(y_test))},
    ]
    top = sorted(zip(numeric_cols, importances), key=lambda t: t[1], reverse=True)[:10]
    for name, score in top:
        rows.append({"feature": name, "importance": float(score)})
    return rows


def _build_lgbm_classification(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.preprocessing import LabelEncoder
    except Exception:
        return []

    y_raw = df[target].dropna()
    if y_raw.nunique() < 2:
        return []

    work = df.copy()
    y = work[target]
    work = work.drop(columns=[target])
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    if not numeric_cols:
        return []
    X = work[numeric_cols].copy().apply(smart_numeric_series)
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    if len(y) < 30:
        return []

    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=0)
    model = lgb.LGBMClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rows = [
        {"metric": "accuracy", "value": round(float(acc), 6)},
        {"metric": "n_train", "value": int(len(y_train))},
        {"metric": "n_test", "value": int(len(y_test))},
    ]

    if len(le.classes_) == 2:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            rows.append({"metric": "auc", "value": round(float(auc), 6)})
        except Exception:
            pass

    importances = model.feature_importances_
    top = sorted(zip(numeric_cols, importances), key=lambda t: t[1], reverse=True)[:10]
    for name, score in top:
        rows.append({"feature": name, "importance": float(score)})
    return rows


# ---------------------------------------------------------------------------
# Segmentation / group analysis
# ---------------------------------------------------------------------------

def _build_segmentation(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    group = resolve_col(_tag_value(plan.tags, "group"), df)
    if not metric or not group:
        return []
    if metric not in df.columns or group not in df.columns:
        return []

    work = df[[metric, group]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work = work.dropna(subset=[metric])
    if work.empty:
        return []

    total = len(work)
    agg = (
        work.groupby(group)[metric]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
    )
    agg["pct_of_rows"] = agg["count"].apply(lambda v: round(100.0 * v / max(1, total), 2))
    agg = agg.sort_values("mean", ascending=False)
    limit = plan.options.top_n or 15
    return df_to_records_safe(agg.head(limit))


def _build_cohort_change(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    entity = resolve_col(_tag_value(plan.tags, "entity"), df)
    if not metric or not temporal or not entity:
        return []
    if metric not in df.columns or temporal not in df.columns or entity not in df.columns:
        return []

    work = df[[metric, temporal, entity]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work = work.dropna(subset=[metric])
    if work.empty:
        return []

    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=["_time"])
    if work.empty:
        return []

    work["_period"] = work["_time"].dt.to_period("M").dt.to_timestamp()
    first_period = work.groupby(entity)["_period"].min().rename("cohort")
    work = work.join(first_period, on=entity)

    idx_min = work.groupby(entity)["_period"].idxmin()
    idx_max = work.groupby(entity)["_period"].idxmax()
    base = work.loc[idx_min, [entity, metric, "cohort"]].set_index(entity)
    last = work.loc[idx_max, [entity, metric]].set_index(entity)
    joined = base.join(last, lsuffix="_base", rsuffix="_last", how="inner")
    if joined.empty:
        return []
    joined["change"] = joined[f"{metric}_last"] - joined[f"{metric}_base"]

    agg = (
        joined.groupby("cohort")["change"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("cohort")
    )
    agg["cohort"] = agg["cohort"].astype(str)
    return df_to_records_safe(agg)


def _build_group_comparison(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    group = resolve_col(_tag_value(plan.tags, "group"), df)
    if not metric or not group:
        return []
    if metric not in df.columns or group not in df.columns:
        return []

    work = df[[metric, group]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work = work.dropna(subset=[metric, group])
    if work.empty:
        return []

    counts = work[group].value_counts().head(2)
    if len(counts) < 2:
        return []

    g1, g2 = counts.index[0], counts.index[1]
    s1 = work[work[group] == g1][metric]
    s2 = work[work[group] == g2][metric]
    if s1.empty or s2.empty:
        return []

    mean1 = float(s1.mean())
    mean2 = float(s2.mean())
    diff = mean1 - mean2
    pooled = float(np.sqrt((s1.var(ddof=1) + s2.var(ddof=1)) / 2.0)) if (len(s1) > 1 and len(s2) > 1) else 0.0
    d = diff / pooled if pooled else 0.0

    return [
        {"group": str(g1), "mean": round(mean1, 6), "n": int(len(s1))},
        {"group": str(g2), "mean": round(mean2, 6), "n": int(len(s2))},
        {"metric": "diff_in_means", "value": round(diff, 6), "note": "naive (not causal)"},
        {"metric": "cohens_d", "value": round(d, 6)},
    ]


def _build_matched_comparison(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    treatment = resolve_col(_tag_value(plan.tags, "treatment"), df)
    if not metric or not treatment:
        return []
    work = df.copy()
    y = smart_numeric_series(work[metric])
    t = work[treatment]
    covars = [c for c in work.columns if c not in (metric, treatment) and pd.api.types.is_numeric_dtype(work[c])]
    if not covars:
        return []
    X = work[covars].apply(smart_numeric_series)
    mask = (~y.isna()) & (~t.isna())
    X = X[mask]; y = y[mask]; t = t[mask]
    if t.nunique() != 2 or len(y) < 30:
        return []
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return []
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    treated_mask = t.astype("string") == str(t.unique()[0])
    X_t = Xs[treated_mask]; X_c = Xs[~treated_mask]
    y_t = y[treated_mask]; y_c = y[~treated_mask]
    if len(y_t) < 5 or len(y_c) < 5:
        return []
    nn = NearestNeighbors(n_neighbors=1).fit(X_c)
    dist, idx = nn.kneighbors(X_t)
    matched_y_c = y_c.iloc[idx.flatten()].reset_index(drop=True)
    att = float((y_t.reset_index(drop=True) - matched_y_c).mean())
    return [
        {"metric": "att", "value": round(att, 6), "note": "naive matching"},
        {"metric": "n_treated", "value": int(len(y_t))},
        {"metric": "n_control", "value": int(len(y_c))},
    ]


def _build_diff_in_diff(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    treatment = resolve_col(_tag_value(plan.tags, "treatment"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not treatment or not temporal:
        return []
    work = df[[metric, treatment, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, treatment, "_time"])
    if work.empty:
        return []
    cut = work["_time"].median()
    work["post"] = work["_time"] >= cut
    groups = work.groupby([treatment, "post"])[metric].mean().reset_index()
    if groups[treatment].nunique() != 2:
        return []
    t_vals = groups[treatment].unique().tolist()
    pre_t  = groups[(groups[treatment] == t_vals[0]) & (~groups["post"])][metric].mean()
    post_t = groups[(groups[treatment] == t_vals[0]) & ( groups["post"])][metric].mean()
    pre_c  = groups[(groups[treatment] == t_vals[1]) & (~groups["post"])][metric].mean()
    post_c = groups[(groups[treatment] == t_vals[1]) & ( groups["post"])][metric].mean()
    did = float((post_t - pre_t) - (post_c - pre_c))
    return [
        {"group": str(t_vals[0]), "pre_mean": round(float(pre_t), 6), "post_mean": round(float(post_t), 6)},
        {"group": str(t_vals[1]), "pre_mean": round(float(pre_c), 6), "post_mean": round(float(post_c), 6)},
        {"metric": "diff_in_diff", "value": round(did, 6), "note": "naive"},
    ]


def _build_uplift_check(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    treatment = resolve_col(_tag_value(plan.tags, "treatment"), df)
    target    = resolve_col(_tag_value(plan.tags, "target"), df)
    segment   = resolve_col(_tag_value(plan.tags, "segment"), df)
    if not treatment or not target:
        return []
    work = df[[treatment, target] + ([segment] if segment else [])].copy()
    work = work.dropna(subset=[treatment, target])
    if work.empty:
        return []
    t_vals = work[treatment].astype("string").unique().tolist()
    if len(t_vals) != 2:
        return []
    treated, control = t_vals[0], t_vals[1]

    def rate(s: pd.Series) -> float:
        s_num = smart_numeric_series(s)
        if s_num.dropna().empty:
            return float((s.astype("string") == treated).mean())
        return float(s_num.mean())

    rows: List[Dict[str, Any]] = []
    if segment and segment in work.columns:
        for val, grp in work.groupby(segment):
            rt = rate(grp[grp[treatment].astype("string") == treated][target])
            rc = rate(grp[grp[treatment].astype("string") == control][target])
            rows.append({"segment": str(val), "uplift": round(rt - rc, 6)})
        return rows[:10]

    rt = rate(work[work[treatment].astype("string") == treated][target])
    rc = rate(work[work[treatment].astype("string") == control][target])
    return [{"uplift": round(rt - rc, 6), "treated": treated, "control": control}]


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def _build_seasonality_test(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not temporal:
        return []
    if metric not in df.columns or temporal not in df.columns:
        return []
    work = df[[metric, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, "_time"]).sort_values("_time")
    if work.empty:
        return []
    series = work.set_index("_time")[metric].resample("M").mean().dropna()
    if len(series) < 6:
        return []
    try:
        from statsmodels.tsa.stattools import acf
    except Exception:
        return []
    vals = acf(series.values, nlags=min(24, len(series) - 1), fft=False)
    rows = [{"lag": lag, "acf": round(float(vals[lag]), 6)} for lag in range(1, min(13, len(vals)))]
    rows.sort(key=lambda r: abs(r["acf"]), reverse=True)
    return rows[:5]


def _build_autocorrelation_test(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not temporal:
        return []
    if metric not in df.columns or temporal not in df.columns:
        return []
    work = df[[metric, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, "_time"]).sort_values("_time")
    series = work[metric].values
    if len(series) < 10:
        return []
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except Exception:
        return []
    max_lag = min(10, len(series) - 1)
    acf_vals  = acf(series,  nlags=max_lag, fft=False)
    pacf_vals = pacf(series, nlags=max_lag, method="yw")
    return [
        {"lag": lag, "acf": round(float(acf_vals[lag]), 6), "pacf": round(float(pacf_vals[lag]), 6)}
        for lag in range(1, max_lag + 1)
    ]


def _build_lag_feature_search(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not temporal:
        return []
    work = df[[metric, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, "_time"]).sort_values("_time")
    if len(work) < 20:
        return []
    series = work[metric].values
    rows = []
    for lag in range(1, min(13, len(series) // 3)):
        corr = np.corrcoef(series[lag:], series[:-lag])[0, 1]
        rows.append({"lag": lag, "corr": round(float(corr), 6)})
    rows.sort(key=lambda r: abs(r["corr"]), reverse=True)
    return rows[:6]


def _build_rolling_stats(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not temporal:
        return []
    work = df[[metric, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, "_time"]).sort_values("_time")
    if len(work) < 10:
        return []
    series = work.set_index("_time")[metric]
    rows = []
    for window in (7, 30):
        roll = series.rolling(window=window, min_periods=max(2, window // 3))
        mean_val = roll.mean().iloc[-1]
        std_val  = roll.std().iloc[-1]
        rows.append({
            "window": window,
            "rolling_mean": round(float(mean_val), 6) if pd.notna(mean_val) else None,
            "rolling_std":  round(float(std_val),  6) if pd.notna(std_val)  else None,
        })
    return rows


def _build_trend_breaks(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not temporal:
        return []
    work = df[[metric, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, "_time"]).sort_values("_time")
    if len(work) < 10:
        return []
    n = len(work)
    thirds = [work.iloc[: n // 3], work.iloc[n // 3: 2 * n // 3], work.iloc[2 * n // 3:]]
    rows = []
    for i, seg in enumerate(thirds, start=1):
        if seg.empty:
            continue
        rows.append({"segment": i, "mean": round(float(seg[metric].mean()), 6), "n": int(len(seg))})
    if len(rows) >= 2:
        rows.append({"metric": "delta_seg3_seg1", "value": round(float(rows[-1]["mean"] - rows[0]["mean"]), 6)})
    return rows


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _build_numeric_transforms(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    if not metric or metric not in df.columns:
        return []
    s = smart_numeric_series(df[metric]).dropna()
    if len(s) < 10:
        return []

    def skew(x: pd.Series) -> float:
        return float(x.skew()) if len(x) > 2 else 0.0

    rows = [{"transform": "none", "skew": round(skew(s), 6)}]
    if (s >= 0).all():
        rows.append({"transform": "log1p", "skew": round(skew(np.log1p(s)), 6)})
        rows.append({"transform": "sqrt",  "skew": round(skew(np.sqrt(s)),  6)})
    return rows


def _build_interaction_scan(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    y = smart_numeric_series(df[target])
    features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if len(features) < 2:
        return []
    rows = []
    for i in range(min(5, len(features))):
        for j in range(i + 1, min(6, len(features))):
            a = smart_numeric_series(df[features[i]])
            b = smart_numeric_series(df[features[j]])
            inter = a * b
            mask = (~y.isna()) & (~inter.isna())
            if mask.sum() < 20:
                continue
            corr = np.corrcoef(inter[mask], y[mask])[0, 1]
            rows.append({"feature_pair": f"{features[i]}*{features[j]}", "corr_with_target": round(float(corr), 6)})
    rows.sort(key=lambda r: abs(r["corr_with_target"]), reverse=True)
    return rows[:10]


def _build_binning_optimizer(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target  = resolve_col(_tag_value(plan.tags, "target"), df)
    feature = resolve_col(_tag_value(plan.tags, "feature"), df)
    if not target or not feature:
        return []
    x = smart_numeric_series(df[feature])
    y = smart_numeric_series(df[target])
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]; y = y[mask]
    if len(x) < 20:
        return []
    bins = pd.qcut(x.rank(method="first"), q=5, duplicates="drop")
    agg = pd.DataFrame({"bin": bins, "target": y}).groupby("bin")["target"].mean().reset_index()
    agg["bin"] = agg["bin"].astype(str)
    return df_to_records_safe(agg)


def _build_date_part_features(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not metric or not temporal:
        return []
    work = df[[metric, temporal]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=[metric, "_time"])
    if work.empty:
        return []
    work["month"] = work["_time"].dt.month
    work["dow"]   = work["_time"].dt.dayofweek
    by_month = work.groupby("month")[metric].mean().reset_index(); by_month["part"] = "month"
    by_dow   = work.groupby("dow")[metric].mean().reset_index();   by_dow["part"]   = "dow"
    return df_to_records_safe(pd.concat([by_month, by_dow], ignore_index=True))


def _build_target_encoding(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target  = resolve_col(_tag_value(plan.tags, "target"), df)
    feature = resolve_col(_tag_value(plan.tags, "feature"), df)
    if not target or not feature:
        return []
    work = df[[target, feature]].copy()
    work[target] = smart_numeric_series(work[target])
    work = work.dropna(subset=[feature])
    if work.empty:
        return []
    agg = work.groupby(feature)[target].agg(["count", "mean"]).reset_index()
    agg = agg.sort_values("mean", ascending=False).head(plan.options.top_n or 15)
    return df_to_records_safe(agg)


# ---------------------------------------------------------------------------
# SHAP / partial dependence
# ---------------------------------------------------------------------------

def _build_shap_summary(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    try:
        import lightgbm as lgb
        import shap
    except Exception:
        return []
    work = df.copy()
    y = work[target]; work = work.drop(columns=[target])
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    if not numeric_cols:
        return []
    X = work[numeric_cols].apply(smart_numeric_series)
    mask = ~y.isna(); X = X[mask]; y = y[mask]
    if len(y) < 50:
        return []
    X = X.sample(min(len(X), 2000), random_state=0); y = y.loc[X.index]
    is_class = y.nunique() <= 2
    model = lgb.LGBMClassifier(n_estimators=200, random_state=0) if is_class else lgb.LGBMRegressor(n_estimators=200, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return [
        {"feature": name, "mean_abs_shap": round(float(val), 6)}
        for name, val in sorted(zip(numeric_cols, mean_abs), key=lambda t: t[1], reverse=True)[:15]
    ]


def _build_shap_dependence(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    try:
        import lightgbm as lgb
        import shap
    except Exception:
        return []
    work = df.copy()
    y = work[target]; work = work.drop(columns=[target])
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    if not numeric_cols:
        return []
    X = work[numeric_cols].apply(smart_numeric_series)
    mask = ~y.isna(); X = X[mask]; y = y[mask]
    if len(y) < 50:
        return []
    X = X.sample(min(len(X), 1000), random_state=0); y = y.loc[X.index]
    model = lgb.LGBMRegressor(n_estimators=200, random_state=0) if y.nunique() > 2 else lgb.LGBMClassifier(n_estimators=200, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    feature = numeric_cols[top_idx]
    vals = X[feature].values; svals = shap_vals[:, top_idx]
    corr = float(np.corrcoef(vals, svals)[0, 1]) if len(vals) > 2 else 0.0
    return [{"feature": feature, "shap_mean": round(float(svals.mean()), 6), "shap_std": round(float(svals.std()), 6), "corr": round(corr, 6)}]


def _build_partial_dependence(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target  = resolve_col(_tag_value(plan.tags, "target"), df)
    feature = resolve_col(_tag_value(plan.tags, "feature"), df)
    if not target or not feature:
        return []
    try:
        import lightgbm as lgb
        from sklearn.inspection import partial_dependence
    except Exception:
        return []
    work = df.copy()
    y = work[target]; work = work.drop(columns=[target])
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    if feature not in numeric_cols:
        return []
    X = work[numeric_cols].apply(smart_numeric_series)
    mask = ~y.isna(); X = X[mask]; y = y[mask]
    if len(y) < 50:
        return []
    model = lgb.LGBMRegressor(n_estimators=200, random_state=0) if y.nunique() > 2 else lgb.LGBMClassifier(n_estimators=200, random_state=0)
    model.fit(X, y)
    pdp = partial_dependence(model, X, [feature], grid_resolution=20)
    return [{"feature_value": float(g), "partial_dep": float(v)} for g, v in zip(pdp["values"][0], pdp["average"][0])]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _build_leakage_scan(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    y = smart_numeric_series(df[target])
    rows = []
    for col in df.columns:
        if col == target or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = smart_numeric_series(df[col])
        mask = (~x.isna()) & (~y.isna())
        if mask.sum() < 20:
            continue
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        rows.append({"feature": col, "corr": round(float(corr), 6)})
    rows.sort(key=lambda r: abs(r["corr"]), reverse=True)
    return rows[:15]


def _build_drift_check(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    if not temporal or temporal not in df.columns:
        return []
    work = df.copy()
    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=["_time"])
    if work.empty:
        return []
    cut = work["_time"].median()
    early = work[work["_time"] <  cut]
    late  = work[work["_time"] >= cut]
    rows = []
    for col in work.columns:
        if col in (temporal, "_time"):
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            e = smart_numeric_series(early[col]).dropna()
            l = smart_numeric_series(late[col]).dropna()
            if e.empty or l.empty:
                continue
            diff = float(l.mean() - e.mean())
            std  = float(e.std() or 1.0)
            rows.append({"feature": col, "mean_shift": round(diff, 6), "std_units": round(diff / std, 6)})
    rows.sort(key=lambda r: abs(r["std_units"]), reverse=True)
    return rows[:15]


def _build_missingness_mechanism(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    target = resolve_col(_tag_value(plan.tags, "target"), df)
    if not target or target not in df.columns:
        return []
    y = smart_numeric_series(df[target])
    rows = []
    for col in df.columns:
        if col == target:
            continue
        miss = df[col].isna()
        if miss.sum() == 0:
            continue
        y_miss = y[miss]; y_obs = y[~miss]
        if y_miss.dropna().empty or y_obs.dropna().empty:
            continue
        diff = float(y_miss.mean() - y_obs.mean())
        rows.append({"feature": col, "diff_in_target_mean": round(diff, 6), "missing_pct": round(100.0 * miss.mean(), 2)})
    rows.sort(key=lambda r: abs(r["diff_in_target_mean"]), reverse=True)
    return rows[:15]


# ---------------------------------------------------------------------------
# Hypothesis / meta tools
# ---------------------------------------------------------------------------

def _build_hypothesis_generator(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    numeric     = [c for c in df.columns if     pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    if numeric and categorical:
        rows.append({"hypothesis": f"{numeric[0]} differs by {categorical[0]}."})
    if len(numeric) >= 2:
        rows.append({"hypothesis": f"{numeric[0]} increases with {numeric[1]}."})
    if numeric:
        rows.append({"hypothesis": f"{numeric[0]} shows outliers or heavy skew."})
    return rows


def _build_test_selector(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    q = _tag_value(plan.tags, "query") or ""
    tokens = set(re.findall(r"[a-z0-9]+", q.lower()))
    tools = []
    if tokens & {"seasonality", "lag", "autocorr"}:
        tools.append("seasonality_test")
    if tokens & {"regression", "predict"}:
        tools.append("lgbm_regression")
    if tokens & {"uplift", "treatment"}:
        tools.append("uplift_check")
    if not tools:
        tools.append("generic_query_charts")
    return [{"recommended_tool": t} for t in tools]


def _build_result_validator(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    q = _tag_value(plan.tags, "query") or ""
    return [{"status": "needs_evidence", "note": "Run a focused tool and compare effect sizes.", "query": q}]
