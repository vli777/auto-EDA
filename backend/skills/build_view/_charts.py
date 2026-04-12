"""
Standard chart builders: bar, line, scatter, histogram, box, table,
heatmap, pie, area, and the missingness overview chart.

Each builder takes a ViewPlan + DataFrame and returns pre-aggregated
data_inline rows ready for the frontend renderer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import ViewPlan
from core.utils import df_to_records_safe, smart_numeric_series
from ._utils import _coerce_numeric, _resolve


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------

def _aggregate_bar(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group-by aggregation for bar charts."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field:
        return []

    agg = (y_enc.aggregate or "").lower() if y_enc else ""

    if agg == "count" or y_field is None or y_field == x_field:
        grouped = df[x_field].value_counts().reset_index()
        grouped.columns = [x_field, "count"]
        y_col = "count"
    else:
        y_series = _coerce_numeric(df, y_field)
        tmp = df[[x_field]].copy()
        tmp["__y__"] = y_series
        agg_fn = agg if agg in ("sum", "mean", "median", "min", "max") else "mean"
        grouped = tmp.groupby(x_field, dropna=False)["__y__"].agg(agg_fn).reset_index()
        grouped.columns = [x_field, y_field]
        y_col = y_field

    if plan.options.sort == "descending":
        grouped = grouped.sort_values(y_col, ascending=False)
    elif plan.options.sort == "ascending":
        grouped = grouped.sort_values(y_col, ascending=True)
    else:
        grouped = grouped.sort_values(y_col, ascending=False)

    if plan.options.top_n:
        grouped = grouped.head(plan.options.top_n)

    return df_to_records_safe(grouped)


# ---------------------------------------------------------------------------
# Line / Area
# ---------------------------------------------------------------------------

def _aggregate_line(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Time-series or sequential line data."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field or not y_field:
        return []

    tmp = df[[x_field]].copy()
    tmp[y_field] = _coerce_numeric(df, y_field)
    tmp = tmp.dropna(subset=[y_field])

    color_enc = plan.encoding.color
    c_field = None
    if color_enc:
        c_field = _resolve(color_enc.field, df)
        if c_field and c_field in df.columns:
            tmp[c_field] = df.loc[tmp.index, c_field]

    if x_enc.type == "temporal":
        try:
            x_series = tmp[x_field]
            if pd.api.types.is_numeric_dtype(x_series):
                valid_x = x_series.dropna()
                if (not valid_x.empty
                        and valid_x.min() >= 1900 and valid_x.max() <= 2100):
                    tmp[x_field] = x_series.astype(int)
                else:
                    tmp[x_field] = pd.to_datetime(x_series, unit="s", errors="coerce")
                    tmp = tmp.dropna(subset=[x_field])
                    tmp[x_field] = tmp[x_field].dt.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                tmp[x_field] = pd.to_datetime(tmp[x_field], errors="coerce")
                tmp = tmp.dropna(subset=[x_field])
                tmp[x_field] = tmp[x_field].dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            tmp = tmp.sort_values(x_field)
        agg = (y_enc.aggregate or "mean").lower()
        if agg not in {"sum", "mean", "min", "max", "median", "count"}:
            agg = "mean"
        group_cols = [x_field]
        if c_field:
            group_cols.append(c_field)
        tmp = tmp.groupby(group_cols, dropna=False)[y_field].agg(agg).reset_index()
        tmp = tmp.sort_values(x_field)
    else:
        tmp = tmp.sort_values(x_field)

    if len(tmp) > 2000:
        tmp = tmp.head(2000)

    return df_to_records_safe(tmp)


def _build_area(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Area chart — same logic as line."""
    return _aggregate_line(plan, df)


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

def _build_scatter(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Scatter data — coerce numeric, drop NaN."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field or not y_field:
        return []

    tmp = df[[]].copy()
    tmp[x_field] = _coerce_numeric(df, x_field)
    tmp[y_field] = _coerce_numeric(df, y_field)

    color_enc = plan.encoding.color
    if color_enc:
        c_field = _resolve(color_enc.field, df)
        if c_field and c_field in df.columns:
            tmp[c_field] = df[c_field]

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_field, y_field])

    if plan.options.log:
        tmp = tmp[(tmp[x_field] > 0) & (tmp[y_field] > 0)]
        tmp[x_field] = np.log10(tmp[x_field])
        tmp[y_field] = np.log10(tmp[y_field])

    if len(tmp) > 5000:
        tmp = tmp.sample(5000, random_state=0)

    return df_to_records_safe(tmp)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def _build_histogram(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Histogram — bin numeric data into percentiles/deciles."""
    x_enc = plan.encoding.x
    if not x_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    if not x_field:
        return []

    series = _coerce_numeric(df, x_field).dropna()
    if series.empty:
        return []

    total = len(series)
    bin_count = plan.options.bin_count or 20
    records: List[Dict[str, Any]] = []

    try:
        p1 = float(np.percentile(series, 1))
        p99 = float(np.percentile(series, 99))
    except Exception:
        p1, p99 = float(series.min()), float(series.max())

    below = series[series < p1]
    above = series[series > p99]
    mid = series[(series >= p1) & (series <= p99)]

    if len(below) > 0:
        records.append({
            "bin_start": float(series.min()),
            "bin_end": p1,
            "bin_label": "<P1",
            "count": int(len(below)),
            "percent": round(100.0 * len(below) / total, 2),
        })

    mid_bins = max(1, bin_count - 2)
    counts, edges = np.histogram(mid, bins=mid_bins)
    for i, count in enumerate(counts):
        records.append({
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "bin_label": f"B{i+1}: {edges[i]:.6g}-{edges[i+1]:.6g}",
            "count": int(count),
            "percent": round(100.0 * count / total, 2),
        })

    if len(above) > 0:
        records.append({
            "bin_start": p99,
            "bin_end": float(series.max()),
            "bin_label": ">P99",
            "count": int(len(above)),
            "percent": round(100.0 * len(above) / total, 2),
        })

    try:
        markers = [
            {"value": float(np.percentile(series, p)), "label": f"P{p}"}
            for p in (10, 25, 50, 75, 90)
        ]
        if markers:
            plan.options.markers = markers
    except Exception:
        pass

    return records


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------

def _build_box(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Box plot stats — precompute quartiles, whiskers."""
    y_enc = plan.encoding.y
    x_enc = plan.encoding.x
    if not y_enc:
        return []

    y_field = _resolve(y_enc.field, df)
    if not y_field:
        return []

    series = _coerce_numeric(df, y_field)

    groups: List[tuple] = []
    if x_enc:
        x_field = _resolve(x_enc.field, df)
        if x_field and x_field in df.columns:
            for name, group in df.groupby(x_field, dropna=False):
                g_series = _coerce_numeric(group, y_field).dropna()
                if not g_series.empty:
                    groups.append((str(name), g_series))

    if not groups:
        valid = series.dropna()
        if not valid.empty:
            groups = [("all", valid)]

    records = []
    for label, g in groups:
        q1 = float(g.quantile(0.25))
        median = float(g.quantile(0.5))
        q3 = float(g.quantile(0.75))
        iqr = q3 - q1
        whisker_lo = float(g[g >= q1 - 1.5 * iqr].min())
        whisker_hi = float(g[g <= q3 + 1.5 * iqr].max())
        outliers = g[(g < whisker_lo) | (g > whisker_hi)].tolist()
        records.append({
            "group": label,
            "min": whisker_lo,
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": whisker_hi,
            "outliers": [float(o) for o in outliers[:50]],
        })

    return records


# ---------------------------------------------------------------------------
# Table / describe
# ---------------------------------------------------------------------------

def _build_describe_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build df.describe()-style summary with skew and kurtosis."""
    records: List[Dict[str, Any]] = []

    for col in df.columns:
        row: Dict[str, Any] = {"column": col, "dtype": str(df[col].dtype)}
        total = len(df)
        missing = int(df[col].isna().sum())
        row["count"] = total - missing
        row["missing"] = missing

        num = smart_numeric_series(df[col])
        valid = num.dropna()

        if len(valid) >= 2:
            row["mean"] = round(float(valid.mean()), 4)
            row["std"] = round(float(valid.std()), 4)
            row["min"] = round(float(valid.min()), 4)
            row["25%"] = round(float(valid.quantile(0.25)), 4)
            row["50%"] = round(float(valid.quantile(0.50)), 4)
            row["75%"] = round(float(valid.quantile(0.75)), 4)
            row["max"] = round(float(valid.max()), 4)
            row["skew"] = round(float(valid.skew()), 4)
            row["kurtosis"] = round(float(valid.kurtosis()), 4)
        else:
            nunique = df[col].nunique(dropna=True)
            row["unique"] = nunique
            top = df[col].value_counts().head(1)
            if not top.empty:
                row["top"] = str(top.index[0])[:60]
                row["freq"] = int(top.iloc[0])

        records.append(row)

    return records


def _build_table(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Summary table — df.describe()-style stats when tagged as summary."""
    if "summary" in plan.tags:
        return _build_describe_table(df)

    fields = plan.fields_used or list(df.columns[:10])
    resolved = [_resolve(f, df) for f in fields]
    cols = [c for c in resolved if c and c in df.columns]
    if not cols:
        cols = list(df.columns[:10])

    subset = df[cols].head(100)
    return df_to_records_safe(subset)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _build_heatmap(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Cross-tab heatmap between two categoricals."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field or not y_field:
        return []

    ct = pd.crosstab(df[y_field], df[x_field])
    records = []
    for row_label in ct.index:
        for col_label in ct.columns:
            records.append({
                x_field: str(col_label),
                y_field: str(row_label),
                "count": int(ct.loc[row_label, col_label]),
            })

    return records


# ---------------------------------------------------------------------------
# Pie
# ---------------------------------------------------------------------------

def _build_pie(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pie chart — aggregate category frequencies or use a value column."""
    color_enc = plan.encoding.color
    theta_enc = plan.encoding.theta

    cat_field = None
    if color_enc:
        cat_field = _resolve(color_enc.field, df)
    if not cat_field and plan.encoding.x:
        cat_field = _resolve(plan.encoding.x.field, df)

    if not cat_field:
        return []

    agg = (theta_enc.aggregate or "").lower() if theta_enc else ""

    if agg == "count" or not theta_enc or not theta_enc.field or theta_enc.field == cat_field:
        counts = df[cat_field].value_counts().reset_index()
        counts.columns = [cat_field, "count"]
        return df_to_records_safe(counts)
    else:
        val_field = _resolve(theta_enc.field, df)
        if val_field and val_field in df.columns:
            tmp = df[[cat_field]].copy()
            tmp["__val__"] = _coerce_numeric(df, val_field)
            agg_fn = agg if agg in ("sum", "mean") else "sum"
            grouped = tmp.groupby(cat_field, dropna=False)["__val__"].agg(agg_fn).reset_index()
            grouped.columns = [cat_field, val_field]
            return df_to_records_safe(grouped)

    return []


# ---------------------------------------------------------------------------
# Missingness overview
# ---------------------------------------------------------------------------

def _build_missingness(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Special builder for the missingness overview chart."""
    records = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        total = len(df)
        pct = round(100.0 * missing / total, 2) if total > 0 else 0.0
        if pct > 0:
            records.append({
                "__column__": col,
                "__missing_pct__": pct,
            })
    records.sort(key=lambda r: r["__missing_pct__"], reverse=True)
    return records
