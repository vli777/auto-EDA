"""
Shared utilities for the build_view package.

Used by both _charts.py (standard chart builders) and _analytics.py (query tool builders).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from core.models import ChartEncoding, ChartType, EncodingChannel
from core.utils import resolve_col, smart_numeric_series


# ---------------------------------------------------------------------------
# Encoding reconciliation
# ---------------------------------------------------------------------------

def _reconcile_encoding(
    chart_type: ChartType,
    encoding: ChartEncoding,
    data: List[Dict[str, Any]],
) -> ChartEncoding:
    """Fix encoding field references to match actual data column names.

    Builders sometimes produce columns with different names than what the
    plan's encoding declares (e.g. count aggregation creates a ``"count"``
    column while the encoding still says ``y.field = original_col``).  The
    frontend reads ``spec.encoding.*.field`` to map data keys, so a mismatch
    causes empty charts.
    """
    if not data or chart_type in (ChartType.hist, ChartType.box, ChartType.table):
        return encoding

    keys = set(data[0].keys())
    x_field = encoding.x.field if encoding.x else None

    new_y = encoding.y
    new_theta = encoding.theta
    new_color = encoding.color

    # y field: either missing from data, or same as x with count aggregation
    if encoding.y and encoding.y.field:
        y = encoding.y
        agg = (y.aggregate or "").lower()
        if y.field not in keys:
            replacement = _find_numeric_key(
                keys, exclude={x_field} if x_field else set(), data=data,
            )
            if replacement:
                new_y = EncodingChannel(
                    field=replacement, type=y.type,
                    aggregate=y.aggregate, bin=y.bin,
                )
        elif agg == "count" and y.field == x_field and "count" in keys:
            new_y = EncodingChannel(
                field="count", type=y.type,
                aggregate=y.aggregate, bin=y.bin,
            )

    # theta field: pie chart count aggregation
    if encoding.theta and encoding.theta.field:
        theta = encoding.theta
        agg = (theta.aggregate or "").lower()
        cat_field = encoding.color.field if encoding.color else None
        if theta.field not in keys:
            exclude = {cat_field} if cat_field and cat_field in keys else set()
            replacement = _find_numeric_key(keys, exclude=exclude, data=data)
            if replacement:
                new_theta = EncodingChannel(
                    field=replacement, type=theta.type,
                    aggregate=theta.aggregate, bin=theta.bin,
                )
        elif agg == "count" and theta.field == cat_field and "count" in keys:
            new_theta = EncodingChannel(
                field="count", type=theta.type,
                aggregate=theta.aggregate, bin=theta.bin,
            )

    # color (quantitative) field not in data — heatmap count
    if (encoding.color and encoding.color.field
            and encoding.color.type == "quantitative"
            and encoding.color.field not in keys):
        if "count" in keys:
            new_color = EncodingChannel(
                field="count", type=encoding.color.type,
                aggregate=encoding.color.aggregate,
            )

    if new_y is encoding.y and new_theta is encoding.theta and new_color is encoding.color:
        return encoding

    return ChartEncoding(
        x=encoding.x, y=new_y, color=new_color,
        theta=new_theta, facet=encoding.facet, size=encoding.size,
    )


def _find_numeric_key(
    keys: set, exclude: set, data: List[Dict[str, Any]],
) -> Optional[str]:
    """Find a numeric column in data, preferring ``"count"``."""
    if "count" in keys and "count" not in exclude:
        return "count"
    for k in keys:
        if k not in exclude and isinstance(data[0].get(k), (int, float)):
            return k
    return None


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def _resolve(field: Optional[str], df: pd.DataFrame) -> Optional[str]:
    """Resolve a field name against the DataFrame, returning None if not found."""
    if field is None:
        return None
    r = resolve_col(field, df)
    return r if r and r in df.columns else None


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(df[col]):
        return pd.to_numeric(df[col], errors="coerce")
    return smart_numeric_series(df[col])


def _safe_stats(series: pd.Series) -> Dict[str, Any]:
    valid = series.dropna()
    if valid.empty:
        return {}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
    }


# ---------------------------------------------------------------------------
# Tag / time helpers (used by analytics builders)
# ---------------------------------------------------------------------------

def _tag_value(tags: List[str], key: str) -> Optional[str]:
    prefix = f"{key}="
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def _parse_percentile_ranges(tag_value: Optional[str]) -> List[tuple[int, int]]:
    if not tag_value:
        return []
    ranges = []
    for part in tag_value.split(";"):
        part = part.strip()
        if not part or "-" not in part:
            continue
        lo_str, hi_str = part.split("-", 1)
        try:
            lo = int(lo_str)
            hi = int(hi_str)
        except ValueError:
            continue
        if lo < 0 or hi > 100:
            continue
        if lo > hi:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def _coerce_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return series
