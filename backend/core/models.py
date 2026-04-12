"""
Core Pydantic models for the auto-EDA engine.

All domain types live here so every module shares the same vocabulary.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Column & Profile
# ---------------------------------------------------------------------------

class ColumnRole(str, Enum):
    temporal = "temporal"
    geographic = "geographic"
    measure = "measure"
    count = "count"
    categorical = "categorical"
    identifier = "identifier"
    nominal = "nominal"


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: ColumnRole = ColumnRole.nominal
    cardinality: int = 0
    missing_pct: float = 0.0
    stats: Optional[Dict[str, Any]] = None        # min/max/mean/std/quartiles
    top_values: Optional[List[Dict[str, Any]]] = None
    examples: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class DataProfile(BaseModel):
    table_name: str
    row_count: int
    columns: List[ColumnInfo]
    sample_rows: Optional[List[Dict[str, Any]]] = None
    visualization_hints: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Chart specification DSL (replaces Vega-Lite)
# ---------------------------------------------------------------------------

class ChartType(str, Enum):
    bar = "bar"
    line = "line"
    scatter = "scatter"
    hist = "hist"
    box = "box"
    table = "table"
    heatmap = "heatmap"
    pie = "pie"
    area = "area"


class EncodingChannel(BaseModel):
    field: str
    type: Optional[str] = None           # quantitative, nominal, temporal, ordinal
    aggregate: Optional[str] = None      # sum, mean, count, min, max, median
    bin: Optional[bool] = None
    time_unit: Optional[str] = None      # year, yearmonth, yearmonthdate
    sort: Optional[str] = None           # ascending, descending, field ref


class ChartEncoding(BaseModel):
    x: Optional[EncodingChannel] = None
    y: Optional[EncodingChannel] = None
    color: Optional[EncodingChannel] = None
    facet: Optional[EncodingChannel] = None
    size: Optional[EncodingChannel] = None
    theta: Optional[EncodingChannel] = None


class ChartOptions(BaseModel):
    sort: Optional[str] = None            # field or direction
    top_n: Optional[int] = None
    log: Optional[bool] = None
    bin_count: Optional[int] = None
    tooltip_fields: Optional[List[str]] = None
    orientation: Optional[str] = None     # horizontal / vertical
    stacked: Optional[bool] = None
    markers: Optional[List[Dict[str, Any]]] = None


class ChartSpec(BaseModel):
    chart_type: ChartType
    encoding: ChartEncoding
    options: ChartOptions = Field(default_factory=ChartOptions)
    data_inline: Optional[List[Dict[str, Any]]] = None
    title: str = ""
    subtitle: Optional[str] = None


# ---------------------------------------------------------------------------
# View plan & result
# ---------------------------------------------------------------------------

class ViewPlan(BaseModel):
    chart_type: ChartType
    encoding: ChartEncoding
    options: ChartOptions = Field(default_factory=ChartOptions)
    intent: str = ""                      # human description of what the chart shows
    fields_used: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)  # e.g. ["overview", "distribution"]


class ViewResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plan: ViewPlan
    spec: ChartSpec
    data_inline: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    fallback_repaired: bool = False   # True when plan was invalid and deterministic fallback was applied
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
# EDA steps & report
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    summary_stats = "summary_stats"
    analysis_intents = "analysis_intents"
    intent_views = "intent_views"
    quality_overview = "quality_overview"
    relationships = "relationships"
    outliers_segments = "outliers_segments"
    query_driven = "query_driven"


class StepResult(BaseModel):
    step_type: StepType
    headline: str = ""
    views: List[str] = Field(default_factory=list)        # view IDs
    findings: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    decision_trace: Optional[str] = None


class DecisionRecord(BaseModel):
    hypothesis: str = ""
    evidence: List[str] = Field(default_factory=list)
    decision: str = ""
    next_actions: List[str] = Field(default_factory=list)


class AnalysisIntent(BaseModel):
    title: str
    rationale: str = ""
    fields: List[str] = Field(default_factory=list)
    priority: int = 0


class AnalysisInsights(BaseModel):
    intents: List[AnalysisIntent] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class EDAReport(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    table_name: str = ""
    profile: Optional[DataProfile] = None
    analysis_insights: Optional[AnalysisInsights] = None
    steps: List[StepResult] = Field(default_factory=list)
    views: List[ViewResult] = Field(default_factory=list)
    timeline: List[DecisionRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Target analysis models (used by skills/target.py and skills/query_tools.py)
# ---------------------------------------------------------------------------

class TargetSpec(BaseModel):
    column: Optional[str] = None
    task_type: str = "unknown"          # regression | classification | ranking | unknown
    confidence: float = 0.0
    reason: str = ""


class TargetAssociation(BaseModel):
    feature: str
    metric: str                          # cramers_v | eta_squared | corr_spearman | missingness_*
    score: float = 0.0
    direction: Optional[str] = None     # positive | negative
    details: Optional[Dict[str, Any]] = None


class TargetInsights(BaseModel):
    target: TargetSpec
    distribution: Dict[str, Any] = Field(default_factory=dict)
    associations: List["TargetAssociation"] = Field(default_factory=list)
    missingness: List["TargetAssociation"] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Existing models kept for backward compatibility during transition
# ---------------------------------------------------------------------------

class NLQRequest(BaseModel):
    prompt: str
    clientContext: Optional[Dict[str, Any]] = None


class RunRequest(BaseModel):
    table_name: Optional[str] = None
    query: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str
