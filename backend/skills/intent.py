"""
Intent reasoning skill — uses LLM to propose analysis intents from dataset profile.

Falls back to heuristic intents if LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.models import AnalysisInsights, AnalysisIntent, DataProfile, ColumnRole

logger = logging.getLogger("uvicorn.error")
_llm_warn_logged = False

_INTENT_SYSTEM = """You are an expert data analyst.
Given a dataset profile, propose a prioritized list of analysis intents that would surface genuinely useful insights.

Return JSON:
{
  "intents": [
    {"title": "string", "rationale": "string", "fields": ["col1", "col2"], "priority": 1}
  ],
  "warnings": ["string"]
}

Rules:
- Provide 4-8 intents.
- Favor intents grounded in the actual columns and roles.
- Use concrete language (e.g., "Trend of retail sales over time", "Top item types by transfers").
- If you detect a likely target variable, include it as one intent but not the only one.
- Return ONLY valid JSON. No prose, no code fences.

CRITICAL — avoid trivially obvious intents:
- Do NOT propose correlations between fields that are definitionally related (e.g., net_worth vs assets, total_revenue vs sales, age vs birth_year, price vs cost_plus_margin).
- Do NOT propose "X and Y are correlated" when any domain expert would already know this.
- Instead, focus on: unexpected distributions, outliers, breakdowns by segment, time trends, fields where the data might surprise the analyst.
- Ask yourself: "Would a domain expert already know this relationship is true?" If yes, skip it and find something more interesting.
"""

_SELECT_SYSTEM = """You are an expert data analyst.
Given a list of analysis intents, select the single best next intent to pursue.

Return JSON:
{
  "selected_title": "string",
  "rationale": "string",
  "fields": ["col1", "col2"]
}

Rules:
- Select exactly one intent.
- The rationale should be one concise sentence.
- Return ONLY valid JSON. No prose, no code fences.
"""


def infer_analysis_intents(profile: DataProfile, *, context: dict | None = None) -> AnalysisInsights:
    """LLM-driven intent inference with heuristic fallback."""
    try:
        from app.llm import chat_json

        payload = _profile_payload(profile)
        if context:
            payload["analysis_context"] = context
        result = chat_json(_INTENT_SYSTEM, json.dumps(payload, default=str), max_tokens=900)
        return _coerce_insights(result)
    except Exception as e:
        global _llm_warn_logged
        if not _llm_warn_logged:
            logger.warning("LLM intent reasoning unavailable, using heuristic fallback: %s", e)
            _llm_warn_logged = True
        return _heuristic_intents(profile)


def select_intent(
    intents: AnalysisInsights,
    *,
    query: str = "",
    context: dict | None = None,
) -> dict:
    """Select the next intent to pursue."""
    intent_list = intents.intents if intents else []
    if not intent_list:
        return {"selected_title": "Overview of distributions and missingness", "rationale": "Fallback intent.", "fields": []}

    try:
        from app.llm import chat_json

        payload = {
            "query": query,
            "intents": [
                {"title": i.title, "rationale": i.rationale, "fields": i.fields, "priority": i.priority}
                for i in intent_list
            ],
        }
        if context:
            payload["analysis_context"] = context
        result = chat_json(_SELECT_SYSTEM, json.dumps(payload, default=str), max_tokens=400)
        return {
            "selected_title": str(result.get("selected_title") or intent_list[0].title),
            "rationale": str(result.get("rationale") or ""),
            "fields": list(result.get("fields") or []),
        }
    except Exception as e:
        global _llm_warn_logged
        if not _llm_warn_logged:
            logger.warning("LLM intent selection unavailable, using heuristic fallback: %s", e)
            _llm_warn_logged = True
        top = sorted(intent_list, key=lambda i: i.priority or 0)[0]
        return {"selected_title": top.title, "rationale": top.rationale, "fields": top.fields}


def _profile_payload(profile: DataProfile) -> Dict[str, Any]:
    return {
        "table_name": profile.table_name,
        "row_count": profile.row_count,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "role": c.role.value,
                "cardinality": c.cardinality,
                "missing_pct": c.missing_pct,
                "examples": (c.examples or [])[:3],
            }
            for c in profile.columns[:30]
        ],
    }


def _coerce_insights(obj: Dict[str, Any]) -> AnalysisInsights:
    intents_raw = obj.get("intents") or []
    warnings = obj.get("warnings") or []
    intents: List[AnalysisIntent] = []
    for i, it in enumerate(intents_raw):
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        if not title:
            continue
        intents.append(AnalysisIntent(
            title=title,
            rationale=str(it.get("rationale") or ""),
            fields=list(it.get("fields") or []),
            priority=int(it.get("priority") or (i + 1)),
        ))
    return AnalysisInsights(intents=intents, warnings=list(warnings))


def _heuristic_intents(profile: DataProfile) -> AnalysisInsights:
    intents: List[AnalysisIntent] = []
    cols = profile.columns

    temporals = [c for c in cols if c.role == ColumnRole.temporal]
    measures = [c for c in cols if c.role in (ColumnRole.measure, ColumnRole.count)]
    cats = [c for c in cols if c.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic)]

    if temporals and measures:
        intents.append(AnalysisIntent(
            title=f"Trend of {measures[0].name} over time",
            rationale="Temporal + numeric columns support trend analysis.",
            fields=[temporals[0].name, measures[0].name],
            priority=1,
        ))
    if cats and measures:
        intents.append(AnalysisIntent(
            title=f"Top {cats[0].name} by {measures[0].name}",
            rationale="Categorical + numeric columns support ranking.",
            fields=[cats[0].name, measures[0].name],
            priority=2,
        ))
    if len(measures) >= 2:
        intents.append(AnalysisIntent(
            title=f"Relationship between {measures[0].name} and {measures[1].name}",
            rationale="Multiple measures support correlation analysis.",
            fields=[measures[0].name, measures[1].name],
            priority=3,
        ))
    if cats and temporals and measures:
        intents.append(AnalysisIntent(
            title=f"Seasonality by {cats[0].name}",
            rationale="Temporal + categorical breakdown reveals seasonality/segments.",
            fields=[temporals[0].name, cats[0].name, measures[0].name],
            priority=4,
        ))
    if not intents:
        intents.append(AnalysisIntent(
            title="Overview of distributions and missingness",
            rationale="Fallback when no strong signals are available.",
            fields=[],
            priority=1,
        ))

    return AnalysisInsights(intents=intents, warnings=[])
