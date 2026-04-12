"""
EDA orchestrator — runs the deterministic analysis pipeline.

Phase A: synchronous run_eda_sync
Phase B: async run_eda_async with SSE channel
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd

from core.models import (
    DataProfile,
    EDAReport,
    StepResult,
    StepType,
    ViewPlan,
    ViewResult,
)
from core.storage import append_view, create_run, save_run
from server.sse import (
    SSEChannel,
    EVT_ERROR,
    EVT_PROGRESS,
    EVT_RUN_COMPLETE,
    EVT_RUN_STARTED,
    EVT_STEP_STARTED,
    EVT_STEP_SUMMARY,
    EVT_ANALYSIS_INTENTS,
    EVT_VIEW_PLANNED,
    EVT_VIEW_READY,
    EVT_WARNING,
)
from skills.build_view import build_view
from skills.classify import classify_columns
from skills.narrate import summarize_step
from skills.profile import build_profile
from skills.recommend import generate_candidates, score_and_select
from skills.intent import infer_analysis_intents
from skills.summary import summarize_dataset, build_summary_stats_view
from skills.validate import deterministic_fallback, validate_plan, validate_view

logger = logging.getLogger("uvicorn.error")

# Default analysis pipeline
DEFAULT_POLICY: List[StepType] = [
    StepType.summary_stats,
    StepType.analysis_intents,
    StepType.intent_views,
]

# Budget per step (max charts generated per step)
STEP_BUDGET = 4


# ---------------------------------------------------------------------------
# Synchronous pipeline (Phase A)
# ---------------------------------------------------------------------------

def _build_llm_context(report: EDAReport) -> dict:
    """Compact context for LLM decisions to avoid repeating analysis."""
    steps = []
    for s in report.steps:
        steps.append({
            "step_type": s.step_type.value,
            "headline": s.headline,
            "findings": (s.findings or [])[:2],
        })

    views = []
    for v in report.views[-20:]:
        views.append({
            "intent": v.plan.intent,
            "fields": v.plan.fields_used,
            "title": v.spec.title,
        })

    intents = []
    if report.analysis_insights:
        for i in report.analysis_insights.intents[:8]:
            intents.append({"title": i.title, "fields": i.fields, "priority": i.priority})

    return {
        "steps_done": steps,
        "views_done": views,
        "intents": intents,
        "table": report.table_name,
    }

def _validate_plan_with_fallback(
    plan: ViewPlan, profile: DataProfile
) -> tuple[ViewPlan | None, bool, List[str]]:
    """
    Validate a plan; attempt deterministic repair if invalid.

    Returns (resolved_plan_or_None, was_repaired, all_warnings).
    None means both the original and the fallback were invalid — caller should skip.
    """
    warnings: List[str] = []
    is_valid, w = validate_plan(plan, profile)
    warnings.extend(w)
    if is_valid:
        return plan, False, warnings

    logger.warning("  Plan invalid: %s — attempting fallback", w)
    repaired = deterministic_fallback(plan, profile)
    is_valid, w = validate_plan(repaired, profile)
    warnings.extend(w)
    if not is_valid:
        logger.warning("  Fallback also invalid, skipping")
        return None, True, warnings
    return repaired, True, warnings


def run_eda_sync(
    df: pd.DataFrame,
    table_name: str,
    session_id: str,
    *,
    query: Optional[str] = None,
    budget: int = 12,
) -> EDAReport:
    """
    Run the full deterministic EDA pipeline synchronously.

    1. Profile the dataset
    2. For each step in DEFAULT_POLICY:
       a. Generate candidate views
       b. Score & select top views (respecting budget)
       c. Validate each plan
       d. Build each view (pre-aggregate data)
       e. Validate the result
       f. Append to the report
    3. Return the complete EDAReport
    """
    if query:
        # Queries should run fresh to avoid replaying prior charts.
        logger.info("Running query-specific analysis (no reuse): %s", query)
    # Create run
    report = create_run(session_id, table_name)
    logger.info("EDA run %s started for table '%s' (%d rows)", report.run_id, table_name, len(df))

    # Step 1: Profile
    profile = build_profile(df, table_name)
    profile = classify_columns(profile)
    report.profile = profile
    logger.info("Profile complete: %d columns, %d rows", len(profile.columns), profile.row_count)

    # Track all views for deduplication
    views_done: List[ViewPlan] = []
    all_view_results: List[ViewResult] = []
    total_views = 0

    # Build the step pipeline
    pipeline = [StepType.query_driven] if query else list(DEFAULT_POLICY)

    intent_texts: List[str] = []

    step_idx = 0
    intent_queue: List[dict] = []

    # Step 2: Run each analysis step
    for step_type in pipeline:
        if total_views >= budget and step_type not in (
            StepType.summary_stats, StepType.analysis_intents
        ):
            break

        if step_type == StepType.summary_stats:
            summary = summarize_dataset(profile)
            summary_view = build_summary_stats_view(df, profile)
            append_view(report.run_id, summary_view)
            step_result = StepResult(
                step_type=step_type,
                headline=summary["headline"],
                views=[summary_view.id],
                findings=summary["findings"],
            )
            report.steps.append(step_result)
            step_idx += 1
            continue

        if step_type == StepType.analysis_intents:
            if report.analysis_insights is None:
                report.analysis_insights = infer_analysis_intents(
                    profile, context=_build_llm_context(report)
                )
            intents = report.analysis_insights.intents if report.analysis_insights else []
            intent_queue = [
                {"title": i.title, "fields": i.fields or []}
                for i in intents
            ]
            step_result = StepResult(
                step_type=step_type,
                headline="Analysis intents derived from dataset structure",
                findings=[i.title for i in intents],
            )
            report.steps.append(step_result)
            step_idx += 1
            continue

        if step_type == StepType.intent_views:
            if not intent_queue:
                continue
            for idx, intent in enumerate(intent_queue, start=1):
                intent_payload = [{"title": intent["title"], "fields": intent["fields"]}]
                remaining_intents = len(intent_queue) - idx + 1
                step_budget = min(STEP_BUDGET, max(1, (budget - total_views) // remaining_intents))
                if step_budget <= 0:
                    break

                logger.info("Step: intent_views %d/%d (budget=%d)", idx, len(intent_queue), step_budget)

                candidates = generate_candidates(
                    profile,
                    step_type,
                    query=query or "",
                    intents=intent_payload,
                    context=_build_llm_context(report),
                )
                logger.info("  Generated %d candidates", len(candidates))

                selected = score_and_select(candidates, views_done, budget=step_budget)
                logger.info("  Selected %d views", len(selected))

                step_views: List[str] = []
                step_findings: List[str] = []
                step_warnings: List[str] = []
                step_view_results: List[ViewResult] = []

                for plan in selected:
                    resolved_plan, was_repaired, plan_warnings = _validate_plan_with_fallback(plan, profile)
                    step_warnings.extend(plan_warnings)
                    if resolved_plan is None:
                        continue

                    try:
                        view = build_view(resolved_plan, df)
                    except Exception as e:
                        logger.warning("  Build failed for '%s': %s", resolved_plan.intent, e)
                        step_warnings.append(f"Build failed: {e}")
                        continue

                    view_valid, view_warnings = validate_view(view)
                    step_warnings.extend(view_warnings)
                    if not view_valid:
                        logger.warning("  View invalid: %s", view_warnings)
                        continue

                    view.fallback_repaired = was_repaired
                    append_view(report.run_id, view)
                    step_views.append(view.id)
                    step_view_results.append(view)
                    views_done.append(resolved_plan)
                    all_view_results.append(view)
                    total_views += 1

                    step_findings.append(view.explanation)
                    logger.info("  View built: %s (%s, %d rows)",
                                 view.spec.title, view.spec.chart_type.value, len(view.data_inline))

                step_result = StepResult(
                    step_type=step_type,
                    headline=f"Intent {idx}/{len(intent_queue)}: {intent['title']}",
                    views=step_views,
                    findings=step_findings,
                    warnings=step_warnings,
                )

                if step_view_results:
                    narration = summarize_step(
                        profile, step_result, step_view_results, context=_build_llm_context(report)
                    )
                    step_result.headline = narration.get("headline", step_result.headline)
                    step_result.findings = narration.get("findings", step_result.findings)

                report.steps.append(step_result)
                step_idx += 1
            continue

        if step_type == StepType.query_driven:
            step_budget = min(STEP_BUDGET, budget - total_views)
            candidates = generate_candidates(
                profile,
                step_type,
                query=query or "",
                intents=None,
                context=_build_llm_context(report),
            )
            selected = score_and_select(candidates, views_done, budget=step_budget)

            step_views: List[str] = []
            step_findings: List[str] = []
            step_warnings: List[str] = []
            step_view_results: List[ViewResult] = []

            for plan in selected:
                resolved_plan, was_repaired, plan_warnings = _validate_plan_with_fallback(plan, profile)
                step_warnings.extend(plan_warnings)
                if resolved_plan is None:
                    continue

                try:
                    view = build_view(resolved_plan, df)
                except Exception as e:
                    logger.warning("  Build failed for '%s': %s", resolved_plan.intent, e)
                    step_warnings.append(f"Build failed: {e}")
                    continue

                view_valid, view_warnings = validate_view(view)
                step_warnings.extend(view_warnings)
                if not view_valid:
                    logger.warning("  View invalid: %s", view_warnings)
                    continue

                view.fallback_repaired = was_repaired
                append_view(report.run_id, view)
                step_views.append(view.id)
                step_view_results.append(view)
                views_done.append(resolved_plan)
                all_view_results.append(view)
                total_views += 1

                step_findings.append(view.explanation)

            step_result = StepResult(
                step_type=step_type,
                headline=f"Query: {query}",
                views=step_views,
                findings=step_findings,
                warnings=step_warnings,
            )

            if step_view_results:
                narration = summarize_step(
                    profile, step_result, step_view_results, context=_build_llm_context(report)
                )
                step_result.headline = narration.get("headline", step_result.headline)
                step_result.findings = narration.get("findings", step_result.findings)

            report.steps.append(step_result)
            step_idx += 1
            continue

        # Skip other step types in this intent-driven flow
        step_idx += 1
        continue

    logger.info("EDA run %s complete: %d views across %d steps",
                report.run_id, total_views, len(report.steps))

    # Persist
    save_run(report)
    return report


# ---------------------------------------------------------------------------
# Thread pool for CPU-bound view building
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Async pipeline (Phase B) — same logic, emits SSE events
# ---------------------------------------------------------------------------

async def run_eda_async(
    df: pd.DataFrame,
    table_name: str,
    session_id: str,
    channel: SSEChannel,
    *,
    query: Optional[str] = None,
    budget: int = 12,
) -> EDAReport:
    """
    Async EDA pipeline that streams progress via SSE.

    Same logic as run_eda_sync but emits events at each milestone.
    CPU-bound view building runs in a thread pool executor.
    """
    loop = asyncio.get_event_loop()

    if query:
        # Queries should run fresh to avoid replaying prior charts.
        logger.info("Running query-specific analysis (no reuse): %s", query)

    report = create_run(session_id, table_name)
    await channel.emit(EVT_RUN_STARTED, {
        "run_id": report.run_id,
        "table_name": table_name,
        "row_count": len(df),
    })

    # Profile (CPU-bound)
    profile = await loop.run_in_executor(_executor, build_profile, df, table_name)
    profile = await loop.run_in_executor(_executor, classify_columns, profile)
    report.profile = profile
    await channel.emit(EVT_PROGRESS, {
        "stage": "profile_complete",
        "columns": len(profile.columns),
        "row_count": profile.row_count,
    })

    views_done: List[ViewPlan] = []
    all_view_results: List[ViewResult] = []
    total_views = 0

    # Build the step pipeline
    pipeline = [StepType.query_driven] if query else list(DEFAULT_POLICY)

    step_idx = 0
    intent_queue: List[dict] = []

    for step_type in pipeline:
        if total_views >= budget and step_type not in (
            StepType.summary_stats, StepType.analysis_intents
        ):
            break

        if step_type == StepType.summary_stats:
            await channel.emit(EVT_STEP_STARTED, {
                "step_type": step_type.value,
                "budget": 0,
                "step_index": step_idx,
            })
            summary = summarize_dataset(profile)
            summary_view = await loop.run_in_executor(_executor, build_summary_stats_view, df, profile)
            append_view(report.run_id, summary_view)
            step_result = StepResult(
                step_type=step_type,
                headline=summary["headline"],
                views=[summary_view.id],
                findings=summary["findings"],
                warnings=[],
            )
            report.steps.append(step_result)
            await channel.emit(EVT_VIEW_READY, summary_view.model_dump())
            await channel.emit(EVT_STEP_SUMMARY, {
                "step_type": step_type.value,
                "headline": step_result.headline,
                "view_count": 0,
                "findings": step_result.findings,
                "step_index": step_idx,
            })
            step_idx += 1
            continue

        if step_type == StepType.analysis_intents:
            await channel.emit(EVT_STEP_STARTED, {
                "step_type": step_type.value,
                "budget": 0,
                "step_index": step_idx,
            })
            if report.analysis_insights is None:
                report.analysis_insights = await loop.run_in_executor(
                    _executor, lambda: infer_analysis_intents(profile, context=_build_llm_context(report))
                )
            intents = report.analysis_insights.intents if report.analysis_insights else []
            intent_queue = [
                {"title": i.title, "fields": i.fields or []}
                for i in intents
            ]
            step_result = StepResult(
                step_type=step_type,
                headline="Analysis intents derived from dataset structure",
                views=[],
                findings=[i.title for i in intents],
                warnings=[],
            )
            report.steps.append(step_result)
            await channel.emit(EVT_ANALYSIS_INTENTS, report.analysis_insights.model_dump())
            await channel.emit(EVT_STEP_SUMMARY, {
                "step_type": step_type.value,
                "headline": step_result.headline,
                "view_count": 0,
                "findings": step_result.findings,
                "step_index": step_idx,
            })
            step_idx += 1
            continue

        if step_type == StepType.intent_views:
            if not intent_queue:
                continue
            for idx, intent in enumerate(intent_queue, start=1):
                intent_payload = [{"title": intent["title"], "fields": intent["fields"]}]
                remaining_intents = len(intent_queue) - idx + 1
                step_budget = min(STEP_BUDGET, max(1, (budget - total_views) // remaining_intents))
                if step_budget <= 0:
                    break

                await channel.emit(EVT_STEP_STARTED, {
                    "step_type": step_type.value,
                    "budget": step_budget,
                    "step_index": step_idx,
                })

                candidates = generate_candidates(
                    profile,
                    step_type,
                    query=query or "",
                    intents=intent_payload,
                    context=_build_llm_context(report),
                )
                selected = score_and_select(candidates, views_done, budget=step_budget)

                step_views: List[str] = []
                step_findings: List[str] = []
                step_warnings: List[str] = []
                step_view_results: List[ViewResult] = []

                for plan in selected:
                    resolved_plan, was_repaired, plan_warnings = _validate_plan_with_fallback(plan, profile)
                    step_warnings.extend(plan_warnings)
                    if resolved_plan is None:
                        for w in plan_warnings:
                            await channel.emit(EVT_WARNING, {"message": w})
                        continue

                    await channel.emit(EVT_VIEW_PLANNED, {
                        "intent": resolved_plan.intent,
                        "chart_type": resolved_plan.chart_type.value,
                    })

                    try:
                        view = await loop.run_in_executor(_executor, build_view, resolved_plan, df)
                    except Exception as e:
                        logger.warning("Build failed: %s", e)
                        await channel.emit(EVT_WARNING, {"message": f"Build failed: {e}"})
                        continue

                    view_valid, view_warnings = validate_view(view)
                    step_warnings.extend(view_warnings)
                    if not view_valid:
                        for w in view_warnings:
                            await channel.emit(EVT_WARNING, {"message": w})
                        continue

                    view.fallback_repaired = was_repaired
                    append_view(report.run_id, view)
                    step_views.append(view.id)
                    step_view_results.append(view)
                    views_done.append(resolved_plan)
                    all_view_results.append(view)
                    total_views += 1

                    step_findings.append(view.explanation)
                    await channel.emit(EVT_VIEW_READY, view.model_dump())

                step_result = StepResult(
                    step_type=step_type,
                    headline=f"Intent {idx}/{len(intent_queue)}: {intent['title']}",
                    views=step_views,
                    findings=step_findings,
                    warnings=step_warnings,
                )

                if step_view_results:
                    narration = await loop.run_in_executor(
                        _executor, lambda: summarize_step(
                            profile, step_result, step_view_results, context=_build_llm_context(report)
                        ),
                    )
                    step_result.headline = narration.get("headline", step_result.headline)
                    step_result.findings = narration.get("findings", step_result.findings)

                report.steps.append(step_result)
                await channel.emit(EVT_STEP_SUMMARY, {
                    "step_type": step_type.value,
                    "headline": step_result.headline,
                    "view_count": len(step_views),
                    "findings": step_result.findings,
                    "decision_trace": step_result.decision_trace,
                    "step_index": step_idx,
                })
                step_idx += 1
            continue

        if step_type == StepType.query_driven:
            step_budget = min(STEP_BUDGET, budget - total_views)
            await channel.emit(EVT_STEP_STARTED, {
                "step_type": step_type.value,
                "budget": step_budget,
                "step_index": step_idx,
            })

            candidates = generate_candidates(
                profile,
                step_type,
                query=query or "",
                intents=None,
                context=_build_llm_context(report),
            )
            selected = score_and_select(candidates, views_done, budget=step_budget)

            step_views: List[str] = []
            step_findings: List[str] = []
            step_warnings: List[str] = []
            step_view_results: List[ViewResult] = []

            for plan in selected:
                resolved_plan, was_repaired, plan_warnings = _validate_plan_with_fallback(plan, profile)
                step_warnings.extend(plan_warnings)
                if resolved_plan is None:
                    for w in plan_warnings:
                        await channel.emit(EVT_WARNING, {"message": w})
                    continue

                await channel.emit(EVT_VIEW_PLANNED, {
                    "intent": resolved_plan.intent,
                    "chart_type": resolved_plan.chart_type.value,
                })

                try:
                    view = await loop.run_in_executor(_executor, build_view, resolved_plan, df)
                except Exception as e:
                    logger.warning("Build failed: %s", e)
                    await channel.emit(EVT_WARNING, {"message": f"Build failed: {e}"})
                    continue

                view_valid, view_warnings = validate_view(view)
                step_warnings.extend(view_warnings)
                if not view_valid:
                    for w in view_warnings:
                        await channel.emit(EVT_WARNING, {"message": w})
                    continue

                view.fallback_repaired = was_repaired
                append_view(report.run_id, view)
                step_views.append(view.id)
                step_view_results.append(view)
                views_done.append(resolved_plan)
                all_view_results.append(view)
                total_views += 1

                step_findings.append(view.explanation)
                await channel.emit(EVT_VIEW_READY, view.model_dump())

            step_result = StepResult(
                step_type=step_type,
                headline=f"Query: {query}",
                views=step_views,
                findings=step_findings,
                warnings=step_warnings,
            )

            if step_view_results:
                narration = await loop.run_in_executor(
                    _executor, lambda: summarize_step(
                        profile, step_result, step_view_results, context=_build_llm_context(report)
                    ),
                )
                step_result.headline = narration.get("headline", step_result.headline)
                step_result.findings = narration.get("findings", step_result.findings)

            report.steps.append(step_result)
            await channel.emit(EVT_STEP_SUMMARY, {
                "step_type": step_type.value,
                "headline": step_result.headline,
                "view_count": len(step_views),
                "findings": step_result.findings,
                "decision_trace": step_result.decision_trace,
                "step_index": step_idx,
            })
            step_idx += 1
            continue

    save_run(report)

    await channel.emit(EVT_RUN_COMPLETE, {
        "run_id": report.run_id,
        "total_views": total_views,
        "total_steps": len(report.steps),
    })
    await channel.close()

    return report
