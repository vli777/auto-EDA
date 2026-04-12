"""
build_view package — takes a ViewPlan + DataFrame and returns a ViewResult.

Public API: build_view()

Internal modules:
  _utils.py      — shared utilities (encoding reconciliation, column helpers, tag/time helpers)
  _charts.py     — standard chart builders (bar, line, scatter, hist, box, table, heatmap, pie, area)
  _analytics.py  — analytics / query-tool builders (regression, SHAP, cohort, segmentation, etc.)
  _dispatch.py   — build_view() dispatch + auto-explanation
"""

from ._dispatch import build_view

__all__ = ["build_view"]
