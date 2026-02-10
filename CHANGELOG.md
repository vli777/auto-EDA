# Changelog

All notable changes to Auto-EDA are documented here.

## [Unreleased]

### Planned
- Jupyter notebook export (see JUPYTER_PLAN.md)

## 2025-05 — Advanced Analysis Skills

### Added
- 28 query-driven analysis tools: linear/logistic regression, LightGBM (regression + classification), quantile regression, SHAP summary/dependence, partial dependence, seasonality test, autocorrelation test, lag feature search, rolling stats, trend breaks, numeric transforms, interaction scan, binning optimizer, date-part features, target encoding, group comparison, matched comparison, diff-in-diff, uplift check, percentile compare, segmentation, cohort change, leakage scan, drift check, missingness mechanism, hypothesis generator, test selector, result validator
- LLM-powered query planner that maps user questions to tool selections
- Percentile-based comparison tool for metric analysis

## 2025-04 — EDA Pipeline & Streaming UI

### Added
- Core EDA pipeline: profile → classify → intent → recommend → build view → narrate
- SSE streaming for real-time progress during analysis
- Analysis intents skill — LLM infers what questions to explore from data structure
- Decision trace logging for pipeline transparency
- 9 chart types: bar, line, scatter, histogram, box, table, heatmap, pie, area
- Recharts-based frontend renderers for all chart types
- Histogram percentile markers (P10/P25/P50/P75/P90 reference lines)
- Column classification skill (temporal, geographic, measure, categorical, etc.)
- Summary statistics table (df.describe-style with skew/kurtosis)
- Follow-up query support — append new analysis to existing report
- Auto-start EDA on file upload

### Changed
- Light theme for frontend
- Chart sizing and layout adjustments

## 2025-03 — Data Handling & Table Preview

### Added
- Table preview panel with dropdown expansion and infinite scroll
- Cursor pagination for table preview API
- LRU caching for table previews
- Numeric-string parsing (currency, percentages, SI suffixes)
- Datetime parsing and format inference helpers
- SI label conversion for chart axes
- PyArrow read engine for better column type detection
- Duplicate file upload prevention via content hashing

### Fixed
- PyArrow data type conversion edge cases

## 2025-02 — LLM Integration & Visualization

### Added
- Multi-provider LLM support (Groq, OpenAI, NVIDIA) via modular loader
- Structured LLM outputs with deterministic fallback
- Diagnostic helpers for data type inference and chart type recommendation
- Vega-Lite chart rendering with data normalization
- Prompt-token overlap scoring for better column selection
- Upload zone with spinner status

### Changed
- Switched to Tailwind CSS
- Moved to Groq as default LLM provider
- Consolidated system prompts into dedicated module

### Fixed
- JSON parsing errors from LLM responses
- Vega-Lite schema parsing and data coercion
- Filter transform validation for Vega compatibility
- NVIDIA empty response fallback via structured-output retry

## 2025-01 — Initial Release

### Added
- FastAPI backend with CSV upload and in-memory session storage
- Next.js frontend with file upload and prompt bar
- LangChain-based LLM client
- Natural language query to visualization pipeline
