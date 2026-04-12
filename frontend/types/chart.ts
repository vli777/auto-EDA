/**
 * TypeScript types matching the backend ChartSpec DSL.
 * These mirror the Pydantic models in core/models.py.
 */

export type ChartType =
  | "bar"
  | "line"
  | "scatter"
  | "hist"
  | "box"
  | "table"
  | "heatmap"
  | "pie"
  | "area";

export type EncodingChannel = {
  field: string;
  type?: string; // quantitative, nominal, temporal, ordinal
  aggregate?: string; // sum, mean, count, min, max, median
  bin?: boolean;
  time_unit?: string;
  sort?: string;
};

export type ChartEncoding = {
  x?: EncodingChannel;
  y?: EncodingChannel;
  color?: EncodingChannel;
  facet?: EncodingChannel;
  size?: EncodingChannel;
  theta?: EncodingChannel;
};

export type ChartOptions = {
  sort?: string;
  top_n?: number;
  log?: boolean;
  bin_count?: number;
  tooltip_fields?: string[];
  orientation?: string;
  stacked?: boolean;
  markers?: { value: number; label?: string }[];
};

export type ChartSpec = {
  chart_type: ChartType;
  encoding: ChartEncoding;
  options: ChartOptions;
  data_inline: Record<string, any>[];
  title: string;
  subtitle?: string;
};

export type ViewPlan = {
  chart_type: ChartType;
  encoding: ChartEncoding;
  options: ChartOptions;
  intent: string;
  fields_used: string[];
  tags: string[];
};

export type ViewResult = {
  id: string;
  plan: ViewPlan;
  spec: ChartSpec;
  data_inline: Record<string, any>[];
  explanation: string;
  fallback_repaired?: boolean; // true when plan was invalid and deterministic fallback was applied
  created_at: string;
};

export type StepType =
  | "summary_stats"
  | "analysis_intents"
  | "intent_views"
  | "quality_overview"
  | "relationships"
  | "outliers_segments"
  | "query_driven";

export type StepResult = {
  step_type: StepType;
  headline: string;
  views: string[]; // view IDs
  findings: string[];
  warnings: string[];
  decision_trace?: string | null;
};

export type ColumnInfo = {
  name: string;
  dtype: string;
  role: string;
  cardinality: number;
  missing_pct: number;
  stats?: Record<string, any>;
  top_values?: { value: string; n: number }[];
  examples?: string[];
  warnings?: string[];
};

export type DataProfile = {
  table_name: string;
  row_count: number;
  columns: ColumnInfo[];
  sample_rows?: Record<string, any>[];
  visualization_hints?: Record<string, any>;
};

export type AnalysisIntent = {
  title: string;
  rationale: string;
  fields: string[];
  priority: number;
};

export type AnalysisInsights = {
  intents: AnalysisIntent[];
  warnings: string[];
};

export type EDAReport = {
  run_id: string;
  table_name: string;
  profile?: DataProfile;
  analysis_insights?: AnalysisInsights;
  steps: StepResult[];
  views: ViewResult[];
  timeline: any[];
};
