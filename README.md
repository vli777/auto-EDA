# Auto-EDA

Agent-driven, autonomous exploratory data analysis. Upload a dataset, the system profiles the data, reasons about intent, and returns a notebook-like report. You can additionally submit queries that the agent will explore further.

![Recording 2026-02-08 223201 (1)](https://github.com/user-attachments/assets/970026c2-8785-47d0-8632-07a8e633432f)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Visualization | Recharts, Vega-Lite |
| Backend | FastAPI, Python 3, Pandas, Pydantic |
| LLM | LangChain (Groq, OpenAI, NVIDIA) |
| Streaming | Server-Sent Events (SSE) |

## Features

- **Autonomous EDA Pipeline**: Upload a CSV and get a full analysis — profiling, intent detection, chart recommendations, and narrated insights — with no manual steps
- **Real-Time Streaming**: SSE-powered progress updates render charts and findings as they're generated
- **Smart Chart Recommendations**: Deterministic recommendation engine picks chart types based on column roles, cardinality, and data distributions
- **Semantic Column Detection**: Automatically classifies columns as temporal, geographic, measure, categorical, or identifier
- **String-Encoded Numeric Parsing**: Recognizes financial shorthand like "$1.3B", "1,234.56" and converts to proper numbers
- **Multi-Provider LLM Support**: Works with Groq, OpenAI, or NVIDIA — falls back to deterministic logic on LLM failure
- **Follow-Up Queries**: Ask natural language questions to drill deeper into your data after the initial analysis
- **Interactive Charts**: Bar, line, scatter, histogram, box, pie, heatmap, and table views with tooltips and responsive layout

## How It Works

All results stream to the frontend in real time via SSE as each step completes.

### Initial EDA Pipeline

1. **Upload** — CSV is parsed, stored in-memory, and bound to the session
2. **Profile** — Statistics computed per column: nulls, cardinality, distribution shape, value ranges
3. **Classify** — LLM refines column roles (temporal, geographic, measure, categorical, identifier) using semantic understanding
4. **Summary** — A dataset overview card is built and streamed immediately
5. **Intent inference** — LLM inspects the full profile and derives a ranked list of analysis intents (e.g. "compare revenue by region over time"), each with relevant fields and priority
6. **Intent views** — For each intent: candidates are generated, scored, deduplicated against prior views, validated, built into chart specs with pre-aggregated data, then narrated by the LLM

### Follow-Up Query Pipeline

When the user submits a natural language query, a separate agent-driven flow runs:

1. **Query planning** — LLM agent reads the query, dataset profile, and prior analysis context, then selects up to 2 analysis tools from a library of 30+ (regression, segmentation, cohort analysis, time-series tests, SHAP, causal inference, feature engineering diagnostics, etc.)
2. **Tool execution** — Each selected tool builds a `ViewPlan` from the profile and query data
3. **Validate → Build → Narrate** — Same pipeline as above: plans are validated (with deterministic fallback on failure), chart specs are built, and the LLM narrates findings
4. **Stream** — Results are appended to the existing report and streamed to the frontend

## Quickstart

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # ensure NEXT_PUBLIC_API_BASE matches backend URL
npm run dev
```

Open http://localhost:3000, upload a CSV, and try prompts like:
- "Create a pie chart representing industry breakdown"
- "Create a scatter plot of founded year and valuation"
- "Create a table to see which investors appear most frequently"
- "Give me the best representation of data if I want to understand the correlation of ARR and Valuation"

## Configuration

Create a `.env` file in the `backend/` directory:

```env
LLM_PROVIDER=groq          # Options: groq, openai, nvidia
LLM_API_KEY=your_api_key
LLM_MODEL=llama-3.1-8b-instant
```

| Provider | Notes |
|----------|-------|
| **Groq** (default) | Fast inference with Llama models |
| **OpenAI** | GPT-4o-mini or other models with strict schema validation |
| **NVIDIA** | Access to NVIDIA-hosted models |

Frontend config lives in `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

## Project Structure

```
backend/
  main.py                  # FastAPI entry point
  app/
    llm.py                 # LLM chat wrapper
    llm_loader.py          # Multi-provider LLM factory (Groq, OpenAI, NVIDIA)
    nlq_llm.py             # Natural language query LLM helpers
    models.py              # LLM request/response models
    prompts.py             # Prompt templates
    session_store.py       # Per-session state management
  core/
    models.py              # Pydantic domain models
    storage.py             # In-memory dataset storage
    utils.py               # Shared utilities
  server/
    api.py                 # API route handlers
    orchestrator.py        # Pipeline orchestration
    sse.py                 # Server-Sent Events streaming
  skills/                  # Agent skill modules (one per pipeline step)
    profile.py             # Data profiling and column role detection
    classify.py            # LLM-based column classification
    intent.py              # User intent detection
    target.py              # Column targeting from intent
    recommend.py           # Chart recommendation engine
    build_view.py          # Chart spec and data aggregation
    narrate.py             # LLM-generated chart narration
    query_planner.py       # NLQ → analysis plan via tool-calling agent
    query_tools.py         # Tools available to the query planning agent
    validate.py            # Chart spec validation
    summary.py             # Report summary generation
  test_llm_features.py     # Test suite

frontend/
  app/                     # Next.js pages and layout
  components/              # UI components (upload, prompt bar, charts)
  components/charts/       # Chart renderers (bar, line, scatter, histogram, etc.)
  lib/                     # API client, SSE hook, utilities
  types/                   # TypeScript type definitions
```

## Testing

```bash
cd backend
python -m pytest test_llm_features.py -v
```

Covers column role detection, numeric parsing, dataset profiling, chart recommendations, model validation, and provider capabilities.
