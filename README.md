# Vendor Quote Extractor

Vendor Quote Extractor is a Streamlit and FastAPI prototype for turning vendor quote PDFs into structured JSON. It combines deterministic PDF parsing, LLM-based field extraction, and post-extraction validation so operators can review finance-critical data before it reaches downstream systems.

The public release ships with a fully synthetic fixture suite. No challenge documents, private PDFs, or real credentials are included in this repository.

## What it does

- Parses digital PDFs with `PyMuPDF` and `pdfplumber`
- Detects scanned PDFs and prepares page images for vision-capable providers
- Extracts structured quote data with provider routing and failover
- Validates arithmetic, totals, credits, dates, and confidence signals
- Presents a review-friendly Streamlit UI with PDF preview and JSON export

## Product Goals

The public release is designed around three operating goals:

- High accuracy on finance-critical fields, with an aspirational target of `95%+` correctness on key outputs such as grand total, line items, and vendor metadata
- Human-in-the-loop safety, so low-confidence outputs, ambiguous totals, credits, and validation mismatches are routed to review instead of silently accepted
- Faster than `20 seconds` end-to-end ingestion for supported documents, with typical digital PDFs expected to complete materially faster than the upper bound

## Architecture

The extraction flow has three stages:

1. Parse
   - Extract text and tables from the PDF
   - Detect scanned/image-heavy documents
   - Build source-region metadata for UI highlighting
2. Extract
   - Send parsed content to the configured LLM provider
   - Use structured outputs to map raw content into the quote schema
   - Fall back across providers when API failures occur
3. Validate
   - Reconcile line items against totals
   - Flag TCV vs net confusion, discounts, credits, tax anomalies, and missing data
   - Mark low-confidence results for human review

## Edge Cases Covered

- Discounted vs list pricing in the same document
- Subscription terms implied by coverage date ranges
- Service orders with credits and both TCV and net totals
- Mixed hardware, software, services, tax, and shipping
- Multi-page tables with repeated headers and large line-item counts
- Scanned PDFs that need page rendering for vision extraction

## Repository Layout

- `src/api/` FastAPI routes and request/response models
- `src/extraction/` parser, router, LLM extractor, validator, pipeline
- `src/schemas/` Pydantic models for extracted quote data
- `src/ui/` Streamlit app
- `config/` environment-backed settings and logging setup
- `tests/` unit and integration tests
- `tests/fixtures/` synthetic PDFs and expected output summaries
- `scripts/generate_fixture_suite.py` regenerate the public fixture corpus

## Local Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Start from `.env.example`:

```bash
cp .env.example .env
```

Set at least one provider key:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `FIREWORKS_API_KEY`
- `GEMINI_API_KEY`

The shared provider configuration variables are:

- `LLM_PROVIDER`
- `LLM_MODEL`
- `ANTHROPIC_MODEL`
- `GROQ_MODEL`
- `FIREWORKS_MODEL`
- `GEMINI_MODEL`

### 3. Run the Streamlit UI

The Streamlit app is the main public demo surface and runs the extraction pipeline directly without requiring the API server.

```bash
streamlit run src/ui/app.py
```

### 4. Run the API server

Use this when you want the persistence and correction endpoints.

```bash
uvicorn src.api.routes:app --reload --port 8000
```

## Validation Commands

Run these before publishing changes:

```bash
pytest
pyright src/
ruff check src/ config/ tests/
ruff format src/ config/ tests/
```

## Synthetic Fixture Suite

The checked-in PDFs under `tests/fixtures/` are synthetic and safe to publish.

Fixture set:

- `discounted_hardware_quote.pdf`
- `enterprise_term_quote.pdf`
- `service_order_with_credits.pdf`
- `mixed_category_quote.pdf`
- `multi_page_quote.pdf`

Regenerate the suite:

```bash
python scripts/generate_fixture_suite.py
```

The generator also refreshes JSON summaries in `tests/fixtures/expected/`.

## Benchmarking

The repository includes a benchmark harness used during development to compare providers, model choices, and extraction behavior against the synthetic fixture suite.

What it measures:

- End-to-end processing time per fixture
- Parse, extract, and validate timing breakdowns
- Grand total agreement against expected synthetic outputs
- Line-item extraction counts per document
- Success and failure rates across the suite

Why it matters:

- It keeps the speed target explicit: stay under the `20 second` ingestion ceiling for supported documents
- It keeps the quality target explicit: push toward `95%+` accuracy on important finance fields
- It supports the human-in-the-loop operating model by making it easy to see when validation, confidence, or provider behavior degrades
- It helps tune routing, prompt budgets, chunking behavior, and fallback order

Run the benchmark suite with:

```bash
python benchmark_providers.py
```

This script requires whichever provider keys you want to benchmark.

Benchmark notes:

- The public repository uses synthetic fixtures only
- Results will vary by provider availability, model version, account limits, and network conditions
- The README intentionally avoids hard-coded public leaderboard claims; rerun the benchmark in your own environment to establish current numbers

## Deployment

### GitHub

Create a fresh public repository and push this clean export:

```bash
git init
git add .
git commit -m "Initial public release"
gh repo create alannguyen1/vendor-quote-extractor --public --source=. --remote=origin --push
```

### Streamlit Cloud

Create a new app that points at the GitHub repository above and set:

- Repository: `alannguyen1/vendor-quote-extractor`
- Branch: `main`
- Main file path: `src/ui/app.py`

Add secrets in Streamlit Cloud for whichever provider keys you use. After saving secrets, reboot the app to force a clean install and restart.

## Data and Privacy Notes

- This repository intentionally excludes any private or challenge-specific source material.
- `.env`, local databases, logs, caches, and private sample folders are gitignored.
- The public fixture PDFs and JSON summaries are synthetic and intended for reproducible testing only.
