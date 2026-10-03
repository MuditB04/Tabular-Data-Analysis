# Tabular Data Analysis Agent

An autonomous, router-driven AI agent that ingests tabular datasets (CSV, XLSX, Parquet) and answers natural language questions with grounded outputs, verified computations, and interactive visualizations.

```
                            ┌─────────────────────────┐
                            │  Natural Language Query │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │     Router & Planner    │
                            │  - Intent Classification│
                            │  - Ambiguity Guardrail  │
                            │  - Out-of-Domain Reject │
                            └────────────┬────────────┘
                                         │  (Dynamic Tool Plan)
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
             ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │ SchemaInspector │         │ SQLEngine (Duck)│         │StatAnalyzer(IQR)│
    └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   Visualization Tool    │
                            │   (Plotly HTML/JSON)    │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │     Answer Verifier     │
                            │ - Hallucination Check   │
                            │ - Numerical Grounding   │
                            │ - Repair Engine         │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │ Verified Grounded Output│
                            └─────────────────────────┘
```

---

## 1. Project Summary
**Tabular Data Analysis Agent** is a production-minded Python framework engineered to query tabular datasets reliably without hallucinations. It employs a deterministic planner/router that classifies user intent, detects fuzzy column ambiguities (e.g. typos), intercepts out-of-domain conversational queries, and orchestrates an array of five specialized analysis tools. Every response is audited by an automated Answer Verifier to ensure that all numerical and column claims match underlying execution metrics.

---

## 2. Architecture Overview

### Intent Routing & Execution Flow
1. **Intake & Memory Context**: User queries are ingested alongside active dataset metadata (columns, data types, cardinality). The system checks persistent memory for user preferences (preferred metrics, column aliases).
2. **Guards & Routing**:
   - **Out-of-Domain Guardrail**: Identifies non-analytical or off-topic queries and rejects them before tool execution.
   - **Ambiguity Interception**: Detects fuzzy matches or typos against actual column names and issues a targeted clarification question.
   - **Intent Classifier**: Maps queries into `lookup`, `aggregation`, `trend`, `comparison`, or `quality_check` and selects a minimal tool execution plan.
3. **Tool Execution**: Tools execute sequentially, passing structured outputs and registered DataFrames.
4. **Answer Verification**: Audits candidate answers against tool datasets, flags unsupported figures or invalid columns, and provides a repaired answer.

---

## 3. The Five Core Tools

| Tool | Module | Description |
| :--- | :--- | :--- |
| **Schema Inspector** | `src/tools/schema_inspector.py` | Calculates row/column counts, data types, null rates, unique cardinality, and samples. |
| **SQL Query Engine** | `src/tools/sql_engine.py` | Synthesizes and executes safe, read-only SQL queries via DuckDB (or SQLite fallback) for aggregations, group-bys, and filters. |
| **Python Stat Analyzer** | `src/tools/python_stat_analyzer.py` | Computes descriptive distributions (mean, median, skewness), Pearson correlation matrices, and IQR / Z-score outliers. |
| **Visualization Tool** | `src/tools/visualization.py` | Generates Plotly charts (bar, line, scatter, box, histogram) and exports self-contained HTML/JSON artifacts. |
| **Answer Verifier** | `src/tools/answer_verifier.py` | Audits numbers, column references, and absolute superlatives against tool results to prevent hallucinations. |

---

## 4. Setup & Installation

### Requirements
- Python 3.11+
- Virtual environment recommended

### Quickstart Setup
```bash
# Clone the repository
git clone https://github.com/your-username/tabular-data-analysis-agent.git
cd tabular-data-analysis-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

---

## 5. Running the Project

### CLI Interactive Mode
To run queries directly in the terminal:
```bash
# Interactive REPL mode
python3 -m src.agent.service --data data/sample_sales.csv

# Single-query execution
python3 -m src.agent.service --data data/sample_sales.csv --query "What is the total revenue by region?"
```

### FastAPI REST API
To launch the backend API service:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at `http://localhost:8000/docs`.

---

## 6. API Usage Examples

### 1. Upload Dataset
```bash
curl -X POST "http://localhost:8000/dataset/upload" \
  -H "accept: application/json" \
  -F "file=@data/sample_sales.csv"
```
**Response:**
```json
{
  "message": "Successfully loaded 'sample_sales'.",
  "dataset_name": "sample_sales",
  "row_count": 100,
  "column_count": 10,
  "columns": ["order_id", "date", "product", "category", "region", "quantity", "unit_price", "revenue", "customer_type", "salesperson"]
}
```

### 2. Analytical Aggregation Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total revenue by region?"}'
```
**Response:**
```json
{
  "question": "What is the total revenue by region?",
  "intent": "aggregation",
  "confidence": 0.84,
  "tool_plan": ["sql_engine", "answer_verifier"],
  "answer": "Query returned 4 rows:\n- region=West, sum_revenue=114778.41\n- region=North, sum_revenue=114010.54\n- region=South, sum_revenue=110032.15\n- region=East, sum_revenue=92963.5",
  "status": "success"
}
```

### 3. Ambiguous Query (Typo Interception)
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total quantiti?"}'
```
**Response:**
```json
{
  "question": "What is the total quantiti?",
  "intent": "ambiguous",
  "answer": "Did you mean column 'quantity' when you referenced 'quantiti'? Possible matching columns in this dataset: quantity.",
  "status": "needs_clarification"
}
```

### 4. Out-of-Domain Guardrail Rejection
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recipe for apple pie?"}'
```
**Response:**
```json
{
  "question": "What is the recipe for apple pie?",
  "intent": "out_of_domain",
  "answer": "The question 'What is the recipe for apple pie?' appears unrelated to the loaded dataset 'sample_sales'. This agent only answers questions regarding the active data table (10 columns: order_id, date, product, category, region...).",
  "status": "rejected_out_of_domain"
}
```

---

## 7. Memory Design
The agent uses a non-sensitive persistent preference store (`src/memory/store.py`):
- **Stored Content**: High-level user preferences (e.g. `preferred_metric: "median"`), dataset column aliases (`sales -> revenue`), and custom metric definitions.
- **Zero Raw Data Policy**: Dataset rows and sensitive values are never persisted to the memory store.
- **Eviction**: Automatic TTL eviction purges entries inactive for more than 30 days.

---

## 8. Running Automated Tests

Run the complete test suite across Router, Tools, Ambiguity, Out-of-Domain, and Pipeline:
```bash
# Using pytest
pytest -v tests/

# Or via the standalone runner
python3 tests/run_tests.py
```

---

## 9. Implementation Notes: Custom vs. Libraries

| Component | Author Implementation | External Libraries Used |
| :--- | :--- | :--- |
| **Router & Planner** | Custom intent scoring, fuzzy ambiguity detector, out-of-domain pattern guards, minimal tool planner | `difflib`, `re` |
| **Answer Verifier** | Custom regex numeric parser, cross-tool number matching, column verification, hallucination repair | Python standard library |
| **SQL Engine** | Custom query synthesis engine & SQL injection guard | `duckdb`, `sqlite3`, `pandas` |
| **Statistical Analyzer** | Custom IQR / Z-score outlier detection and skewness categorization | `numpy`, `pandas` |
| **Memory Store** | Custom SQLite preference store with TTL eviction and alias mappings | `sqlite3` |
| **Visualization** | Custom chart specification resolver & pipeline integration | `plotly` |
| **API Layer** | Custom endpoint contracts and error handling | `fastapi`, `uvicorn`, `pydantic` |

---

## 10. Limitations & Future Roadmap
- **Multi-Table Joins**: The current router assumes single-table operations; multi-table joins can be enabled via foreign key discovery.
- **Natural Language to Complex Window Functions**: Complex cohort retention or cumulative metrics benefit from integrating local LLMs (e.g. LiteLLM / Ollama).
- **Automated Data Cleaning**: Future releases can add automated missing-value imputation and schema type casting.
