# Tabular Data Analysis Agent

An autonomous, router-driven AI agent that ingests tabular datasets (CSV, XLSX, Parquet) and answers natural language questions with grounded outputs, verified computations, and interactive visualizations.


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



## 4. Running the Project

### CLI Interactive Mode
To run queries directly in the terminal:
```bash
# Interactive REPL mode
python3 -m src.agent.service --data data/sample_sales.csv

---\
