"""Tool 2: SQL Query Engine.

Executes deterministic SQL queries over the registered in-memory DataFrame using DuckDB
or in-memory SQLite emulation fallback. Supports filtering, aggregations, group-by,
ordering, and window functions without risk of data mutation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.router.models import IntentType, ToolOutput
from src.tools.base import BaseTool, ToolContext

# Safe SQL generation mappings
AGG_KEYWORDS = {
    "total": "SUM", "sum": "SUM",
    "average": "AVG", "avg": "AVG", "mean": "AVG",
    "count": "COUNT", "how many": "COUNT",
    "maximum": "MAX", "max": "MAX", "highest": "MAX", "most": "MAX",
    "minimum": "MIN", "min": "MIN", "lowest": "MIN", "least": "MIN"
}


class SQLEngineTool(BaseTool):
    """Executes safe read-only SQL queries on tabular data."""

    @property
    def name(self) -> str:
        return "sql_engine"

    @property
    def description(self) -> str:
        return "Runs SQL aggregations, filtering, group bys, ordering, and window queries using DuckDB."

    def execute(self, context: ToolContext) -> ToolOutput:
        try:
            sql_query = context.route_decision.parameters.get("custom_sql")
            if not sql_query:
                sql_query = self._synthesize_query(context)

            # Security sanity check: block mutation commands
            forbidden_tokens = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "COPY"]
            for token in forbidden_tokens:
                if re.search(r"\b" + token + r"\b", sql_query, re.IGNORECASE):
                    return ToolOutput(
                        tool_name=self.name,
                        success=False,
                        summary="Mutating SQL queries are strictly prohibited.",
                        error=f"Forbidden statement: {token}",
                    )

            # Execution
            result_df = self._execute_sql(sql_query, context)
            
            # Format and sanitize output records
            row_count = len(result_df)
            columns = list(result_df.columns)
            
            # Convert to serializable dictionary
            raw_records = result_df.head(50).to_dict(orient="records")
            cleaned_records: List[Dict[str, Any]] = []
            for row in raw_records:
                clean_row = {}
                for k, v in row.items():
                    if pd.isna(v):
                        clean_row[k] = None
                    elif hasattr(v, "item"):
                        clean_row[k] = v.item()
                    else:
                        clean_row[k] = v
                cleaned_records.append(clean_row)

            # Generate natural language summary
            summary = self._build_summary(sql_query, cleaned_records, row_count, columns)

            return ToolOutput(
                tool_name=self.name,
                success=True,
                data={
                    "query": sql_query,
                    "row_count": row_count,
                    "columns": columns,
                    "records": cleaned_records,
                },
                summary=summary,
            )

        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                success=False,
                data={"query": sql_query if "sql_query" in locals() else "N/A"},
                summary=f"SQL query execution failed: {str(e)}",
                error=str(e),
            )

    def _execute_sql(self, query: str, context: ToolContext) -> pd.DataFrame:
        """Execute query either via DuckDB or pandasql / sqlite fallback."""
        table_name = context.metadata.table_name
        df = context.df

        try:
            import duckdb
            conn = context.db_connection or duckdb.connect(database=":memory:")
            conn.register(table_name, df)
            return conn.execute(query).fetchdf()
        except ImportError:
            # High-performance SQLite in-memory fallback
            import sqlite3
            conn = sqlite3.connect(":memory:")
            df.to_sql(table_name, conn, index=False)
            return pd.read_sql_query(query, conn)

    def _synthesize_query(self, context: ToolContext) -> str:
        """Deterministically synthesize a SQL query based on intent, targets, and question cues."""
        table = context.metadata.table_name
        df = context.df
        q_lower = context.question.lower()
        targets = context.route_decision.target_columns
        num_cols = context.metadata.numeric_columns
        cat_cols = context.metadata.categorical_columns

        # Detect target numeric metric
        metric_col = None
        for col in targets:
            if col in num_cols:
                metric_col = col
                break
        if not metric_col and num_cols:
            metric_col = num_cols[0]

        # Detect group-by column
        group_col = None
        group_matches = re.findall(r"\b(?:by|per|across|for each)\s+([a-zA-Z_]+)", q_lower)
        if group_matches:
            requested_group = group_matches[0].lower().replace(" ", "_")
            for col in cat_cols + context.metadata.columns:
                if col.lower() == requested_group or col.lower().startswith(requested_group):
                    group_col = col
                    break

        if not group_col and targets:
            for col in targets:
                if col in cat_cols:
                    group_col = col
                    break

        # Detect aggregation function
        agg_func = "SUM"
        for kw, func in AGG_KEYWORDS.items():
            if re.search(r"\b" + kw + r"\b", q_lower):
                agg_func = func
                break

        # Check for count specifics
        if "count" in q_lower or "how many" in q_lower:
            agg_func = "COUNT"

        # Case A: Aggregation with Group By
        if group_col and metric_col and group_col != metric_col:
            if agg_func == "COUNT":
                return f"SELECT {group_col}, COUNT(*) AS total_count FROM {table} GROUP BY {group_col} ORDER BY total_count DESC LIMIT 20"
            return (
                f"SELECT {group_col}, ROUND({agg_func}({metric_col}), 2) AS {agg_func.lower()}_{metric_col} "
                f"FROM {table} GROUP BY {group_col} ORDER BY {agg_func.lower()}_{metric_col} DESC LIMIT 20"
            )

        # Case B: Direct Scalar Aggregation
        if context.route_decision.intent == IntentType.AGGREGATION and metric_col:
            if agg_func == "COUNT":
                return f"SELECT COUNT(*) AS total_count FROM {table}"
            return f"SELECT ROUND({agg_func}({metric_col}), 2) AS {agg_func.lower()}_{metric_col} FROM {table}"

        # Case C: Trend (Time-based grouping)
        if context.metadata.datetime_columns:
            date_col = context.metadata.datetime_columns[0]
            val_col = metric_col or "id"
            return (
                f"SELECT {date_col}, ROUND(SUM({val_col}), 2) AS total_value "
                f"FROM {table} GROUP BY {date_col} ORDER BY {date_col} ASC LIMIT 30"
            )

        # Case D: Lookup / Filtered exploration
        where_clauses = []
        for col in cat_cols:
            unique_vals = df[col].dropna().unique()[:20]
            for u in unique_vals:
                if str(u).lower() in q_lower:
                    where_clauses.append(f"{col} = '{u}'")
                    break

        where_stmt = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return f"SELECT * FROM {table}{where_stmt} LIMIT 10"

    def _build_summary(
        self, query: str, records: List[Dict[str, Any]], row_count: int, columns: List[str]
    ) -> str:
        """Generate concise natural language summary of query results."""
        if row_count == 0:
            return f"Query returned 0 records. Executed: `{query}`"

        if row_count == 1:
            record = records[0]
            items_str = ", ".join(f"{k}: {v}" for k, v in record.items())
            return f"Result: {items_str}. [SQL: `{query}`]"

        if row_count <= 5:
            lines = [", ".join(f"{k}={v}" for k, v in r.items()) for r in records]
            return f"Query returned {row_count} rows:\n- " + "\n- ".join(lines)

        top_val = records[0]
        top_str = ", ".join(f"{k}={v}" for k, v in top_val.items())
        return f"Query returned {row_count} records. Top record: {top_str}."
