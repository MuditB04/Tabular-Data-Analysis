"""Tool 1: Schema Inspector.

Extracts dataset dimensions, column types, null counts and rates, unique cardinality,
and representative sample records.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from src.router.models import ToolOutput
from src.tools.base import BaseTool, ToolContext


class SchemaInspectorTool(BaseTool):
    """Inspects dataset structure, column types, missingness, and cardinality."""

    @property
    def name(self) -> str:
        return "schema_inspector"

    @property
    def description(self) -> str:
        return "Returns columns, datatypes, null percentage, unique cardinality, and sample rows."

    def execute(self, context: ToolContext) -> ToolOutput:
        try:
            df = context.df
            total_rows = len(df)
            total_cols = len(df.columns)

            columns_profile: Dict[str, Dict[str, Any]] = {}
            for col in df.columns:
                series = df[col]
                null_count = int(series.isna().sum())
                null_pct = round((null_count / total_rows * 100), 2) if total_rows > 0 else 0.0
                unique_count = int(series.nunique(dropna=True))
                dtype_str = str(series.dtype)

                col_meta: Dict[str, Any] = {
                    "dtype": dtype_str,
                    "null_count": null_count,
                    "null_percentage": null_pct,
                    "unique_count": unique_count,
                }

                if pd.api.types.is_numeric_dtype(series):
                    valid_num = series.dropna()
                    if not valid_num.empty:
                        col_meta["min"] = float(valid_num.min())
                        col_meta["max"] = float(valid_num.max())
                        col_meta["mean"] = round(float(valid_num.mean()), 3)
                elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                    sample_values = series.dropna().unique()[:4].tolist()
                    col_meta["sample_categories"] = [str(x) for x in sample_values]

                columns_profile[col] = col_meta

            # Identify columns with missing values
            missing_cols = [
                f"{col} ({m['null_percentage']}% null)"
                for col, m in columns_profile.items()
                if m["null_count"] > 0
            ]

            summary_text = (
                f"Dataset contains {total_rows} rows and {total_cols} columns. "
            )
            if missing_cols:
                summary_text += f"Missing data detected in: {', '.join(missing_cols)}."
            else:
                summary_text += "No missing values detected across any column."

            # Clean sample rows for JSON serialization
            sample_records = df.head(3).to_dict(orient="records")
            cleaned_samples = []
            for record in sample_records:
                clean_rec = {}
                for k, v in record.items():
                    clean_rec[k] = None if pd.isna(v) else v
                cleaned_samples.append(clean_rec)

            return ToolOutput(
                tool_name=self.name,
                success=True,
                data={
                    "total_rows": total_rows,
                    "total_columns": total_cols,
                    "columns": columns_profile,
                    "sample_records": cleaned_samples,
                },
                summary=summary_text,
            )

        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                success=False,
                data={},
                summary=f"Schema inspection failed: {str(e)}",
                error=str(e),
            )
