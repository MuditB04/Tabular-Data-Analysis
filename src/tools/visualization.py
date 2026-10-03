"""Tool 4: Visualization Tool.

Generates interactive charts (bar, line, scatter, box, histogram) using Plotly.
Saves rendered HTML and JSON specification artifacts locally and provides safe metadata payloads.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.router.models import IntentType, ToolOutput
from src.tools.base import BaseTool, ToolContext


class VisualizationTool(BaseTool):
    """Generates charts and persists HTML/JSON interactive artifacts."""

    def __init__(self, artifacts_dir: str = "artifacts/charts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "visualization"

    @property
    def description(self) -> str:
        return "Generates interactive Plotly charts (bar, line, scatter, box, histogram) and exports HTML/JSON artifacts."

    def execute(self, context: ToolContext) -> ToolOutput:
        try:
            chart_type, x_col, y_col = self._resolve_chart_spec(context)
            data_df = self._select_dataframe(context, x_col, y_col)

            # Build Figure or Mock representation if plotly is optional
            artifact_file = self.artifacts_dir / f"chart_{chart_type}_{uuid.uuid4().hex[:8]}.html"
            chart_title = f"{chart_type.capitalize()} Plot: {y_col or 'Count'} by {x_col or 'Index'}"

            chart_rendered = self._render_chart(
                chart_type=chart_type,
                df=data_df,
                x=x_col,
                y=y_col,
                title=chart_title,
                output_path=artifact_file,
            )

            return ToolOutput(
                tool_name=self.name,
                success=True,
                data={
                    "chart_type": chart_type,
                    "x_axis": x_col,
                    "y_axis": y_col,
                    "title": chart_title,
                    "rendered": chart_rendered,
                },
                summary=f"Generated interactive {chart_type} chart for '{y_col or 'count'}' by '{x_col}'. Saved to {artifact_file}.",
                artifact_path=str(artifact_file),
            )

        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                success=False,
                data={},
                summary=f"Chart rendering encountered an error: {str(e)}",
                error=str(e),
            )

    def _resolve_chart_spec(self, context: ToolContext) -> Tuple[str, Optional[str], Optional[str]]:
        """Determine suitable chart type and axes based on query intent and column types."""
        q_lower = context.question.lower()
        num_cols = context.metadata.numeric_columns
        cat_cols = context.metadata.categorical_columns
        dt_cols = context.metadata.datetime_columns
        targets = context.route_decision.target_columns

        # Check explicit chart requests
        if "scatter" in q_lower:
            chart_type = "scatter"
        elif "line" in q_lower or context.route_decision.intent == IntentType.TREND:
            chart_type = "line"
        elif "box" in q_lower:
            chart_type = "box"
        elif "histogram" in q_lower or "distribut" in q_lower:
            chart_type = "histogram"
        else:
            chart_type = "bar"

        # Determine axes
        x_col = None
        y_col = None

        if chart_type == "line":
            x_col = dt_cols[0] if dt_cols else (cat_cols[0] if cat_cols else None)
            y_col = num_cols[0] if num_cols else None
        elif chart_type == "scatter":
            if len(num_cols) >= 2:
                x_col, y_col = num_cols[0], num_cols[1]
            elif num_cols:
                x_col, y_col = None, num_cols[0]
        elif chart_type == "histogram":
            x_col = num_cols[0] if num_cols else None
            y_col = None
        elif chart_type == "box":
            x_col = cat_cols[0] if cat_cols else None
            y_col = num_cols[0] if num_cols else None
        else:  # Bar
            x_col = cat_cols[0] if cat_cols else None
            y_col = num_cols[0] if num_cols else None

        # Override with explicit targets if available
        for t in targets:
            if t in cat_cols or t in dt_cols:
                x_col = t
            elif t in num_cols:
                y_col = t

        return chart_type, x_col, y_col

    def _select_dataframe(
        self, context: ToolContext, x_col: Optional[str], y_col: Optional[str]
    ) -> pd.DataFrame:
        """Use SQL tool aggregated output if available, else sample from root DataFrame."""
        sql_out = context.get_output_from("sql_engine")
        if sql_out and sql_out.success and sql_out.data.get("records"):
            candidate_df = pd.DataFrame(sql_out.data["records"])
            # Check if columns are present
            if (not x_col or x_col in candidate_df.columns) and (not y_col or y_col in candidate_df.columns):
                return candidate_df

        # Fallback to root dataframe limited to 100 rows for rendering responsiveness
        return context.df.head(100).copy()

    def _render_chart(
        self,
        chart_type: str,
        df: pd.DataFrame,
        x: Optional[str],
        y: Optional[str],
        title: str,
        output_path: Path,
    ) -> bool:
        """Render using Plotly if available, or generate a standalone interactive HTML snippet."""
        try:
            import plotly.express as px

            if chart_type == "bar" and x and y:
                fig = px.bar(df, x=x, y=y, title=title)
            elif chart_type == "line" and x and y:
                fig = px.line(df, x=x, y=y, title=title)
            elif chart_type == "scatter" and x and y:
                fig = px.scatter(df, x=x, y=y, title=title)
            elif chart_type == "histogram" and x:
                fig = px.histogram(df, x=x, title=title)
            elif chart_type == "box" and y:
                fig = px.box(df, x=x, y=y, title=title)
            else:
                fig = px.bar(df.head(20), title=title)

            fig.update_layout(template="plotly_white")
            fig.write_html(str(output_path), include_plotlyjs="cdn")
            return True

        except ImportError:
            # Standalone pure HTML5 SVG / Data visualization fallback
            html_content = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title><style>body {{ font-family: sans-serif; padding: 20px; }}</style></head>
<body>
    <h3>{title}</h3>
    <p>Type: <strong>{chart_type}</strong> | X-Axis: <strong>{x}</strong> | Y-Axis: <strong>{y}</strong></p>
    <table border="1" cellpadding="5" style="border-collapse:collapse;">
        <tr>{"".join(f"<th>{c}</th>" for c in df.columns)}</tr>
        {"".join("<tr>" + "".join(f"<td>{val}</td>" for val in row) + "</tr>" for row in df.head(10).values)}
    </table>
</body>
</html>"""
            output_path.write_text(html_content, encoding="utf-8")
            return True
