"""Tool 3: Python Stat Analyzer.

Performs parametric & non-parametric statistical evaluations:
- Descriptive summaries (mean, std, median, skewness, kurtosis)
- Pearson & Spearman correlations
- Robust outlier detection via Interquartile Range (IQR) and Z-score
- Multi-group variance and categorical comparisons
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.router.models import ToolOutput
from src.tools.base import BaseTool, ToolContext


class PythonStatAnalyzerTool(BaseTool):
    """Calculates statistics, distributions, correlations, and identifies outliers."""

    @property
    def name(self) -> str:
        return "python_stat_analyzer"

    @property
    def description(self) -> str:
        return "Calculates descriptive statistics, correlations, IQR/Z-score outliers, and group metrics."

    def execute(self, context: ToolContext) -> ToolOutput:
        try:
            df = context.df
            num_cols = context.metadata.numeric_columns
            q_lower = context.question.lower()

            if not num_cols:
                return ToolOutput(
                    tool_name=self.name,
                    success=True,
                    data={"message": "No numeric columns available for statistical analysis."},
                    summary="No numeric columns found in the dataset to perform statistical operations.",
                )

            analysis_mode = self._determine_mode(q_lower)

            if analysis_mode == "outliers":
                return self._detect_outliers(df, num_cols, context)
            elif analysis_mode == "correlation":
                return self._compute_correlations(df, num_cols, context)
            elif analysis_mode == "comparison":
                return self._compute_group_comparison(df, num_cols, context)
            else:
                return self._compute_descriptive_stats(df, num_cols, context)

        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                success=False,
                data={},
                summary=f"Statistical analysis encountered an error: {str(e)}",
                error=str(e),
            )

    def _determine_mode(self, q_lower: str) -> str:
        if any(term in q_lower for term in ["outlier", "anomaly", "extreme", "irregular", "z-score", "iqr"]):
            return "outliers"
        if any(term in q_lower for term in ["correlat", "relationship", "association", "covarian"]):
            return "correlation"
        if any(term in q_lower for term in ["compare", "versus", "vs", "difference", "by"]):
            return "comparison"
        return "descriptive"

    def _compute_descriptive_stats(
        self, df: pd.DataFrame, num_cols: List[str], context: ToolContext
    ) -> ToolOutput:
        """Calculate detailed summary stats for numeric features."""
        stats_payload: Dict[str, Dict[str, Any]] = {}
        summary_lines: List[str] = []

        # Target columns if specified, otherwise up to 5 primary numeric columns
        target_cols = [c for c in context.route_decision.target_columns if c in num_cols] or num_cols[:5]

        for col in target_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            q25 = float(series.quantile(0.25))
            median = float(series.median())
            q75 = float(series.quantile(0.75))
            std_val = float(series.std()) if len(series) > 1 else 0.0
            mean_val = float(series.mean())

            skew_val = float(series.skew()) if len(series) > 2 else 0.0

            stats_payload[col] = {
                "count": int(len(series)),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": round(float(series.min()), 2),
                "p25": round(q25, 2),
                "median": round(median, 2),
                "p75": round(q75, 2),
                "max": round(float(series.max()), 2),
                "iqr": round(q75 - q25, 2),
                "skewness": round(skew_val, 2),
            }

            skew_label = "fairly symmetrical" if abs(skew_val) < 0.5 else ("right-skewed" if skew_val > 0 else "left-skewed")
            summary_lines.append(
                f"{col}: Mean = {mean_val:.2f}, Median = {median:.2f}, Std = {std_val:.2f} ({skew_label})"
            )

        return ToolOutput(
            tool_name=self.name,
            success=True,
            data={"descriptive_statistics": stats_payload},
            summary="Statistical Profile:\n- " + "\n- ".join(summary_lines),
        )

    def _compute_correlations(
        self, df: pd.DataFrame, num_cols: List[str], context: ToolContext
    ) -> ToolOutput:
        """Compute Pearson correlation matrix and identify significant relationships."""
        if len(num_cols) < 2:
            return self._compute_descriptive_stats(df, num_cols, context)

        corr_df = df[num_cols].corr(method="pearson").round(3)
        pairs: List[Dict[str, Any]] = []

        seen = set()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                val = float(corr_df.loc[c1, c2])
                if not math.isnan(val):
                    pairs.append({"col_a": c1, "col_b": c2, "r": val, "abs_r": abs(val)})

        pairs.sort(key=lambda x: x["abs_r"], reverse=True)
        top_pairs = pairs[:5]

        findings = []
        for p in top_pairs:
            strength = "strong" if p["abs_r"] > 0.7 else ("moderate" if p["abs_r"] > 0.4 else "weak")
            direction = "positive" if p["r"] > 0 else "negative"
            findings.append(f"{p['col_a']} and {p['col_b']}: r = {p['r']} ({strength} {direction})")

        return ToolOutput(
            tool_name=self.name,
            success=True,
            data={
                "matrix": corr_df.to_dict(),
                "top_correlations": top_pairs,
            },
            summary="Correlation Analysis:\n- " + ("\n- ".join(findings) if findings else "No correlation could be calculated."),
        )

    def _detect_outliers(
        self, df: pd.DataFrame, num_cols: List[str], context: ToolContext
    ) -> ToolOutput:
        """Detect extreme anomalies using both IQR (1.5x) and Z-score (>3.0)."""
        target_cols = [c for c in context.route_decision.target_columns if c in num_cols] or num_cols
        outlier_reports: Dict[str, Any] = {}
        summary_parts: List[str] = []

        for col in target_cols:
            series = df[col].dropna()
            if len(series) < 5:
                continue

            # 1. IQR method
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (1.5 * iqr)
            upper_bound = q75 + (1.5 * iqr)

            iqr_mask = (series < lower_bound) | (series > upper_bound)
            iqr_outliers = series[iqr_mask].tolist()

            # 2. Z-score method
            mean = series.mean()
            std = series.std()
            z_outliers = []
            if std > 0:
                z_scores = ((series - mean) / std).abs()
                z_outliers = series[z_scores > 3.0].tolist()

            count = len(iqr_outliers)
            pct = round((count / len(series)) * 100, 2)
            outlier_reports[col] = {
                "iqr_count": count,
                "iqr_percentage": pct,
                "bounds": [round(float(lower_bound), 2), round(float(upper_bound), 2)],
                "iqr_sample": [round(float(x), 2) for x in iqr_outliers[:5]],
                "z_score_extreme_count": len(z_outliers),
            }

            if count > 0:
                summary_parts.append(
                    f"{col}: {count} IQR outliers ({pct}% of data; bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

        if not summary_parts:
            summary = "Outlier Inspection: No significant statistical anomalies detected."
        else:
            summary = "Outlier Detection (IQR Method):\n- " + "\n- ".join(summary_parts)

        return ToolOutput(
            tool_name=self.name,
            success=True,
            data={"outliers": outlier_reports},
            summary=summary,
        )

    def _compute_group_comparison(
        self, df: pd.DataFrame, num_cols: List[str], context: ToolContext
    ) -> ToolOutput:
        """Compare statistics across categorical groups."""
        cat_cols = context.metadata.categorical_columns
        if not cat_cols or not num_cols:
            return self._compute_descriptive_stats(df, num_cols, context)

        group_col = cat_cols[0]
        val_col = num_cols[0]

        # Use target columns if matched
        for col in context.route_decision.target_columns:
            if col in cat_cols:
                group_col = col
            elif col in num_cols:
                val_col = col

        grouped = df.groupby(group_col)[val_col].agg(["count", "mean", "median", "std"]).round(2)
        records = grouped.reset_index().to_dict(orient="records")

        lines = [
            f"{r[group_col]}: Mean={r['mean']}, Median={r['median']}, N={r['count']}"
            for r in records[:6]
        ]

        return ToolOutput(
            tool_name=self.name,
            success=True,
            data={"group_by": group_col, "metric": val_col, "records": records},
            summary=f"Group Comparison ({val_col} across {group_col}):\n- " + "\n- ".join(lines),
        )
