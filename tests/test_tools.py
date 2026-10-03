"""Unit tests for the five distinct analysis tools."""

import pytest
import pandas as pd
from pathlib import Path

from src.router.models import DatasetMetadata, IntentType, RouteDecision, ToolOutput
from src.tools.base import ToolContext
from src.tools.schema_inspector import SchemaInspectorTool
from src.tools.sql_engine import SQLEngineTool
from src.tools.python_stat_analyzer import PythonStatAnalyzerTool
from src.tools.visualization import VisualizationTool
from src.tools.answer_verifier import AnswerVerifierTool


@pytest.fixture
def sample_df():
    data = {
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
        "region": ["East", "West", "East", "West", "East", "West", "East", "West", "East", "West"],
        "price": [10.0, 20.0, 15.0, 25.0, 12.0, 18.0, 22.0, 28.0, 14.0, 500.0],  # 500 is extreme outlier
        "quantity": [1, 2, 1, 3, 2, 1, 4, 2, 1, None],  # Contains 1 missing value
    }
    return pd.DataFrame(data)


@pytest.fixture
def tool_context(sample_df):
    meta = DatasetMetadata(
        name="test_data",
        source_path="mock.csv",
        file_format="csv",
        row_count=10,
        column_count=5,
        columns=["order_id", "category", "region", "price", "quantity"],
        dtypes={"order_id": "int64", "category": "object", "region": "object", "price": "float64", "quantity": "float64"},
        numeric_columns=["order_id", "price", "quantity"],
        categorical_columns=["category", "region"],
        datetime_columns=[],
        table_name="test_table",
    )
    decision = RouteDecision(
        intent=IntentType.AGGREGATION,
        confidence=0.9,
        tool_plan=["sql_engine", "answer_verifier"],
        target_columns=["price"],
    )
    return ToolContext(
        question="What is the average price?",
        df=sample_df,
        metadata=meta,
        route_decision=decision,
    )


def test_schema_inspector(tool_context):
    tool = SchemaInspectorTool()
    out = tool.execute(tool_context)
    assert out.success is True
    assert out.data["total_rows"] == 10
    assert out.data["total_columns"] == 5
    assert "quantity" in out.data["columns"]
    assert out.data["columns"]["quantity"]["null_count"] == 1


def test_sql_engine_aggregation(tool_context):
    tool = SQLEngineTool()
    out = tool.execute(tool_context)
    assert out.success is True
    assert "records" in out.data
    assert len(out.data["records"]) > 0


def test_python_stat_analyzer_outliers(tool_context):
    tool = PythonStatAnalyzerTool()
    tool_context.question = "Detect any outliers in price"
    out = tool.execute(tool_context)
    assert out.success is True
    assert "outliers" in out.data
    # 500 should be caught as an IQR outlier
    assert out.data["outliers"]["price"]["iqr_count"] >= 1


def test_visualization_tool(tool_context, tmp_path):
    tool = VisualizationTool(artifacts_dir=str(tmp_path))
    out = tool.execute(tool_context)
    assert out.success is True
    assert out.artifact_path is not None
    assert Path(out.artifact_path).exists()


def test_answer_verifier_detects_hallucinations(tool_context):
    # Candidate answer contains an unsupported number 999999.0
    tool = AnswerVerifierTool()
    tool_context.previous_outputs = [
        ToolOutput(
            tool_name="sql_engine",
            success=True,
            data={"result": 66.4},
            summary="Average price is 66.4.",
        )
    ]
    tool_context.route_decision.parameters["candidate_answer"] = (
        "The average price is 66.4 and revenue is guaranteed to hit 999999.0 next quarter."
    )
    out = tool.execute(tool_context)
    assert out.success is True
    assert out.data["is_verified"] is False
    assert out.data["repair_applied"] is True
    assert any("999999" in iss["description"] for iss in out.data["issues"])
