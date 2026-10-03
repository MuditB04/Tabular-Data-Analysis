"""Data models for Router, Planner, Tools, and Pipeline Execution."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass
    from src.compat.pydantic_compat import BaseModel, Field


class IntentType(str, Enum):
    """Enumeration of supported user query intent categories."""
    LOOKUP = "lookup"
    AGGREGATION = "aggregation"
    TREND = "trend"
    COMPARISON = "comparison"
    QUALITY_CHECK = "quality_check"
    AMBIGUOUS = "ambiguous"
    OUT_OF_DOMAIN = "out_of_domain"


class RouteDecision(BaseModel):
    """Structured decision output from the Router & Planner."""
    intent: IntentType
    confidence: float
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    is_out_of_domain: bool = False
    rejection_message: Optional[str] = None
    tool_plan: List[str] = Field(default_factory=list)
    reasoning: str = ""
    target_columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        self.intent = kwargs.get("intent")
        self.confidence = float(kwargs.get("confidence", 0.0))
        self.needs_clarification = bool(kwargs.get("needs_clarification", False))
        self.clarification_question = kwargs.get("clarification_question")
        self.is_out_of_domain = bool(kwargs.get("is_out_of_domain", False))
        self.rejection_message = kwargs.get("rejection_message")
        self.tool_plan = list(kwargs.get("tool_plan", []))
        self.reasoning = kwargs.get("reasoning", "")
        self.target_columns = list(kwargs.get("target_columns", []))
        self.parameters = dict(kwargs.get("parameters", {}))


class ToolOutput(BaseModel):
    """Normalized payload emitted by any tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    summary: str
    error: Optional[str] = None
    artifact_path: Optional[str] = None

    def __init__(self, **kwargs):
        self.tool_name = kwargs.get("tool_name", "")
        self.success = bool(kwargs.get("success", True))
        self.data = dict(kwargs.get("data", {}))
        self.summary = kwargs.get("summary", "")
        self.error = kwargs.get("error")
        self.artifact_path = kwargs.get("artifact_path")


class VerificationIssue(BaseModel):
    """Specific discrepancy or hallucination caught by the verifier."""
    issue_type: str
    description: str
    offending_segment: Optional[str] = None

    def __init__(self, **kwargs):
        self.issue_type = kwargs.get("issue_type", "")
        self.description = kwargs.get("description", "")
        self.offending_segment = kwargs.get("offending_segment")


class VerificationResult(BaseModel):
    """Structured outcome of candidate answer verification."""
    is_verified: bool
    confidence_score: float
    issues: List[VerificationIssue] = Field(default_factory=list)
    original_answer: str
    repaired_answer: str
    repair_applied: bool = False

    def __init__(self, **kwargs):
        self.is_verified = bool(kwargs.get("is_verified", True))
        self.confidence_score = float(kwargs.get("confidence_score", 1.0))
        self.issues = list(kwargs.get("issues", []))
        self.original_answer = kwargs.get("original_answer", "")
        self.repaired_answer = kwargs.get("repaired_answer", "")
        self.repair_applied = bool(kwargs.get("repair_applied", False))


class DatasetMetadata(BaseModel):
    """Descriptive metadata for an active tabular dataset."""
    name: str
    source_path: str
    file_format: str
    row_count: int
    column_count: int
    columns: List[str]
    dtypes: Dict[str, str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    table_name: str = "dataset"

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "")
        self.source_path = kwargs.get("source_path", "")
        self.file_format = kwargs.get("file_format", "")
        self.row_count = int(kwargs.get("row_count", 0))
        self.column_count = int(kwargs.get("column_count", 0))
        self.columns = list(kwargs.get("columns", []))
        self.dtypes = dict(kwargs.get("dtypes", {}))
        self.numeric_columns = list(kwargs.get("numeric_columns", []))
        self.categorical_columns = list(kwargs.get("categorical_columns", []))
        self.datetime_columns = list(kwargs.get("datetime_columns", []))
        self.table_name = kwargs.get("table_name", "dataset")
