"""Data models for Router, Planner, Tools, and Pipeline Execution."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class ToolOutput(BaseModel):
    """Normalized payload emitted by any tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    summary: str
    error: Optional[str] = None
    artifact_path: Optional[str] = None


class VerificationIssue(BaseModel):
    """Specific discrepancy or hallucination caught by the verifier."""
    issue_type: str
    description: str
    offending_segment: Optional[str] = None


class VerificationResult(BaseModel):
    """Structured outcome of candidate answer verification."""
    is_verified: bool
    confidence_score: float
    issues: List[VerificationIssue] = Field(default_factory=list)
    original_answer: str
    repaired_answer: str
    repair_applied: bool = False


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
