"""Request and response schemas for FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from src.compat.pydantic_compat import BaseModel, Field


class HealthResponse(BaseModel):
    """Healthcheck status schema."""
    status: str = "ok"
    version: str = "0.1.0"
    active_dataset: Optional[str] = None

    def __init__(self, **kwargs):
        self.status = kwargs.get("status", "ok")
        self.version = kwargs.get("version", "0.1.0")
        self.active_dataset = kwargs.get("active_dataset")


class QueryRequest(BaseModel):
    """Request model for submitting natural-language queries."""
    question: str
    user_id: str = "default_user"
    custom_sql: Optional[str] = None

    def __init__(self, **kwargs):
        self.question = kwargs.get("question", "")
        self.user_id = kwargs.get("user_id", "default_user")
        self.custom_sql = kwargs.get("custom_sql")


class ToolOutputResponse(BaseModel):
    tool_name: str
    success: bool
    summary: str
    artifact_path: Optional[str] = None

    def __init__(self, **kwargs):
        self.tool_name = kwargs.get("tool_name", "")
        self.success = bool(kwargs.get("success", True))
        self.summary = kwargs.get("summary", "")
        self.artifact_path = kwargs.get("artifact_path")


class QueryResponse(BaseModel):
    """Response returned upon query processing."""
    question: str
    answer: str
    intent: str
    confidence: float
    tool_plan: List[str]
    tool_outputs: List[Dict[str, Any]]
    verification: Optional[Dict[str, Any]] = None
    artifacts: List[str] = Field(default_factory=list)
    status: str

    def __init__(self, **kwargs):
        self.question = kwargs.get("question", "")
        self.answer = kwargs.get("answer", "")
        self.intent = kwargs.get("intent", "")
        self.confidence = float(kwargs.get("confidence", 0.0))
        self.tool_plan = list(kwargs.get("tool_plan", []))
        self.tool_outputs = list(kwargs.get("tool_outputs", []))
        self.verification = kwargs.get("verification")
        self.artifacts = list(kwargs.get("artifacts", []))
        self.status = kwargs.get("status", "success")


class UploadResponse(BaseModel):
    """Response returned after dataset upload."""
    message: str
    dataset_name: str
    row_count: int
    column_count: int
    columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]

    def __init__(self, **kwargs):
        self.message = kwargs.get("message", "")
        self.dataset_name = kwargs.get("dataset_name", "")
        self.row_count = int(kwargs.get("row_count", 0))
        self.column_count = int(kwargs.get("column_count", 0))
        self.columns = list(kwargs.get("columns", []))
        self.numeric_columns = list(kwargs.get("numeric_columns", []))
        self.categorical_columns = list(kwargs.get("categorical_columns", []))
        self.datetime_columns = list(kwargs.get("datetime_columns", []))
