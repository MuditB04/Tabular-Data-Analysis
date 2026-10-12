"""Request and response schemas for FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Healthcheck status schema."""
    status: str = "ok"
    version: str = "0.1.0"
    active_dataset: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for submitting natural-language queries."""
    question: str
    user_id: str = "default_user"
    custom_sql: Optional[str] = None


class ToolOutputResponse(BaseModel):
    tool_name: str
    success: bool
    summary: str
    artifact_path: Optional[str] = None


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
