"""Schemas for persistent user preference and semantic memory."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    from src.compat.pydantic_compat import BaseModel, Field


class UserPreferences(BaseModel):
    """Stores high-level user preferences and defaults without storing raw data."""
    preferred_metric: str = "mean"
    default_date_range: Optional[str] = None
    enable_visualization: bool = True
    max_output_rows: int = 10
    custom_definitions: Dict[str, str] = Field(default_factory=dict)
    column_aliases: Dict[str, str] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def __init__(self, **kwargs):
        self.preferred_metric = kwargs.get("preferred_metric", "mean")
        self.default_date_range = kwargs.get("default_date_range")
        self.enable_visualization = bool(kwargs.get("enable_visualization", True))
        self.max_output_rows = int(kwargs.get("max_output_rows", 10))
        self.custom_definitions = dict(kwargs.get("custom_definitions", {}))
        self.column_aliases = dict(kwargs.get("column_aliases", {}))
        self.last_updated = kwargs.get("last_updated", datetime.utcnow().isoformat())


class MemoryAuditRecord(BaseModel):
    """Audit log entry capturing memory write/read operations."""
    session_id: str
    action: str
    key: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id", "")
        self.action = kwargs.get("action", "")
        self.key = kwargs.get("key", "")
        self.timestamp = kwargs.get("timestamp", datetime.utcnow().isoformat())
