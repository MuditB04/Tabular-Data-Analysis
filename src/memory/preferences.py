"""Schemas for persistent user preference and semantic memory."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """Stores high-level user preferences and defaults without storing raw data."""
    preferred_metric: str = "mean"
    default_date_range: Optional[str] = None
    enable_visualization: bool = True
    max_output_rows: int = 10
    custom_definitions: Dict[str, str] = Field(default_factory=dict)
    column_aliases: Dict[str, str] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MemoryAuditRecord(BaseModel):
    """Audit log entry capturing memory write/read operations."""
    session_id: str
    action: str
    key: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
