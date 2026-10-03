"""Abstract base tool interface and shared context representation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd

from src.router.models import DatasetMetadata, RouteDecision, ToolOutput


class ToolContext:
    """Execution context passed into every tool invocation."""

    def __init__(
        self,
        question: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        route_decision: RouteDecision,
        previous_outputs: Optional[List[ToolOutput]] = None,
        db_connection: Optional[Any] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
    ):
        self.question = question
        self.df = df
        self.metadata = metadata
        self.route_decision = route_decision
        self.previous_outputs = previous_outputs or []
        self.db_connection = db_connection
        self.user_preferences = user_preferences or {}

    def get_output_from(self, tool_name: str) -> Optional[ToolOutput]:
        """Fetch output emitted by an earlier tool in the pipeline."""
        for out in self.previous_outputs:
            if out.tool_name == tool_name:
                return out
        return None


class BaseTool(ABC):
    """Abstract base class for all tabular analysis tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique machine identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable explanation of tool capabilities."""
        pass

    @abstractmethod
    def execute(self, context: ToolContext) -> ToolOutput:
        """Run tool computation and return a structured ToolOutput."""
        pass
