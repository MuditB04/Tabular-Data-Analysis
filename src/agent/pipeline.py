"""Agent Pipeline: Orchestrates routing, tool execution, verification, and output formation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from src.memory.preferences import UserPreferences
from src.memory.store import MemoryStore
from src.router.models import (
    DatasetMetadata,
    IntentType,
    RouteDecision,
    ToolOutput,
    VerificationResult,
)
from src.router.router import Router
from src.tools import BaseTool, get_default_tools
from src.tools.base import ToolContext

logger = logging.getLogger("agent.pipeline")


class AgentPipeline:
    """Orchestrates the entire tabular data analysis workflow."""

    def __init__(
        self,
        router: Optional[Router] = None,
        tools: Optional[Dict[str, BaseTool]] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self.router = router or Router()
        self.tools = tools or get_default_tools()
        self.memory = memory_store or MemoryStore()

    def process_query(
        self,
        question: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata,
        user_id: str = "default_user",
        custom_sql: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end question processing pipeline."""
        # 1. Fetch user preferences and dataset aliases
        user_prefs = self.memory.get_preferences(user_id)
        aliases = self.memory.get_column_aliases(metadata.name)

        # Apply aliases to question if any exist
        normalized_q = question
        for alias, real_col in aliases.items():
            normalized_q = normalized_q.replace(alias, real_col)

        # 2. Plan and Route
        route_decision = self.router.route(normalized_q, metadata)
        if custom_sql:
            route_decision.parameters["custom_sql"] = custom_sql

        # 3. Handle Guards: Out-of-Domain or Ambiguity
        if route_decision.is_out_of_domain:
            return {
                "question": question,
                "answer": route_decision.rejection_message,
                "route_decision": route_decision.model_dump(),
                "tool_outputs": [],
                "verification": None,
                "artifacts": [],
                "status": "rejected_out_of_domain",
            }

        if route_decision.needs_clarification:
            return {
                "question": question,
                "answer": route_decision.clarification_question,
                "route_decision": route_decision.model_dump(),
                "tool_outputs": [],
                "verification": None,
                "artifacts": [],
                "status": "needs_clarification",
            }

        # 4. Sequentially Execute Planned Tools
        tool_outputs: List[ToolOutput] = []
        artifacts: List[str] = []

        context = ToolContext(
            question=normalized_q,
            df=df,
            metadata=metadata,
            route_decision=route_decision,
            previous_outputs=tool_outputs,
            user_preferences=user_prefs.model_dump(),
        )

        for tool_name in route_decision.tool_plan:
            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning("Tool %s not found in registry; skipping.", tool_name)
                continue

            # Update context with prior outputs
            context.previous_outputs = list(tool_outputs)
            output = tool.execute(context)
            tool_outputs.append(output)

            if output.artifact_path:
                artifacts.append(output.artifact_path)

        # 5. Extract Verification Output and Build Final Answer
        verification_data = None
        final_answer = ""

        # Check if answer verifier ran
        verifier_output = None
        for out in tool_outputs:
            if out.tool_name == "answer_verifier" and out.success:
                verifier_output = out
                verification_data = out.data
                break

        if verifier_output and "repaired_answer" in verifier_output.data:
            final_answer = verifier_output.data["repaired_answer"]
        else:
            # Fallback synthesis
            summaries = [out.summary for out in tool_outputs if out.success and out.summary]
            final_answer = "\n\n".join(summaries) or "Analysis complete."

        return {
            "question": question,
            "answer": final_answer,
            "route_decision": route_decision.model_dump(),
            "tool_outputs": [out.model_dump() for out in tool_outputs],
            "verification": verification_data,
            "artifacts": artifacts,
            "status": "success",
        }
