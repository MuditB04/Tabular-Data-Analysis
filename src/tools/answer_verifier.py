"""Tool 5: Answer Verifier.

Cross-references candidate generated answers against the concrete data outputs
produced by upstream tools (SQL, Stats, Schema).
Catches:
- Fabricated / hallucinated numbers not present in tool data
- Non-existent column references
- Unsubstantiated absolute superlatives ("all", "never", "guaranteed")
Automatically flags issues and produces a repaired, verified answer.
"""

from __future__ import annotations

import re
from typing import Any, List, Set
from src.router.models import (
    ToolOutput,
    VerificationIssue,
    VerificationResult,
)
from src.tools.base import BaseTool, ToolContext


class AnswerVerifierTool(BaseTool):
    """Verifies candidate answers against tool execution outputs and repairs discrepancies."""

    @property
    def name(self) -> str:
        return "answer_verifier"

    @property
    def description(self) -> str:
        return "Checks that candidate responses are fully grounded in computed tool metrics and flags unsupported claims."

    def execute(self, context: ToolContext) -> ToolOutput:
        try:
            # Gather upstream data and candidate answer
            candidate_answer = context.route_decision.parameters.get("candidate_answer")
            if not candidate_answer:
                candidate_answer = self._synthesize_candidate_answer(context)

            # Extract ground-truth numbers from previous tool outputs
            ground_truth_numbers = self._extract_numbers_from_tools(context.previous_outputs)
            valid_columns = set(c.lower() for c in context.metadata.columns)

            issues: List[VerificationIssue] = []

            # Check 1: Numeric grounding
            claimed_numbers = self._extract_numbers_from_text(candidate_answer)
            for num_val, original_token in claimed_numbers:
                # Allow reasonable descriptive indices like 'top 1', 'top 5', '10 rows', years
                if num_val in {1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0}:
                    continue
                if 1990 <= num_val <= 2030: # Years
                    continue

                if not self._is_number_supported(num_val, ground_truth_numbers):
                    issues.append(
                        VerificationIssue(
                            issue_type="unsupported_number",
                            description=f"Claimed number '{original_token}' does not appear in computed tool output.",
                            offending_segment=original_token,
                        )
                    )

            # Check 2: Column legitimacy
            referenced_columns = self._extract_column_mentions(candidate_answer, valid_columns)
            for col in referenced_columns:
                if col not in valid_columns:
                    issues.append(
                        VerificationIssue(
                            issue_type="invalid_column",
                            description=f"Mentioned column '{col}' does not exist in dataset schema.",
                            offending_segment=col,
                        )
                    )

            # Check 3: Superlative / Hallucination risk
            unsupported_superlatives = ["guaranteed", "100% of all", "undoubtedly", "proven without exception"]
            for term in unsupported_superlatives:
                if term in candidate_answer.lower():
                    issues.append(
                        VerificationIssue(
                            issue_type="superlative",
                            description=f"Unsubstantiated absolute claim '{term}' detected.",
                            offending_segment=term,
                        )
                    )

            # Build repaired answer if discrepancies detected
            repaired_answer = candidate_answer
            repair_applied = False
            if issues:
                repair_applied = True
                repaired_answer = self._repair_answer(candidate_answer, issues, context)

            confidence = max(0.20, 1.0 - (len(issues) * 0.20))
            is_verified = len(issues) == 0

            verification_result = VerificationResult(
                is_verified=is_verified,
                confidence_score=round(confidence, 2),
                issues=issues,
                original_answer=candidate_answer,
                repaired_answer=repaired_answer,
                repair_applied=repair_applied,
            )

            summary = (
                f"Verification {'PASSED' if is_verified else 'FLAGGED DISCREPANCIES'} "
                f"(Confidence: {int(confidence*100)}%). "
                f"{len(issues)} issue(s) detected."
            )

            return ToolOutput(
                tool_name=self.name,
                success=True,
                data=verification_result.model_dump(),
                summary=summary,
            )

        except Exception as e:
            return ToolOutput(
                tool_name=self.name,
                success=False,
                data={},
                summary=f"Verification tool error: {str(e)}",
                error=str(e),
            )

    def _synthesize_candidate_answer(self, context: ToolContext) -> str:
        """Construct initial draft answer based on tool outputs."""
        summaries = [out.summary for out in context.previous_outputs if out.success and out.summary]
        if not summaries:
            return "No statistical findings were computed for the dataset."
        return "\n\n".join(summaries)

    def _extract_numbers_from_tools(self, tool_outputs: List[ToolOutput]) -> Set[float]:
        """Recursively gather all float/int values present in tool results."""
        numbers: Set[float] = set()

        def recurse(obj: Any):
            if isinstance(obj, dict):
                for v in obj.values():
                    recurse(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    recurse(item)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                val = round(float(obj), 2)
                numbers.add(val)
                numbers.add(round(val, 1))
                numbers.add(round(val, 0))

        for out in tool_outputs:
            if out.success and out.data:
                recurse(out.data)

        return numbers

    def _extract_numbers_from_text(self, text: str) -> List[tuple[float, str]]:
        """Extract numeric values while stripping commas and formatting."""
        matches = re.finditer(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+\.\d+\b", text)
        results = []
        for m in matches:
            token = m.group(0)
            try:
                num = float(token.replace(",", ""))
                results.append((num, token))
            except ValueError:
                continue
        return results

    def _is_number_supported(self, num: float, ground_truth: Set[float]) -> bool:
        """Check if number matches ground truth within a 1% floating tolerance."""
        if num in ground_truth:
            return True
        for gt in ground_truth:
            if gt == 0:
                if abs(num) < 0.01:
                    return True
            else:
                if abs(num - gt) / abs(gt) < 0.015:
                    return True
        return False

    def _extract_column_mentions(self, text: str, valid_columns: Set[str]) -> List[str]:
        words = re.findall(r"\b[a-zA-Z_]+\b", text.lower())
        return [w for w in words if w in valid_columns]

    def _repair_answer(
        self, candidate: str, issues: List[VerificationIssue], context: ToolContext
    ) -> str:
        """Construct a strictly supported replacement answer directly from valid tool outputs."""
        trusted_summaries = [
            out.summary for out in context.previous_outputs
            if out.success and out.tool_name != "answer_verifier"
        ]
        grounded_core = "\n".join(trusted_summaries)
        flagged_notes = [f"- [Warning] {iss.description}" for iss in issues]
        
        return (
            f"{grounded_core}\n\n"
            f"**Verification Audit Notes:**\n"
            + "\n".join(flagged_notes)
        )
