"""Router and Planner implementation for tabular question routing.

Features:
- Deterministic keyword and structural scoring with optional ML-classifier fallback.
- Ambiguity detection via Levenshtein / fuzzy similarity against actual dataset columns.
- Plural and morphological suffix normalization to avoid false-positive ambiguities.
- Out-of-domain detection with polite domain redirection.
- Minimal, purpose-fit tool plan construction.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple, Set

from src.router.models import (
    DatasetMetadata,
    IntentType,
    RouteDecision,
)

# Standard tool routing patterns per intent
DEFAULT_TOOL_PLANS: Dict[IntentType, List[str]] = {
    IntentType.LOOKUP: ["schema_inspector", "sql_engine", "answer_verifier"],
    IntentType.AGGREGATION: ["sql_engine", "answer_verifier"],
    IntentType.TREND: ["sql_engine", "visualization", "answer_verifier"],
    IntentType.COMPARISON: ["sql_engine", "python_stat_analyzer", "visualization", "answer_verifier"],
    IntentType.QUALITY_CHECK: ["schema_inspector", "python_stat_analyzer", "answer_verifier"],
    IntentType.AMBIGUOUS: [],
    IntentType.OUT_OF_DOMAIN: [],
}

# Strong keyword indicators for intent classification
INTENT_INDICATORS: Dict[IntentType, List[str]] = {
    IntentType.LOOKUP: [
        "what is", "who is", "show me", "find", "list", "display", "get",
        "which", "who", "give me", "first", "last", "sample"
    ],
    IntentType.AGGREGATION: [
        "total", "sum", "average", "avg", "mean", "median", "count",
        "how many", "max", "maximum", "highest", "min", "minimum",
        "lowest", "most", "least", "stddev", "variance"
    ],
    IntentType.TREND: [
        "over time", "trend", "change", "growth", "monthly", "yearly",
        "quarterly", "daily", "timeline", "time series", "by month",
        "by year", "history", "progression"
    ],
    IntentType.COMPARISON: [
        "compare", "versus", "vs", "difference between", "differ",
        "against", "higher than", "lower than", "better", "worse",
        "breakdown by", "relative to"
    ],
    IntentType.QUALITY_CHECK: [
        "missing", "null", "empty", "nan", "duplicate", "data quality",
        "integrity", "invalid", "outlier", "anomaly", "corrupt", "clean", "zeros"
    ],
}

OUT_OF_DOMAIN_PATTERNS: List[str] = [
    r"\bweather\b", r"\brecipe\b", r"\bpoem\b", r"\bjoke\b", r"\bcapital of\b",
    r"\bmovie\b", r"\bhoroscope\b", r"\bflight status\b", r"\btranslate\b",
    r"\bwho is the president\b", r"\bsing a song\b", r"\bwrite code to\b"
]

STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "what", "which", "who",
    "across"
}


def _lemmatize_token(token: str) -> str:
    """Normalize simple English plurals to match singular column names."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


class Router:
    """Classifies user queries, detects ambiguity and out-of-domain queries, and creates execution plans."""

    def __init__(self, confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold

    def route(self, question: str, metadata: DatasetMetadata) -> RouteDecision:
        """Route a user question given the active dataset metadata."""
        cleaned_question = question.strip()
        lower_q = cleaned_question.lower()

        # Step 1: Out-of-Domain Guardrail Check
        ood_decision = self._check_out_of_domain(cleaned_question, lower_q, metadata)
        if ood_decision:
            return ood_decision

        # Step 2: Ambiguity & Typo Detection
        ambiguity_decision = self._check_ambiguity(cleaned_question, lower_q, metadata)
        if ambiguity_decision:
            return ambiguity_decision

        # Step 3: Intent Classification
        intent, confidence, matched_terms = self._classify_intent(lower_q, metadata)

        # Step 4: Extract Column Targets and Parameters
        target_columns = self._extract_columns(lower_q, metadata)

        # Step 5: Construct Tool Plan
        tool_plan = list(DEFAULT_TOOL_PLANS.get(intent, ["schema_inspector", "answer_verifier"]))

        # Dynamic tool refinement: If user explicitly asks for a chart/plot, add visualization
        if any(term in lower_q for term in ["plot", "chart", "graph", "visualize", "draw"]):
            if "visualization" not in tool_plan:
                tool_plan.insert(-1, "visualization")

        return RouteDecision(
            intent=intent,
            confidence=round(confidence, 2),
            needs_clarification=False,
            is_out_of_domain=False,
            tool_plan=tool_plan,
            reasoning=f"Classified as {intent.value} based on cues: {', '.join(matched_terms) or 'general syntax'}.",
            target_columns=target_columns,
            parameters={
                "matched_terms": matched_terms,
                "requested_visualization": "visualization" in tool_plan
            }
        )

    def _check_out_of_domain(
        self, question: str, lower_q: str, metadata: DatasetMetadata
    ) -> Optional[RouteDecision]:
        """Verify whether the query is completely unrelated to tabular data analysis."""
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, lower_q):
                return RouteDecision(
                    intent=IntentType.OUT_OF_DOMAIN,
                    confidence=0.95,
                    needs_clarification=False,
                    is_out_of_domain=True,
                    rejection_message=(
                        f"The question '{question}' appears unrelated to the loaded dataset "
                        f"'{metadata.name}'. This agent only answers questions regarding the active "
                        f"data table ({len(metadata.columns)} columns: {', '.join(metadata.columns[:5])}...)."
                    ),
                    tool_plan=[],
                    reasoning=f"Matched out-of-domain pattern '{pattern}'."
                )

        words = re.findall(r"\b[a-zA-Z_]{3,}\b", lower_q)
        content_words = [w for w in words if w not in STOP_WORDS]

        col_names = [c.lower() for c in metadata.columns]
        col_tokens = set()
        for c in col_names:
            col_tokens.update(c.split("_"))
            col_tokens.add(_lemmatize_token(c))

        data_syntax_tokens = {
            "data", "dataset", "table", "row", "rows", "column", "columns", "record",
            "records", "csv", "parquet", "count", "sum", "average", "mean", "median",
            "min", "max", "null", "missing", "show", "list", "compare", "trend",
            "stat", "stats", "distribution", "query", "highest", "lowest", "total"
        }

        has_col_overlap = any((w in col_tokens or _lemmatize_token(w) in col_tokens) for w in content_words)
        has_syntax_overlap = any(w in data_syntax_tokens for w in content_words)

        if len(content_words) >= 4 and not has_col_overlap and not has_syntax_overlap:
            return RouteDecision(
                intent=IntentType.OUT_OF_DOMAIN,
                confidence=0.85,
                needs_clarification=False,
                is_out_of_domain=True,
                rejection_message=(
                    f"I cannot relate your question to the available columns in '{metadata.name}'. "
                    f"Available columns are: {', '.join(metadata.columns)}. "
                    f"Please ask a question regarding these fields or request a general dataset summary."
                ),
                tool_plan=[],
                reasoning="Zero overlap found with dataset schema, table tokens, or standard data analysis terminology."
            )

        return None

    def _check_ambiguity(
        self, question: str, lower_q: str, metadata: DatasetMetadata
    ) -> Optional[RouteDecision]:
        """Identify fuzzy typos or ambiguous column references while recognizing natural plurals."""
        words = re.findall(r"\b[a-zA-Z_]{3,}\b", lower_q)
        content_words = [w for w in words if w not in STOP_WORDS]
        col_map = {c.lower(): c for c in metadata.columns}
        normalized_cols = {c.lower().replace("_", ""): c for c in metadata.columns}
        lemmatized_cols = {_lemmatize_token(c.lower()): c for c in metadata.columns}

        for word in content_words:
            lemmatized_word = _lemmatize_token(word)

            # Skip if exact or normalized match or plural form matches
            if word in col_map or word in normalized_cols:
                continue
            if lemmatized_word in col_map or lemmatized_word in lemmatized_cols:
                continue

            # Skip statistical/analytical keywords
            if word in {"avg", "sum", "max", "min", "top", "bot", "null", "rate", "std", "mean", "count"}:
                continue

            # Check for close typo matches
            matches = difflib.get_close_matches(word, list(col_map.keys()), n=2, cutoff=0.75)
            if matches:
                suggested_cols = [col_map[m] for m in matches]
                return RouteDecision(
                    intent=IntentType.AMBIGUOUS,
                    confidence=0.88,
                    needs_clarification=True,
                    clarification_question=(
                        f"Did you mean column '{suggested_cols[0]}' when you referenced '{word}'? "
                        f"Possible matching columns in this dataset: {', '.join(suggested_cols)}."
                    ),
                    is_out_of_domain=False,
                    tool_plan=[],
                    reasoning=f"Ambiguous token '{word}' closely matches dataset column(s): {suggested_cols}."
                )

        return None

    def _classify_intent(
        self, lower_q: str, metadata: DatasetMetadata
    ) -> Tuple[IntentType, float, List[str]]:
        scores: Dict[IntentType, float] = {intent: 0.0 for intent in INTENT_INDICATORS}
        matched_cues: Dict[IntentType, List[str]] = {intent: [] for intent in INTENT_INDICATORS}

        for intent, indicators in INTENT_INDICATORS.items():
            for indicator in indicators:
                if re.search(r"\b" + re.escape(indicator) + r"\b", lower_q):
                    scores[intent] += 1.0
                    matched_cues[intent].append(indicator)

        # Comparison words ("compare", "vs", "versus") carry higher structural weight than sub-metrics
        if any(w in matched_cues[IntentType.COMPARISON] for w in ["compare", "versus", "vs", "difference between"]):
            scores[IntentType.COMPARISON] += 0.6

        if metadata.datetime_columns and any(w in lower_q for w in ["month", "year", "date", "day", "time"]):
            scores[IntentType.TREND] += 1.5

        if re.search(r"\b(by|per|across)\b\s+([a-zA-Z_]+)", lower_q):
            if scores[IntentType.COMPARISON] == 0.0:
                scores[IntentType.AGGREGATION] += 1.0

        best_intent = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_intent]

        if best_score == 0.0:
            return IntentType.LOOKUP, 0.55, ["fallback_lookup"]

        confidence = min(0.95, 0.60 + (best_score * 0.12))
        return best_intent, confidence, matched_cues[best_intent]

    def _extract_columns(self, lower_q: str, metadata: DatasetMetadata) -> List[str]:
        found: List[str] = []
        for col in metadata.columns:
            pattern = r"\b" + re.escape(col.lower().replace("_", " ")) + r"\b"
            raw_pattern = r"\b" + re.escape(col.lower()) + r"\b"
            plural_pattern = r"\b" + re.escape(col.lower()) + r"s\b"
            if re.search(pattern, lower_q) or re.search(raw_pattern, lower_q) or re.search(plural_pattern, lower_q):
                found.append(col)
        return found
