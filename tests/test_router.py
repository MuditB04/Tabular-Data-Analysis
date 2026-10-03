"""Unit and integration tests for Router, Ambiguity, OOD detection, and Intent Classification."""

import pytest
import pandas as pd
from src.router.models import DatasetMetadata, IntentType
from src.router.router import Router


@pytest.fixture
def sample_metadata():
    return DatasetMetadata(
        name="sample_sales",
        source_path="data/sample_sales.csv",
        file_format="csv",
        row_count=100,
        column_count=10,
        columns=[
            "order_id", "date", "product", "category", "region",
            "quantity", "unit_price", "revenue", "customer_type", "salesperson"
        ],
        dtypes={
            "order_id": "int64", "date": "object", "product": "object",
            "category": "object", "region": "object", "quantity": "int64",
            "unit_price": "float64", "revenue": "float64",
            "customer_type": "object", "salesperson": "object"
        },
        numeric_columns=["order_id", "quantity", "unit_price", "revenue"],
        categorical_columns=["product", "category", "region", "customer_type", "salesperson"],
        datetime_columns=["date"],
        table_name="sales",
    )


def test_intent_aggregation_routing(sample_metadata):
    router = Router()
    decision = router.route("What is the total revenue by region?", sample_metadata)
    assert decision.intent == IntentType.AGGREGATION
    assert not decision.is_out_of_domain
    assert not decision.needs_clarification
    assert "sql_engine" in decision.tool_plan
    assert "answer_verifier" in decision.tool_plan


def test_intent_lookup_routing(sample_metadata):
    router = Router()
    decision = router.route("Show me the first 5 orders for Laptop", sample_metadata)
    assert decision.intent == IntentType.LOOKUP
    assert "sql_engine" in decision.tool_plan


def test_intent_trend_routing(sample_metadata):
    router = Router()
    decision = router.route("What is the revenue trend over time by month?", sample_metadata)
    assert decision.intent == IntentType.TREND
    assert "visualization" in decision.tool_plan


def test_intent_comparison_routing(sample_metadata):
    router = Router()
    decision = router.route("Compare the average unit price across categories", sample_metadata)
    assert decision.intent == IntentType.COMPARISON
    assert "python_stat_analyzer" in decision.tool_plan or "sql_engine" in decision.tool_plan


def test_intent_quality_check_routing(sample_metadata):
    router = Router()
    decision = router.route("Check for missing values and null counts", sample_metadata)
    assert decision.intent == IntentType.QUALITY_CHECK
    assert "schema_inspector" in decision.tool_plan


def test_ambiguity_detection_fuzzy_typo(sample_metadata):
    router = Router()
    # "revenu" is a typo for "revenue"
    decision = router.route("Show me total revenu", sample_metadata)
    assert decision.needs_clarification is True
    assert decision.intent == IntentType.AMBIGUOUS
    assert "revenue" in decision.clarification_question.lower()


def test_out_of_domain_detection_irrelevant_query(sample_metadata):
    router = Router()
    decision = router.route("What is the weather in Paris tomorrow?", sample_metadata)
    assert decision.is_out_of_domain is True
    assert decision.intent == IntentType.OUT_OF_DOMAIN
    assert "unrelated" in decision.rejection_message.lower() or "weather" in decision.rejection_message.lower()


def test_out_of_domain_detection_non_tabular_chat(sample_metadata):
    router = Router()
    decision = router.route("Can you write a poem about chocolate chip cookies?", sample_metadata)
    assert decision.is_out_of_domain is True
    assert decision.intent == IntentType.OUT_OF_DOMAIN
