"""End-to-end integration tests for the full Agent Pipeline and Service."""

import pytest
import pandas as pd
from pathlib import Path

from src.agent.pipeline import AgentPipeline
from src.agent.service import AgentService
from src.memory.store import MemoryStore


def test_pipeline_end_to_end_sales(tmp_path):
    # Initialize service and load sample dataset
    db_file = tmp_path / "test_memory.db"
    service = AgentService(memory_db_path=str(db_file))
    
    csv_path = "data/sample_sales.csv"
    assert Path(csv_path).exists(), "sample_sales.csv should exist in data/"
    
    metadata = service.load_dataset(csv_path)
    assert metadata.row_count > 0

    # 1. Test Aggregation Query
    res_agg = service.query("What is the total revenue by region?")
    assert res_agg["status"] == "success"
    assert "revenue" in res_agg["answer"].lower() or "region" in res_agg["answer"].lower()
    assert "sql_engine" in res_agg["route_decision"]["tool_plan"]

    # 2. Test Ambiguity Detection
    res_amb = service.query("Show me total quantiti")
    assert res_amb["status"] == "needs_clarification"
    assert "quantity" in res_amb["answer"].lower()

    # 3. Test Out-of-Domain Guardrail
    res_ood = service.query("What is the recipe for chocolate brownies?")
    assert res_ood["status"] == "rejected_out_of_domain"
    assert "unrelated" in res_ood["answer"].lower()


def test_memory_store_preferences_and_eviction(tmp_path):
    db_file = tmp_path / "memory.db"
    store = MemoryStore(db_path=str(db_file), ttl_days=1)
    
    prefs = store.get_preferences("user_123")
    prefs.preferred_metric = "median"
    prefs.custom_definitions["gross_profit"] = "revenue - cost"
    store.save_preferences("user_123", prefs)

    reloaded = store.get_preferences("user_123")
    assert reloaded.preferred_metric == "median"
    assert reloaded.custom_definitions.get("gross_profit") == "revenue - cost"

    # Alias testing
    store.set_column_alias("sales", "turnover", "revenue")
    aliases = store.get_column_aliases("sales")
    assert aliases.get("turnover") == "revenue"

    store.close()
