"""Standalone test runner providing complete test execution even if pytest is not yet pip-installed."""

import os
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.pipeline import AgentPipeline
from src.agent.service import AgentService
from src.memory.store import MemoryStore
from src.router.models import DatasetMetadata, IntentType, RouteDecision, ToolOutput
from src.router.router import Router
from src.tools.answer_verifier import AnswerVerifierTool
from src.tools.base import ToolContext
from src.tools.python_stat_analyzer import PythonStatAnalyzerTool
from src.tools.schema_inspector import SchemaInspectorTool
from src.tools.sql_engine import SQLEngineTool
from src.tools.visualization import VisualizationTool


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.meta = DatasetMetadata(
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
        self.router = Router()

    def test_intent_aggregation(self):
        dec = self.router.route("What is the total revenue by region?", self.meta)
        self.assertEqual(dec.intent, IntentType.AGGREGATION)
        self.assertIn("sql_engine", dec.tool_plan)
        self.assertIn("answer_verifier", dec.tool_plan)

    def test_intent_lookup(self):
        dec = self.router.route("Show me the first 5 orders for Laptop", self.meta)
        self.assertEqual(dec.intent, IntentType.LOOKUP)
        self.assertIn("sql_engine", dec.tool_plan)

    def test_intent_trend(self):
        dec = self.router.route("What is the revenue trend over time by month?", self.meta)
        self.assertEqual(dec.intent, IntentType.TREND)
        self.assertIn("visualization", dec.tool_plan)

    def test_intent_comparison(self):
        dec = self.router.route("Compare the average unit price across categories", self.meta)
        self.assertEqual(dec.intent, IntentType.COMPARISON)

    def test_intent_quality_check(self):
        dec = self.router.route("Check for missing values and null counts", self.meta)
        self.assertEqual(dec.intent, IntentType.QUALITY_CHECK)
        self.assertIn("schema_inspector", dec.tool_plan)

    def test_ambiguity_detection_fuzzy_typo(self):
        dec = self.router.route("Show me total revenu", self.meta)
        self.assertTrue(dec.needs_clarification)
        self.assertEqual(dec.intent, IntentType.AMBIGUOUS)
        self.assertIn("revenue", dec.clarification_question.lower())

    def test_out_of_domain_detection(self):
        dec = self.router.route("What is the weather in Paris tomorrow?", self.meta)
        self.assertTrue(dec.is_out_of_domain)
        self.assertEqual(dec.intent, IntentType.OUT_OF_DOMAIN)


class TestTools(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "category": ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"],
            "region": ["East", "West", "East", "West", "East", "West", "East", "West", "East", "West"],
            "price": [10.0, 20.0, 15.0, 25.0, 12.0, 18.0, 22.0, 28.0, 14.0, 500.0],
            "quantity": [1, 2, 1, 3, 2, 1, 4, 2, 1, None],
        })
        self.meta = DatasetMetadata(
            name="test_data",
            source_path="mock.csv",
            file_format="csv",
            row_count=10,
            column_count=5,
            columns=["order_id", "category", "region", "price", "quantity"],
            dtypes={"order_id": "int64", "category": "object", "region": "object", "price": "float64", "quantity": "float64"},
            numeric_columns=["order_id", "price", "quantity"],
            categorical_columns=["category", "region"],
            datetime_columns=[],
            table_name="test_table",
        )
        self.decision = RouteDecision(
            intent=IntentType.AGGREGATION,
            confidence=0.9,
            tool_plan=["sql_engine", "answer_verifier"],
            target_columns=["price"],
        )
        self.context = ToolContext(
            question="What is the average price?",
            df=self.df,
            metadata=self.meta,
            route_decision=self.decision,
        )

    def test_schema_inspector(self):
        tool = SchemaInspectorTool()
        out = tool.execute(self.context)
        self.assertTrue(out.success)
        self.assertEqual(out.data["total_rows"], 10)
        self.assertEqual(out.data["columns"]["quantity"]["null_count"], 1)

    def test_sql_engine(self):
        tool = SQLEngineTool()
        out = tool.execute(self.context)
        self.assertTrue(out.success)
        self.assertGreater(len(out.data["records"]), 0)

    def test_python_stat_analyzer_outliers(self):
        tool = PythonStatAnalyzerTool()
        self.context.question = "Detect any outliers in price"
        out = tool.execute(self.context)
        self.assertTrue(out.success)
        self.assertGreaterEqual(out.data["outliers"]["price"]["iqr_count"], 1)

    def test_visualization_tool(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = VisualizationTool(artifacts_dir=tmp_dir)
            out = tool.execute(self.context)
            self.assertTrue(out.success)
            self.assertIsNotNone(out.artifact_path)
            self.assertTrue(Path(out.artifact_path).exists())

    def test_answer_verifier(self):
        tool = AnswerVerifierTool()
        self.context.previous_outputs = [
            ToolOutput(
                tool_name="sql_engine",
                success=True,
                data={"result": 66.4},
                summary="Average price is 66.4.",
            )
        ]
        self.context.route_decision.parameters["candidate_answer"] = (
            "The average price is 66.4 and revenue is guaranteed to hit 999999.0 next quarter."
        )
        out = tool.execute(self.context)
        self.assertTrue(out.success)
        self.assertFalse(out.data["is_verified"])
        self.assertTrue(out.data["repair_applied"])


class TestPipelineAndMemory(unittest.TestCase):
    def test_pipeline_queries(self):
        service = AgentService(memory_db_path=":memory:")
        meta = service.load_dataset("data/sample_sales.csv")
        self.assertGreater(meta.row_count, 0)

        # 1. Aggregation
        res_agg = service.query("What is the total revenue by region?")
        self.assertEqual(res_agg["status"], "success")

        # 2. Ambiguity
        res_amb = service.query("Show me total quantiti")
        self.assertEqual(res_amb["status"], "needs_clarification")

        # 3. OOD
        res_ood = service.query("What is the recipe for chocolate cake?")
        self.assertEqual(res_ood["status"], "rejected_out_of_domain")

    def test_memory_store(self):
        store = MemoryStore()
        prefs = store.get_preferences("user_abc")
        prefs.preferred_metric = "median"
        store.save_preferences("user_abc", prefs)
        self.assertEqual(store.get_preferences("user_abc").preferred_metric, "median")
        store.close()


def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestTools))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineAndMemory))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    run_all_tests()
