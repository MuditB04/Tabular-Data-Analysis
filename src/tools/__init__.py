"""Tool registration catalog and discovery."""

from typing import Dict
from src.tools.base import BaseTool
from src.tools.schema_inspector import SchemaInspectorTool
from src.tools.sql_engine import SQLEngineTool
from src.tools.python_stat_analyzer import PythonStatAnalyzerTool
from src.tools.visualization import VisualizationTool
from src.tools.answer_verifier import AnswerVerifierTool

__all__ = [
    "BaseTool",
    "SchemaInspectorTool",
    "SQLEngineTool",
    "PythonStatAnalyzerTool",
    "VisualizationTool",
    "AnswerVerifierTool",
    "get_default_tools",
]


def get_default_tools() -> Dict[str, BaseTool]:
    """Instantiate and return the registry of default tools."""
    tools = [
        SchemaInspectorTool(),
        SQLEngineTool(),
        PythonStatAnalyzerTool(),
        VisualizationTool(),
        AnswerVerifierTool(),
    ]
    return {tool.name: tool for tool in tools}
