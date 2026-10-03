"""Service layer: High-level API and CLI interface to load datasets and query the agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

from src.agent.pipeline import AgentPipeline
from src.router.models import DatasetMetadata


class AgentService:
    """Service abstraction for dataset management and query evaluation."""

    def __init__(self, memory_db_path: Optional[str] = "artifacts/memory.db"):
        self.pipeline = AgentPipeline()
        self.active_df: Optional[pd.DataFrame] = None
        self.active_metadata: Optional[DatasetMetadata] = None

    def load_dataset(self, file_path: str, table_name: str = "dataset") -> DatasetMetadata:
        """Load tabular data from CSV, XLSX, or Parquet."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: CSV, XLSX, Parquet.")

        # Classify column types
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        
        # Check datetime
        dt_cols = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                dt_cols.append(c)
            elif c.lower() in ["date", "timestamp", "datetime", "created_at"]:
                try:
                    df[c] = pd.to_datetime(df[c])
                    dt_cols.append(c)
                    if c in cat_cols:
                        cat_cols.remove(c)
                except Exception:
                    pass

        metadata = DatasetMetadata(
            name=path.stem,
            source_path=str(path.absolute()),
            file_format=suffix.lstrip("."),
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            numeric_columns=num_cols,
            categorical_columns=cat_cols,
            datetime_columns=dt_cols,
            table_name=table_name,
        )

        self.active_df = df
        self.active_metadata = metadata
        return metadata

    def query(self, question: str, user_id: str = "default_user") -> Dict[str, Any]:
        """Submit a natural language question about the active dataset."""
        if self.active_df is None or self.active_metadata is None:
            raise RuntimeError("No active dataset loaded. Please load a dataset first.")

        return self.pipeline.process_query(
            question=question,
            df=self.active_df,
            metadata=self.active_metadata,
            user_id=user_id,
        )


def main():
    """CLI Entrypoint for interactive analysis."""
    parser = argparse.ArgumentParser(description="Tabular Data Analysis Agent CLI")
    parser.add_argument("--data", type=str, help="Path to CSV, XLSX, or Parquet dataset file", required=False)
    parser.add_argument("--query", type=str, help="Single query to execute", required=False)
    args = parser.parse_args()

    service = AgentService()

    if args.data:
        print(f"Loading dataset: {args.data}...")
        meta = service.load_dataset(args.data)
        print(f"Loaded '{meta.name}': {meta.row_count} rows, {meta.column_count} columns.")
        print(f"Columns: {', '.join(meta.columns)}\n")

    if args.query:
        if not service.active_df:
            print("Error: --query requires --data to specify dataset.")
            sys.exit(1)
        res = service.query(args.query)
        print("=== ANSWER ===")
        print(res["answer"])
        print("\n=== ROUTING ===")
        print(f"Intent: {res['route_decision']['intent']} (Conf: {res['route_decision']['confidence']})")
        print(f"Tools Used: {res['route_decision']['tool_plan']}")
        return

    # Interactive mode
    if not args.data:
        dataset_path = input("Enter path to dataset file (CSV/XLSX/Parquet): ").strip()
        meta = service.load_dataset(dataset_path)
        print(f"Loaded '{meta.name}' successfully.")

    print("\nEnter queries (type 'exit' or 'quit' to stop):")
    while True:
        try:
            q = input("\n[Q] > ").strip()
            if q.lower() in ["exit", "quit"]:
                break
            if not q:
                continue

            result = service.query(q)
            print(f"\n[Intent] {result['route_decision']['intent']} (Tools: {', '.join(result['route_decision']['tool_plan'])})")
            print(f"\n[Answer]\n{result['answer']}")
            if result.get("artifacts"):
                print(f"\n[Artifacts]: {', '.join(result['artifacts'])}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
