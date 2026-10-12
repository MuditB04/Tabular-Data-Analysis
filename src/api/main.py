"""FastAPI application for the Tabular Data Analysis Agent."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from src.agent.service import AgentService
from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)

# Setup directories
UPLOAD_DIR = Path("artifacts/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Tabular Data Analysis Agent API",
    description="Production-minded router-driven AI agent for accurate, grounded tabular analysis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton service instance
service = AgentService()


@app.get("/", include_in_schema=False)
def root():
    """Send visitors to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    """Health status and current loaded dataset."""
    active_name = service.active_metadata.name if service.active_metadata else None
    return HealthResponse(status="ok", version="0.1.0", active_dataset=active_name)


@app.post("/dataset/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new dataset (CSV, XLSX, or Parquet) and register it with the agent."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".csv", ".xlsx", ".xls", ".parquet"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Please upload CSV, XLSX, or Parquet.",
        )

    dest_path = UPLOAD_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        metadata = service.load_dataset(str(dest_path))
        return UploadResponse(
            message=f"Successfully loaded '{metadata.name}'.",
            dataset_name=metadata.name,
            row_count=metadata.row_count,
            column_count=metadata.column_count,
            columns=metadata.columns,
            numeric_columns=metadata.numeric_columns,
            categorical_columns=metadata.categorical_columns,
            datetime_columns=metadata.datetime_columns,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Submit a question to the agent and receive grounded, verified answers."""
    if service.active_df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset currently loaded. Please upload a dataset first via /dataset/upload.",
        )

    try:
        res = service.pipeline.process_query(
            question=request.question,
            df=service.active_df,
            metadata=service.active_metadata,
            user_id=request.user_id,
            custom_sql=request.custom_sql,
        )

        return QueryResponse(
            question=res["question"],
            answer=res["answer"],
            intent=res["route_decision"]["intent"],
            confidence=res["route_decision"]["confidence"],
            tool_plan=res["route_decision"]["tool_plan"],
            tool_outputs=res["tool_outputs"],
            verification=res["verification"],
            artifacts=res["artifacts"],
            status=res["status"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.get("/artifacts/{filename}")
def get_artifact(filename: str):
    """Retrieve rendered chart artifacts."""
    chart_path = Path("artifacts/charts") / filename
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    return FileResponse(chart_path)
