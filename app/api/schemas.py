"""Pydantic request/response schemas for the FastAPI routes."""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's natural language question")


class SourceChunk(BaseModel):
    content: str
    source: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    grounded: bool


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    app_env: str
