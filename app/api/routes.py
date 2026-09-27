"""
FastAPI routes: exposes the ingestion + query pipeline as an HTTP API,
so the Streamlit UI (or any other client) can consume it over the network
instead of importing the RAG pipeline directly in-process.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List

from app.api.schemas import QueryRequest, QueryResponse, SourceChunk, IngestResponse, HealthResponse
from src.config.settings import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.vectorstore.indexing import index_documents
from src.graph.rag_graph import run_rag_pipeline
from src.guardrails.input_guard import InputGuardError
from src.utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Simple liveness/readiness probe."""
    return HealthResponse(status="ok", app_env=settings.app_env)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the full agentic RAG pipeline for a user question."""
    try:
        state = run_rag_pipeline(request.question)
    except InputGuardError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=500, detail="Internal error while answering the question") from e

    sources = [
        SourceChunk(
            content=c["content"],
            source=c["metadata"].get("source"),
            score=c.get("score"),
        )
        for c in state.get("chunks", [])
    ]
    return QueryResponse(
        answer=state.get("final_answer", ""),
        sources=sources,
        grounded=state.get("grounded", False),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)) -> IngestResponse:
    """Upload and ingest one or more documents into the vector store."""
    loader = DocumentLoader()
    all_chunks = []
    documents_ingested = 0

    for upload in files:
        # Persist the upload to disk temporarily so existing file-path-based
        # loaders (PyPDFLoader, etc.) can read it.
        tmp_path = f"data/raw/{upload.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await upload.read())

        docs = loader.load(tmp_path)
        docs = clean_documents(docs)
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)
        documents_ingested += 1

    indexed_count = index_documents(all_chunks) if all_chunks else 0
    return IngestResponse(documents_ingested=documents_ingested, chunks_indexed=indexed_count)
