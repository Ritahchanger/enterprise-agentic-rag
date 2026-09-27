"""
FastAPI application entrypoint (optional companion API to the Streamlit UI).
Run with: `uvicorn app.main:app --reload`
Exposes /health, /query, /ingest, and /metrics (Prometheus).
"""

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.routes import router
from src.config.settings import settings
from src.utils.logger import logger

app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="LangChain + LangGraph agentic RAG system with hybrid retrieval, "
    "guardrails, and evaluation, backed by all-MiniLM embeddings and an "
    "open-source 20B LLM served via Groq.",
    version="1.0.0",
)

app.include_router(router, prefix="/api")

# Mount Prometheus metrics endpoint for scraping.
app.mount("/metrics", make_asgi_app())


@app.on_event("startup")
def on_startup() -> None:
    logger.info(f"Starting Enterprise Agentic RAG API in '{settings.app_env}' mode")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
