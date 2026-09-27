"""
Centralized application settings.

Uses `pydantic-settings` so every value can be overridden via environment
variables or a `.env` file at the project root, without touching code.
Import `settings` (singleton instance) anywhere it's needed:

    from src.config.settings import settings
    print(settings.groq_api_key)
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K_RETRIEVAL,
    DEFAULT_TOP_K_RERANK,
)


class Settings(BaseSettings):
    # ---- Secrets / API keys (loaded from .env, NEVER hard-code these) ----
    groq_api_key: str = ""          # used by src/generation/llm.py -> ChatGroq
    groq_model_name: str = "openai/gpt-oss-20b"  # open-source 20B model served by Groq
    hf_token: str = ""              # used to auth HuggingFace Hub downloads

    # ---- Embedding model config ----
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Vector store (ChromaDB — local, embedded, no server/Docker needed) ----
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "enterprise_documents"

    # ---- App / API ----
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_server_port: int = 8501
    log_level: str = "INFO"

    # ---- RAG behavior ----
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL
    top_k_rerank: int = DEFAULT_TOP_K_RERANK

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Cached factory so we parse the .env file only once per process."""
    return Settings()


# Singleton instance imported across the codebase.
settings = get_settings()
