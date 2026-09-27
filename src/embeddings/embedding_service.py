"""
Embedding service: wraps the HuggingFace `all-MiniLM-L6-v2` sentence-transformer
model behind a single, cached interface used by both indexing and query time
to guarantee vector-space consistency.
"""

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import settings
from src.utils.logger import logger


@lru_cache
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Lazily load and cache the all-MiniLM-L6-v2 embedding model (384-dim vectors).
    HF_TOKEN is passed so gated/rate-limited models or private mirrors also work.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model_name}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},  # switch to "cuda" if a GPU is available
        encode_kwargs={"normalize_embeddings": True},  # cosine-similarity ready
    )


class EmbeddingService:
    """Thin convenience wrapper around the cached embedding model."""

    def __init__(self) -> None:
        self.model = get_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of chunk texts at ingestion time."""
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user query at retrieval time."""
        return self.model.embed_query(text)
