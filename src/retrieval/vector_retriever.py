"""
Dense (embedding-based) retriever: embeds the user query with all-MiniLM and
runs an ANN search against the local Chroma collection.
"""

from typing import List

from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.search import vector_search, SearchResult
from src.config.settings import settings


class VectorRetriever:
    """Semantic retriever backed by dense embeddings + local Chroma ANN search."""

    def __init__(self) -> None:
        self.embedder = EmbeddingService()

    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        top_k = top_k or settings.top_k_retrieval
        query_vector = self.embedder.embed_query(query)
        return vector_search(query_vector, top_k=top_k)
