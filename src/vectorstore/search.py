"""
Raw vector similarity search against the local Chroma collection. This is the
low-level search primitive; higher-level retrieval strategies (hybrid,
reranking) live in `src/retrieval/`.
"""

from typing import List, TypedDict

from src.config.settings import settings
from src.vectorstore.chroma_client import get_chroma_collection
from src.utils.logger import logger


class SearchResult(TypedDict):
    content: str
    metadata: dict
    score: float


def vector_search(query_vector: List[float], top_k: int = None) -> List[SearchResult]:
    """Run a pure vector (ANN) similarity search and return scored results."""
    top_k = top_k or settings.top_k_retrieval
    collection = get_chroma_collection()

    count = collection.count()
    if count == 0:
        return []

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: List[SearchResult] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        # Convert Chroma's cosine distance to a similarity-style score
        # (higher = better), matching the previous Weaviate-based convention.
        score = 1.0 - (distance or 0.0)
        results.append({"content": content, "metadata": metadata or {}, "score": score})

    logger.debug(f"Vector search returned {len(results)} result(s)")
    return results
