"""
Hybrid retriever: merges dense (vector) and sparse (BM25) results using a
weighted score fusion (alpha controls the vector/keyword balance), then
de-duplicates by chunk_id before passing candidates to the reranker.
"""

from typing import List

from src.config.constants import HYBRID_ALPHA
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.vectorstore.search import SearchResult
from src.config.settings import settings
from src.utils.logger import logger


class HybridRetriever:
    """Combines VectorRetriever + KeywordRetriever results via weighted fusion."""

    def __init__(self, alpha: float = HYBRID_ALPHA) -> None:
        self.alpha = alpha  # 1.0 = pure vector, 0.0 = pure keyword
        self.vector_retriever = VectorRetriever()
        self.keyword_retriever = KeywordRetriever()

    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        top_k = top_k or settings.top_k_retrieval

        vector_results = self.vector_retriever.retrieve(query, top_k=top_k * 2)
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k * 2)

        fused: dict[str, SearchResult] = {}

        # Fuse scores per chunk_id: alpha * vector_score + (1 - alpha) * keyword_score
        for r in vector_results:
            key = r["metadata"].get("chunk_id", r["content"][:50])
            fused[key] = {**r, "score": self.alpha * r["score"]}

        for r in keyword_results:
            key = r["metadata"].get("chunk_id", r["content"][:50])
            if key in fused:
                fused[key]["score"] += (1 - self.alpha) * r["score"]
            else:
                fused[key] = {**r, "score": (1 - self.alpha) * r["score"]}

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        logger.debug(f"Hybrid search fused into {len(ranked)} result(s)")
        return ranked
