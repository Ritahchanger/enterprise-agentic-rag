"""
Reranker: re-scores the hybrid retriever's candidate chunks with a
cross-encoder for higher precision, and truncates to the final top-k passed
to the LLM. Cross-encoders are more accurate than embedding similarity
because they jointly attend to (query, chunk) instead of comparing vectors.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import CrossEncoder

from src.config.settings import settings
from src.vectorstore.search import SearchResult
from src.utils.logger import logger

# Lightweight, widely-used cross-encoder reranking model.
_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache
def get_reranker_model() -> CrossEncoder:
    logger.info(f"Loading reranker model: {_RERANKER_MODEL_NAME}")
    return CrossEncoder(_RERANKER_MODEL_NAME)


class Reranker:
    """Cross-encoder reranker used as the final precision pass before generation."""

    def __init__(self) -> None:
        self.model = get_reranker_model()

    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = None) -> List[SearchResult]:
        top_k = top_k or settings.top_k_rerank
        if not candidates:
            return []

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.debug(f"Reranked down to top {len(reranked)} chunk(s)")
        return reranked
