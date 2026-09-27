"""
Retriever agent: runs hybrid retrieval + reranking for every sub-query
produced by the planner, then merges/deduplicates the final candidate set.
"""

from typing import List

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.vectorstore.search import SearchResult
from src.utils.logger import logger


class RetrieverAgent:
    """Orchestrates hybrid retrieval + reranking across one or more sub-queries."""

    def __init__(self) -> None:
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

    def retrieve(self, sub_queries: List[str]) -> List[SearchResult]:
        all_candidates: dict[str, SearchResult] = {}

        for sub_query in sub_queries:
            candidates = self.hybrid_retriever.retrieve(sub_query)
            for c in candidates:
                key = c["metadata"].get("chunk_id", c["content"][:50])
                # Keep the highest-scoring occurrence if the same chunk
                # surfaces for multiple sub-queries.
                if key not in all_candidates or c["score"] > all_candidates[key]["score"]:
                    all_candidates[key] = c

        merged_query = " ".join(sub_queries)
        final = self.reranker.rerank(merged_query, list(all_candidates.values()))
        logger.info(f"RetrieverAgent returning {len(final)} final chunk(s)")
        return final
