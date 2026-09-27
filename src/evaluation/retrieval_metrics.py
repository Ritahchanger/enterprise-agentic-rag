"""
Retrieval-only metrics (independent of RAGAS/LLM cost): precision@k,
recall@k, and MRR, computed from a set of expected relevant chunk_ids vs.
what the retriever actually returned. Useful for fast, cheap regression
testing of the retrieval/reranking stack.
"""

from typing import List


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of the top-k retrieved chunk ids that are actually relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of all relevant chunk ids that were found within the top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in relevant_ids if doc_id in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal rank of the first relevant chunk in the retrieved list."""
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0
