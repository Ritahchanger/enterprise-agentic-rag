"""
Keyword (sparse) retriever using BM25 over the full local corpus stored in
Chroma. Complements the dense retriever for hybrid search — useful for exact
term / acronym / numeric matches that embeddings sometimes miss.

Runs 100% locally and in-memory (`rank_bm25`) — no external search server.
The BM25 index is lazily (re)built whenever the underlying Chroma
collection's chunk count changes (e.g. after a new ingestion run), and
reused across calls otherwise.
"""

from typing import List, Optional

from rank_bm25 import BM25Okapi

from src.vectorstore.chroma_client import get_chroma_collection
from src.config.settings import settings
from src.vectorstore.search import SearchResult
from src.utils.logger import logger


class KeywordRetriever:
    """Sparse retriever backed by an in-memory BM25 index built from Chroma's corpus."""

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[str] = []
        self._metadatas: List[dict] = []
        self._indexed_count: int = -1

    def _refresh_index_if_stale(self) -> None:
        """Rebuild the BM25 index only if the collection's size has changed."""
        collection = get_chroma_collection()
        current_count = collection.count()

        if current_count == self._indexed_count:
            return

        if current_count == 0:
            self._bm25 = None
            self._documents, self._metadatas = [], []
            self._indexed_count = 0
            return

        data = collection.get(include=["documents", "metadatas"])
        self._documents = data.get("documents", []) or []
        self._metadatas = data.get("metadatas", []) or []

        tokenized_corpus = [doc.lower().split() for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._indexed_count = current_count

        logger.debug(f"Rebuilt local BM25 index over {current_count} chunk(s)")

    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        top_k = top_k or settings.top_k_retrieval
        self._refresh_index_if_stale()

        if not self._bm25:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: List[SearchResult] = [
            {
                "content": self._documents[i],
                "metadata": self._metadatas[i] or {},
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ]
        logger.debug(f"Keyword (BM25) search returned {len(results)} result(s)")
        return results


def build_local_bm25_index(corpus: List[str]) -> BM25Okapi:
    """
    Standalone helper to build a BM25 index over an arbitrary in-memory
    corpus — handy for tests or offline experimentation without touching
    the Chroma collection at all.
    """
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)
