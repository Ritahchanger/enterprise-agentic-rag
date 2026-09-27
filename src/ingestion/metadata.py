"""
Metadata enrichment: attaches stable chunk ids, timestamps, and any other
metadata needed downstream (filtering, citations, guardrails) to each chunk.
"""

import time
from typing import List

from langchain_core.documents import Document

from src.utils.helpers import generate_chunk_id


def attach_chunk_metadata(chunks: List[Document]) -> List[Document]:
    """Assign a stable, unique chunk_id and ingestion timestamp to every chunk."""
    # Track a per-document running index so ids stay deterministic across re-runs.
    doc_chunk_counters: dict[str, int] = {}

    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id", "unknown")
        idx = doc_chunk_counters.get(doc_id, 0)
        chunk.metadata["chunk_id"] = generate_chunk_id(doc_id, idx)
        chunk.metadata["ingested_at"] = int(time.time())
        doc_chunk_counters[doc_id] = idx + 1

    return chunks
