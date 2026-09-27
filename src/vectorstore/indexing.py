"""
Indexing: embeds chunk Documents and upserts them into the local ChromaDB
collection in a single batched call. No server round-trips, no network I/O —
everything happens in-process against the on-disk Chroma index.
"""

from typing import List

from langchain_core.documents import Document

from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.chroma_client import get_chroma_collection
from src.utils.logger import logger


def _stringify_metadata(metadata: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool (no None/dict/list)."""
    return {
        k: v if isinstance(v, (str, int, float, bool)) else str(v)
        for k, v in metadata.items()
        if v is not None
    }


def index_documents(chunks: List[Document]) -> int:
    """
    Embed and upsert a batch of chunk Documents into the local Chroma
    collection. Returns the number of chunks successfully indexed.
    """
    if not chunks:
        return 0

    collection = get_chroma_collection()
    embedder = EmbeddingService()

    texts = [c.page_content for c in chunks]
    vectors = embedder.embed_documents(texts)
    ids = [c.metadata.get("chunk_id") or f"chunk-{i}" for i, c in enumerate(chunks)]
    metadatas = [_stringify_metadata(c.metadata) for c in chunks]

    # `upsert` = add-or-update, so re-ingesting a document with stable
    # chunk_ids overwrites the old rows instead of duplicating them.
    collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    logger.info(f"Indexed {len(ids)} chunk(s) into Chroma collection '{collection.name}'")
    return len(ids)
