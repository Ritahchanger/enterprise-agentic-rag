"""
ChromaDB client wrapper.

Runs 100% locally using Chroma's embedded, on-disk `PersistentClient` —
no server process, no Docker container, no network call required.
Data is persisted under `settings.chroma_persist_directory` and survives
across app restarts.

This project uses local Chroma without an API key.
Embeddings are generated externally using all-MiniLM and supplied
directly to Chroma (same pattern the app previously used with Weaviate).
"""

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from src.config.settings import settings
from src.utils.logger import logger


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    """
    Return a cached local, embedded Chroma client.

    Data lives on disk at `settings.chroma_persist_directory` (created
    automatically if it doesn't exist yet) — nothing to run separately.
    """
    persist_dir = settings.chroma_persist_directory

    logger.info(f"Initializing local ChromaDB client (persist_directory={persist_dir})")

    return chromadb.PersistentClient(path=persist_dir)


@lru_cache
def get_chroma_collection() -> Collection:
    """
    Return (creating if necessary) the Chroma collection used by the app.

    We generate embeddings ourselves using:

        sentence-transformers/all-MiniLM-L6-v2

    so the collection is created without Chroma's own embedding function,
    and we explicitly select cosine similarity to match the normalized
    MiniLM embeddings produced by `EmbeddingService`.
    """
    client = get_chroma_client()
    collection_name = settings.chroma_collection_name

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(
        f"Using Chroma collection '{collection_name}' "
        f"({collection.count()} existing chunk(s) on disk)"
    )

    return collection
