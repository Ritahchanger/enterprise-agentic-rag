"""
Chunking: splits cleaned Documents into overlapping chunks sized for the
embedding model's context window, using LangChain's RecursiveCharacterTextSplitter.
Chunk size/overlap are configurable via `src/config/settings.py`.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.ingestion.metadata import attach_chunk_metadata
from src.utils.logger import logger


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller chunks and attach stable chunk_ids
    / metadata so each chunk can be traced back to its source document + page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer splitting on paragraph/sentence boundaries before hard cuts.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for doc in docs:
        split_docs = splitter.split_documents([doc])
        chunks.extend(split_docs)

    chunks = attach_chunk_metadata(chunks)
    logger.info(f"Chunked {len(docs)} document(s) into {len(chunks)} chunk(s)")
    return chunks
