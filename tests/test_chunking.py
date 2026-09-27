"""Unit tests for the chunker and metadata attachment."""

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.metadata import attach_chunk_metadata


def test_chunk_documents_splits_long_text():
    long_text = "Sentence. " * 500  # long enough to force multiple chunks
    doc = Document(page_content=long_text, metadata={"doc_id": "abc123", "source": "test.txt"})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == "abc123"


def test_attach_chunk_metadata_generates_unique_ids():
    docs = [
        Document(page_content="chunk one", metadata={"doc_id": "doc1"}),
        Document(page_content="chunk two", metadata={"doc_id": "doc1"}),
    ]
    result = attach_chunk_metadata(docs)
    ids = [d.metadata["chunk_id"] for d in result]
    assert len(ids) == len(set(ids)), "chunk_ids should be unique"
