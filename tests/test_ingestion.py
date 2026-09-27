"""Unit tests for the ingestion pipeline (cleaner, loader dispatch)."""

import pytest

from src.ingestion.cleaner import clean_text, clean_documents
from src.ingestion.document_loader import DocumentLoader
from langchain_core.documents import Document


def test_clean_text_removes_extra_whitespace():
    dirty = "Hello   world\n\n\n\nThis is  a test.  Page 3 of 10"
    cleaned = clean_text(dirty)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Page 3 of 10" not in cleaned


def test_clean_documents_drops_empty_docs():
    docs = [Document(page_content="   \n\n  ", metadata={}), Document(page_content="Real content", metadata={})]
    cleaned = clean_documents(docs)
    assert len(cleaned) == 1
    assert cleaned[0].page_content == "Real content"


def test_loader_rejects_unsupported_extension():
    loader = DocumentLoader()
    with pytest.raises(ValueError):
        loader.load("somefile.exe")
