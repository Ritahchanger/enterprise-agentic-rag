"""
Text cleaning: normalizes whitespace, strips boilerplate (headers/footers,
page numbers), and removes control characters before chunking/embedding.
"""

import re
from typing import List

from langchain_core.documents import Document

# Common boilerplate patterns to strip (page numbers, repeated headers, etc.)
_BOILERPLATE_PATTERNS = [
    re.compile(r"\bPage \d+ of \d+\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # lone page-number lines
]


def clean_text(text: str) -> str:
    """Apply whitespace normalization and boilerplate removal to a text blob."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Collapse repeated whitespace/newlines introduced by PDF extraction.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    """Clean `page_content` in place for a batch of Documents, preserving metadata."""
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    # Drop empty documents left over after cleaning (e.g. blank OCR pages).
    return [d for d in docs if d.page_content]
