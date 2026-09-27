"""
Low-level format-specific parsers. Each function returns a list of
LangChain `Document` objects (one per page/sheet where relevant).
Table-heavy PDFs are additionally routed through `table_extractor.py`.
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import pandas as pd

from src.ingestion.table_extractor import extract_tables_from_pdf
from src.utils.logger import logger


def parse_pdf(file_path: str) -> List[Document]:
    """Parse a PDF page-by-page using PyPDFLoader, then merge in any extracted tables."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Extract tables separately (e.g. via camelot) and append as extra "documents"
    # so the chunker can treat structured table text differently from prose.
    tables = extract_tables_from_pdf(file_path)
    for i, table_text in enumerate(tables):
        docs.append(
            Document(
                page_content=table_text,
                metadata={"source": file_path, "content_type": "table", "table_index": i},
            )
        )
    return docs


def parse_docx(file_path: str) -> List[Document]:
    """Parse a .docx file into a single Document (LangChain's Docx2txtLoader)."""
    loader = Docx2txtLoader(file_path)
    return loader.load()


def parse_txt(file_path: str) -> List[Document]:
    """Parse a plain text file."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def parse_csv_xlsx(file_path: str) -> List[Document]:
    """
    Parse tabular files (.csv / .xlsx) by converting each row (or the whole
    sheet, for small files) into a text representation the LLM can reason
    over. Large files should be chunked further downstream.
    """
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to parse tabular file {file_path}: {e}")
        return []

    # Represent the table as markdown so structure survives into the LLM prompt.
    content = df.to_markdown(index=False)
    return [Document(page_content=content, metadata={"source": file_path, "content_type": "table"})]
