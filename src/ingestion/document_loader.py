"""
Document loader: entry point of the ingestion pipeline.

Given a file path, it picks the correct loader (LangChain community loaders
or our own parser/ocr/table_extractor helpers) and returns a list of
LangChain `Document` objects with raw text + basic metadata attached.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config.constants import SUPPORTED_EXTENSIONS
from src.ingestion.parser import parse_pdf, parse_docx, parse_csv_xlsx, parse_txt
from src.ingestion.ocr import run_ocr_on_image
from src.utils.helpers import generate_doc_id
from src.utils.logger import logger


class DocumentLoader:
    """Loads raw files from `data/raw/` (or any path) into LangChain Documents."""

    def load(self, file_path: str) -> List[Document]:
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Loading document: {file_path} (type={ext})")
        doc_id = generate_doc_id(file_path)

        # Dispatch to the appropriate low-level parser based on extension.
        if ext == ".pdf":
            docs = parse_pdf(file_path)
        elif ext == ".docx":
            docs = parse_docx(file_path)
        elif ext in (".csv", ".xlsx"):
            docs = parse_csv_xlsx(file_path)
        elif ext == ".txt":
            docs = parse_txt(file_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            # Scanned/image documents go through the OCR pipeline instead.
            docs = run_ocr_on_image(file_path)
        else:
            raise ValueError(f"No loader implemented for {ext}")

        # Attach a shared doc_id to every chunk-less "raw" document so all
        # downstream chunks can trace back to the source file.
        for d in docs:
            d.metadata.setdefault("doc_id", doc_id)
            d.metadata.setdefault("source", file_path)

        return docs

    def load_directory(self, dir_path: str) -> List[Document]:
        """Convenience method used by the Streamlit UI for bulk uploads."""
        all_docs: List[Document] = []
        for path in Path(dir_path).glob("**/*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    all_docs.extend(self.load(str(path)))
                except Exception as e:  # noqa: BLE001 - log and continue batch ingestion
                    logger.error(f"Failed to load {path}: {e}")
        return all_docs
