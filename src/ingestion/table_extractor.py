"""
Table extraction from PDFs using Camelot. Tables are converted to markdown
text so they can flow through the same chunking/embedding pipeline as prose,
while still preserving row/column structure for the LLM.
"""

from typing import List

from src.utils.logger import logger


def extract_tables_from_pdf(file_path: str) -> List[str]:
    """
    Extract every table found in a PDF and return each as a markdown string.
    Returns an empty list gracefully if camelot fails (e.g. scanned PDF,
    missing Ghostscript dependency) so ingestion never hard-fails on tables.
    """
    try:
        import camelot  # imported lazily: heavy optional dependency

        tables = camelot.read_pdf(file_path, pages="all", flavor="lattice")
        markdown_tables = [t.df.to_markdown(index=False) for t in tables]
        logger.info(f"Extracted {len(markdown_tables)} table(s) from {file_path}")
        return markdown_tables
    except Exception as e:  # noqa: BLE001 - tables are best-effort, never block ingestion
        logger.warning(f"Table extraction skipped for {file_path}: {e}")
        return []
