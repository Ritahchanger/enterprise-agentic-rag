"""
OCR module: extracts text from images / scanned PDF pages using Tesseract
via pytesseract. Used for image uploads and as a fallback when a PDF page
has no extractable text layer (i.e. it's a scanned image).
"""

from typing import List

import pytesseract
from PIL import Image
from langchain_core.documents import Document
from pdf2image import convert_from_path

from src.utils.logger import logger


def run_ocr_on_image(file_path: str) -> List[Document]:
    """OCR a single image file (.png/.jpg/.jpeg) and return it as a Document."""
    logger.info(f"Running OCR on image: {file_path}")
    text = pytesseract.image_to_string(Image.open(file_path))
    return [Document(page_content=text, metadata={"source": file_path, "content_type": "ocr_image"})]


def run_ocr_on_pdf_pages(file_path: str) -> List[Document]:
    """
    Fallback OCR for scanned PDFs with no text layer: rasterize each page to
    an image, then OCR it. Only called when `parser.parse_pdf` yields
    empty/near-empty text for a page.
    """
    logger.info(f"Running OCR fallback on scanned PDF: {file_path}")
    pages = convert_from_path(file_path)
    docs: List[Document] = []
    for i, page_image in enumerate(pages):
        text = pytesseract.image_to_string(page_image)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": file_path, "page": i, "content_type": "ocr_pdf_page"},
            )
        )
    return docs
