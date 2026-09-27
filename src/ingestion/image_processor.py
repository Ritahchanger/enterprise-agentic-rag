"""
Image pre-processing utilities used before OCR to improve text extraction
quality (grayscale, thresholding, deskew). Kept separate from ocr.py so the
image pipeline can be swapped/extended independently.
"""

import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> Image.Image:
    """Grayscale + autocontrast an image before OCR to improve accuracy."""
    gray = ImageOps.grayscale(image)
    return ImageOps.autocontrast(gray)


def binarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    """Simple binary threshold, useful for low-contrast scans."""
    arr = np.array(image.convert("L"))
    arr = np.where(arr > threshold, 255, 0).astype("uint8")
    return Image.fromarray(arr)


def load_and_prepare(file_path: str) -> Image.Image:
    """Full prep pipeline: load -> grayscale/contrast -> binarize, ready for OCR."""
    image = Image.open(file_path)
    image = preprocess_image(image)
    return binarize(image)
