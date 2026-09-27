"""Generic helper functions shared across modules."""

import hashlib
import time
from pathlib import Path
from typing import Any, Callable


def generate_doc_id(file_path: str) -> str:
    """Deterministic short hash id for a document, used as a stable metadata key."""
    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Stable id for a chunk: `<doc_id>_<index>`, used as the vector store primary key."""
    return f"{doc_id}_{chunk_index}"


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it doesn't exist yet, return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timed(func: Callable) -> Callable:
    """Decorator that logs how long a function took to run (used for tracing)."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from src.utils.logger import logger  # local import avoids circular import

        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper
