"""
Lightweight tracing: wraps pipeline stages with timing + structured logging
so each request's planner/retriever/answer/validator latencies are visible
in logs even without a full APM tool wired in.
"""

import time
from contextlib import contextmanager

from src.utils.logger import logger


@contextmanager
def trace_stage(stage_name: str):
    """Context manager that logs the start/end/duration of a named pipeline stage."""
    start = time.perf_counter()
    logger.debug(f"[trace] {stage_name} started")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[trace] {stage_name} finished in {elapsed:.3f}s")
