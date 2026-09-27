"""
Input guardrails: run before any retrieval/LLM call. Blocks prompt-injection
attempts, oversized inputs, and other unsafe user queries.
"""

from src.config.constants import MAX_INPUT_TOKENS, BLOCKED_KEYWORDS


class InputGuardError(ValueError):
    """Raised when a user query fails an input guardrail check."""


def check_input(question: str) -> None:
    """
    Validate a raw user question before it enters the RAG pipeline.
    Raises InputGuardError if the input is unsafe; callers should catch this
    and surface a friendly message instead of running the pipeline.
    """
    if not question or not question.strip():
        raise InputGuardError("Question cannot be empty.")

    # Rough token estimate (~4 chars/token) — good enough for a guardrail, not billing.
    if len(question) / 4 > MAX_INPUT_TOKENS:
        raise InputGuardError("Question is too long.")

    lowered = question.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in lowered:
            raise InputGuardError("Question contains a blocked pattern (possible prompt injection).")
