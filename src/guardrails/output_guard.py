"""
Output guardrails: run on the LLM's draft answer before it's returned to the
user — truncates overly long output and applies PII redaction as a final
defense-in-depth layer (on top of the redaction already done in answer_agent).
"""

from src.config.constants import MAX_OUTPUT_TOKENS
from src.guardrails.pii_filter import redact_pii


def check_output(answer: str) -> str:
    """Sanitize and bound the final answer before returning it to the caller."""
    if not answer:
        return "I don't have enough information in the knowledge base to answer that."

    # Hard cap output length (rough token estimate) to avoid runaway generations.
    max_chars = MAX_OUTPUT_TOKENS * 4
    if len(answer) > max_chars:
        answer = answer[:max_chars] + "\n\n[truncated]"

    return redact_pii(answer)
