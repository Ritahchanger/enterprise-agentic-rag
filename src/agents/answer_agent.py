"""
Answer agent: synthesizes the final, citation-aware answer from the
retrieved & reranked context chunks using the Groq-hosted 20B LLM.
"""

from typing import List

from src.generation.llm import get_llm
from src.generation.prompts import ANSWER_PROMPT
from src.vectorstore.search import SearchResult
from src.guardrails.pii_filter import redact_pii
from src.utils.logger import logger


class AnswerAgent:
    """Generates the final grounded answer with inline source citations."""

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.2)
        self.chain = ANSWER_PROMPT | self.llm

    def generate(self, question: str, chunks: List[SearchResult]) -> str:
        # Build a citation-friendly context block: "[source: <file>] <text>"
        context = "\n\n".join(
            f"[source: {c['metadata'].get('source', 'unknown')}] {c['content']}" for c in chunks
        )

        response = self.chain.invoke({"question": question, "context": context})
        answer = response.content

        # Redact any PII that may have leaked into the generated answer
        # (defense-in-depth on top of input_guard/output_guard).
        answer = redact_pii(answer)

        logger.info("AnswerAgent generated final answer")
        return answer
