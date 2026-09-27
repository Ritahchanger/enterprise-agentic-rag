"""
Validator agent: fact-checks the draft answer against retrieved context
before it's returned to the user, catching hallucinations early.
"""

import json
from typing import List, TypedDict

from src.generation.llm import get_llm
from src.generation.prompts import VALIDATOR_PROMPT
from src.vectorstore.search import SearchResult
from src.utils.logger import logger


class ValidationResult(TypedDict):
    grounded: bool
    reason: str


class ValidatorAgent:
    """Checks that the draft answer is fully supported by the retrieved chunks."""

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.0)
        self.chain = VALIDATOR_PROMPT | self.llm

    def validate(self, draft_answer: str, chunks: List[SearchResult]) -> ValidationResult:
        context = "\n\n".join(c["content"] for c in chunks)
        response = self.chain.invoke({"context": context, "draft_answer": draft_answer})

        try:
            result: ValidationResult = json.loads(response.content)
        except json.JSONDecodeError:
            # Fail open with a warning rather than blocking the user response entirely.
            logger.warning("Validator JSON parse failed; assuming grounded=True")
            result = {"grounded": True, "reason": "validator_parse_error"}

        logger.info(f"Validation result: {result}")
        return result
