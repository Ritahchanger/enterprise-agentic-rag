"""
Planner agent: the first node in the agentic RAG graph. Decomposes a
(potentially complex/multi-part) user question into focused search queries.
"""

import json
from typing import List

from src.generation.llm import get_llm
from src.generation.prompts import PLANNER_PROMPT
from src.utils.logger import logger


class PlannerAgent:
    """Decomposes user questions into retrieval sub-queries."""

    def __init__(self) -> None:
        self.llm = get_llm(temperature=0.0)  # deterministic planning
        self.chain = PLANNER_PROMPT | self.llm

    def plan(self, question: str) -> List[str]:
        response = self.chain.invoke({"question": question})
        try:
            sub_queries = json.loads(response.content)
            if not isinstance(sub_queries, list):
                raise ValueError("Planner did not return a list")
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: if the LLM didn't return clean JSON, just use the
            # original question as a single-query plan so the pipeline never blocks.
            logger.warning(f"Planner JSON parse failed ({e}); falling back to raw question")
            sub_queries = [question]

        logger.info(f"Planner produced {len(sub_queries)} sub-quer(y/ies)")
        return sub_queries
