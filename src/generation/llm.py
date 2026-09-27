"""
LLM client: wires up the open-source ~20B parameter model (e.g.
`openai/gpt-oss-20b`) served through Groq's low-latency inference API via
LangChain's `ChatGroq` integration. Swappable for any other GROQ_MODEL_NAME.
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from src.config.settings import settings
from src.utils.logger import logger


@lru_cache
def get_llm(temperature: float = 0.1) -> ChatGroq:
    """
    Return a cached ChatGroq client configured for the open-source 20B model.
    GROQ_API_KEY is required (see .env.example) — Groq hosts the model and
    provides fast inference, no local GPU needed.
    """
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set — LLM calls will fail until configured.")

    logger.info(f"Initializing LLM via Groq: {settings.groq_model_name}")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model_name,  # e.g. "openai/gpt-oss-20b"
        temperature=temperature,
        max_tokens=2000,
    )
