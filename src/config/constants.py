"""
Static, non-secret constants used throughout the application.
Anything that can change at runtime / per-environment belongs in `settings.py`
(loaded from environment variables) instead of here.
"""

# ---- Supported ingestion file types ----
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".png", ".jpg", ".jpeg"]

# ---- Chunking defaults (overridable via env, see settings.py) ----
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# ---- Retrieval defaults ----
DEFAULT_TOP_K_RETRIEVAL = 8
DEFAULT_TOP_K_RERANK = 4
HYBRID_ALPHA = 0.5  # weight between vector (1.0) and keyword (0.0) search

# ---- Agent / graph node names (used by LangGraph state machine) ----
NODE_PLANNER = "planner"
NODE_RETRIEVER = "retriever"
NODE_VALIDATOR = "validator"
NODE_ANSWER = "answer"

# ---- Guardrail limits ----
MAX_INPUT_TOKENS = 4000
MAX_OUTPUT_TOKENS = 2000
BLOCKED_KEYWORDS = ["ignore previous instructions", "system prompt", "jailbreak"]

# ---- Chroma schema ----
# Chroma has no fixed schema (unlike Weaviate) — this is kept only as a
# reference list of the metadata keys the app expects chunks to carry.
CHROMA_TEXT_PROPERTY = "content"
CHROMA_METADATA_PROPERTIES = ["source", "page", "chunk_id", "doc_type"]

# ---- Monitoring ----
METRICS_NAMESPACE = "enterprise_rag"
