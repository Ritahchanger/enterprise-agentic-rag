"""
Centralized prompt templates for every LLM-calling stage of the pipeline
(planning, answer generation, validation). Keeping them here makes prompt
tuning a one-file change instead of hunting through agent code.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---- Planner: decomposes a user question into retrieval sub-queries ----
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query planning agent for an enterprise RAG system. "
            "Break the user's question into 1-3 focused search queries that "
            "would retrieve the most relevant document chunks. "
            "Respond ONLY with a JSON list of strings, no other text.",
        ),
        ("human", "{question}"),
    ]
)

# ---- Answer generation: grounded, citation-aware answer synthesis ----
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an enterprise knowledge assistant. Answer the user's question "
            "using ONLY the provided context. If the context does not contain the "
            "answer, say you don't have enough information. Cite sources using "
            "[source: <filename>] after each claim.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# ---- Validator: checks the draft answer is grounded in retrieved context ----
VALIDATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fact-checking agent. Given the context and a draft answer, "
            "determine if every claim in the draft is supported by the context. "
            "Respond ONLY with JSON: {{\"grounded\": true|false, \"reason\": \"...\"}}",
        ),
        ("human", "Context:\n{context}\n\nDraft answer:\n{draft_answer}"),
    ]
)
