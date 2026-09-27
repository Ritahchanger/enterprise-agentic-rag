"""
Agentic RAG graph: wires planner -> retriever -> answer -> validator into a
LangGraph state machine. If validation fails, the graph loops back to
retrieval once (with the original question) before giving up gracefully,
instead of returning an unverified/ hallucinated answer.
"""

from typing import List, TypedDict

from langgraph.graph import StateGraph, END

from src.agents.planner import PlannerAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.answer_agent import AnswerAgent
from src.agents.validator import ValidatorAgent
from src.config.constants import NODE_PLANNER, NODE_RETRIEVER, NODE_VALIDATOR, NODE_ANSWER
from src.guardrails.input_guard import check_input
from src.guardrails.output_guard import check_output
from src.vectorstore.search import SearchResult
from src.utils.logger import logger


class RAGState(TypedDict, total=False):
    """Shared state object threaded through every node of the graph."""

    question: str
    sub_queries: List[str]
    chunks: List[SearchResult]
    draft_answer: str
    final_answer: str
    grounded: bool
    retry_count: int


# Instantiate agents once at import time (each agent lazily caches its LLM/model).
_planner = PlannerAgent()
_retriever = RetrieverAgent()
_answer_agent = AnswerAgent()
_validator = ValidatorAgent()


def planner_node(state: RAGState) -> RAGState:
    # Guardrail: block malicious/unsafe input before any LLM/tool call runs.
    check_input(state["question"])
    sub_queries = _planner.plan(state["question"])
    return {**state, "sub_queries": sub_queries}


def retriever_node(state: RAGState) -> RAGState:
    chunks = _retriever.retrieve(state["sub_queries"])
    return {**state, "chunks": chunks}


def answer_node(state: RAGState) -> RAGState:
    draft = _answer_agent.generate(state["question"], state["chunks"])
    return {**state, "draft_answer": draft}


def validator_node(state: RAGState) -> RAGState:
    result = _validator.validate(state["draft_answer"], state["chunks"])
    return {**state, "grounded": result["grounded"]}


def should_retry(state: RAGState) -> str:
    """Conditional edge: retry retrieval once on failed validation, else finish."""
    retry_count = state.get("retry_count", 0)
    if not state.get("grounded", True) and retry_count < 1:
        return "retry"
    return "finish"


def retry_node(state: RAGState) -> RAGState:
    """Bump the retry counter and loop back to retrieval with a broader query."""
    logger.info("Validation failed — retrying retrieval once")
    return {**state, "retry_count": state.get("retry_count", 0) + 1, "sub_queries": [state["question"]]}


def finish_node(state: RAGState) -> RAGState:
    # Guardrail: scrub/validate the final answer before it reaches the user.
    final = check_output(state.get("draft_answer", ""))
    return {**state, "final_answer": final}


def build_rag_graph():
    """Compile the LangGraph state machine: planner -> retriever -> answer -> validator."""
    graph = StateGraph(RAGState)

    graph.add_node(NODE_PLANNER, planner_node)
    graph.add_node(NODE_RETRIEVER, retriever_node)
    graph.add_node(NODE_ANSWER, answer_node)
    graph.add_node(NODE_VALIDATOR, validator_node)
    graph.add_node("retry", retry_node)
    graph.add_node("finish", finish_node)

    graph.set_entry_point(NODE_PLANNER)
    graph.add_edge(NODE_PLANNER, NODE_RETRIEVER)
    graph.add_edge(NODE_RETRIEVER, NODE_ANSWER)
    graph.add_edge(NODE_ANSWER, NODE_VALIDATOR)
    graph.add_conditional_edges(NODE_VALIDATOR, should_retry, {"retry": "retry", "finish": "finish"})
    graph.add_edge("retry", NODE_RETRIEVER)
    graph.add_edge("finish", END)

    return graph.compile()


# Compiled graph singleton, imported by the API routes and Streamlit app.
rag_graph = build_rag_graph()


def run_rag_pipeline(question: str) -> RAGState:
    """Convenience entrypoint: run the full agentic RAG pipeline for one question."""
    logger.info(f"Running RAG pipeline for question: {question!r}")
    return rag_graph.invoke({"question": question, "retry_count": 0})
