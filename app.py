"""
Streamlit UI: the primary user-facing entrypoint for the Enterprise Agentic
RAG system. Runs the ingestion pipeline (upload -> clean -> chunk -> index)
and the agentic RAG query graph (planner -> retriever -> answer -> validator)
directly in-process (no need to run the FastAPI server separately, though
`app/main.py` is available if you want a decoupled API instead).

Run with:  streamlit run app.py
"""

import os
import streamlit as st

from src.config.settings import settings
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.vectorstore.indexing import index_documents
from src.graph.rag_graph import run_rag_pipeline
from src.guardrails.input_guard import InputGuardError
from src.utils.logger import logger

st.set_page_config(page_title="Enterprise Agentic RAG", page_icon="🧠", layout="wide")


def _check_required_secrets() -> None:
    """Warn the user in the UI if required API keys haven't been configured."""
    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.hf_token:
        missing.append("HF_TOKEN")
    if missing:
        st.warning(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Set them in your `.env` file (see `.env.example`) before running queries.",
            icon="⚠️",
        )


def _ingest_uploaded_files(uploaded_files) -> None:
    """Persist uploads to data/raw, then run them through the full ingestion pipeline."""
    loader = DocumentLoader()
    all_chunks = []
    os.makedirs("data/raw", exist_ok=True)

    with st.spinner("Ingesting documents (loading -> cleaning -> chunking)..."):
        for uploaded_file in uploaded_files:
            raw_path = os.path.join("data/raw", uploaded_file.name)
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                docs = loader.load(raw_path)
                docs = clean_documents(docs)
                chunks = chunk_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Failed to ingest {uploaded_file.name}")
                st.error(f"Failed to ingest {uploaded_file.name}: {e}")

    if all_chunks:
        with st.spinner(f"Embedding + indexing {len(all_chunks)} chunk(s) into ChromaDB (local)..."):
            indexed = index_documents(all_chunks)
        st.success(f"Indexed {indexed} chunk(s) from {len(uploaded_files)} document(s).")


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("📁 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, CSV, XLSX, PNG, JPG)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "csv", "xlsx", "png", "jpg", "jpeg"],
        )
        if uploaded_files and st.button("Ingest documents", type="primary"):
            _ingest_uploaded_files(uploaded_files)

        st.divider()
        st.caption(
            f"**Embedding model:** {settings.embedding_model_name}\n\n"
            f"**LLM (via Groq):** {settings.groq_model_name}\n\n"
            f"**Vector store:** ChromaDB (local, persisted at "
            f"`{settings.chroma_persist_directory}`)"
        )


def _render_chat() -> None:
    st.title("🧠 Enterprise Agentic RAG")
    st.caption(
        "LangChain + LangGraph agentic RAG — hybrid retrieval (vector + keyword) with "
        "reranking, guardrails, and a self-validating answer loop."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay chat history on every rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"- **{src.get('source', 'unknown')}** (score: {src.get('score', 0):.3f})")

    question = st.chat_input("Ask a question about your ingested documents...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Planning -> retrieving -> generating -> validating..."):
                state = run_rag_pipeline(question)
            answer = state.get("final_answer", "I couldn't generate an answer.")
            grounded = state.get("grounded", False)
            sources = [
                {"source": c["metadata"].get("source", "unknown"), "score": c.get("score", 0)}
                for c in state.get("chunks", [])
            ]

            st.markdown(answer)
            if not grounded:
                st.caption("⚠️ This answer may not be fully grounded in the retrieved documents.")
            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.markdown(f"- **{src['source']}** (score: {src['score']:.3f})")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except InputGuardError as e:
            st.error(f"Input rejected by guardrails: {e}")
        except Exception as e:  # noqa: BLE001
            logger.exception("RAG pipeline failed in Streamlit app")
            st.error(f"Something went wrong: {e}")


def main() -> None:
    _check_required_secrets()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
