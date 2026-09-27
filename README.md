# Enterprise Agentic RAG

An open-source, production-shaped **agentic RAG** (Retrieval-Augmented Generation) system built with **LangChain** + **LangGraph**, using:

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **LLM:** an open-source **~20B parameter model** (`openai/gpt-oss-20b`) served with low latency through **Groq**
- **Vector store:** [ChromaDB](https://www.trychroma.com/) — runs 100% locally and in-process (embedded, on-disk), no server or Docker required
- **UI:** Streamlit (chat + document ingestion)
- **API:** FastAPI (optional decoupled backend)
- **Orchestration:** LangGraph state machine — planner → hybrid retriever → reranker → answer generator → validator (with a self-correcting retry loop)
- **Guardrails:** input/output guardrails + PII redaction (Presidio)
- **Evaluation:** RAGAS metrics (faithfulness, answer relevancy, context precision/recall) + custom retrieval metrics (precision@k, recall@k, MRR)
- **Monitoring:** Prometheus metrics + lightweight tracing

This project runs entirely on your machine — no Docker, no external vector database service. The only network calls are to Groq (LLM inference) and, optionally, Hugging Face (to download the embedding model on first run).

## Architecture

```
enterprise-agentic-rag/
├── app/                     # FastAPI app (optional API) + Streamlit is app.py at the root
│   ├── main.py
│   └── api/
│       ├── routes.py
│       └── schemas.py
├── src/
│   ├── config/              # settings.py (env-driven), constants.py
│   ├── ingestion/            # loaders, OCR, table/image extraction, cleaning, chunking, metadata
│   ├── embeddings/            # all-MiniLM embedding service
│   ├── vectorstore/           # local Chroma client, indexing, search
│   ├── retrieval/             # vector, keyword (local BM25), hybrid, cross-encoder reranker
│   ├── agents/                # planner, retriever, validator, answer agents
│   ├── graph/                 # rag_graph.py — the LangGraph state machine
│   ├── generation/             # llm.py (Groq 20B model), prompts.py
│   ├── guardrails/             # input/output guards, PII filter
│   ├── evaluation/             # RAGAS eval, retrieval metrics, eval dataset
│   ├── monitoring/             # Prometheus metrics, tracing
│   └── utils/                  # logger, helpers
├── data/
│   ├── raw/                 # uploaded source documents
│   ├── processed/
│   ├── evaluation/
│   └── chroma_db/           # local ChromaDB persistence (created automatically)
├── tests/                   # pytest unit + graph tests
├── notebooks/               # evaluation notebook
├── requirements.txt
├── .env.example           # template for .env (copy and fill in keys)
└── app.py                   # Streamlit entrypoint
```

## How it works (agentic flow)

1. **Ingestion**: documents (PDF/DOCX/TXT/CSV/XLSX/images) are loaded, OCR'd if scanned, tables extracted, cleaned, chunked, and embedded with `all-MiniLM-L6-v2`, then indexed into a local ChromaDB collection persisted on disk.
2. **Planner agent**: decomposes the user's question into 1-3 focused sub-queries.
3. **Retriever agent**: runs **hybrid retrieval** (dense vector search via Chroma + local in-memory BM25 keyword search) for each sub-query, fuses/deduplicates results, then **reranks** with a cross-encoder.
4. **Answer agent**: generates a grounded, citation-aware answer using the retrieved context and the Groq-hosted 20B LLM.
5. **Validator agent**: fact-checks the draft answer against the retrieved context; if not grounded, the graph loops back to retrieval once before finishing.
6. **Guardrails**: input is screened before the pipeline runs; output is PII-redacted and length-bounded before being returned.

## Setup

### Prerequisites

- **Python 3.12**
- **System packages for OCR / scanned PDFs:** Tesseract and Poppler

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr poppler-utils
# macOS
brew install tesseract poppler
```

- A **Groq API key** — get one free at https://console.groq.com

### 1. Clone & configure environment

```bash
git clone https://github.com/Ritahchanger/enterprise-agentic-rag.git
cd enterprise-agentic-rag
cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here      # required
HF_TOKEN=your_huggingface_token_here     # optional, higher HF download rate limits
```

Everything else in `.env.example` already has working local defaults.

> **Security note:** `.env` is git-ignored — never commit it or paste real API keys into shared files/chats. If a key has ever been exposed, rotate it immediately from the provider's dashboard.

### 2. Create a virtual environment & install dependencies

```bash
python3 -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # NLP model used by Presidio PII redaction
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501. Chroma persists its index under `data/chroma_db/` automatically — no separate database process to start or stop.

On first run the app downloads the embedding model (`all-MiniLM-L6-v2`) and the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) from Hugging Face, so the first query/ingest takes a little longer.

### 4. (Optional) Run the FastAPI API instead of / alongside Streamlit

```bash
uvicorn app.main:app --reload
```

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check |
| `POST /api/query` | Ask a question through the agentic pipeline |
| `POST /api/ingest` | Ingest documents into the vector store |
| `GET /metrics` | Prometheus metrics |

Interactive API docs at http://localhost:8000/docs.

## Usage

1. Open the Streamlit app, upload documents in the sidebar, click **Ingest documents**.
2. Ask questions in the chat — the agentic pipeline retrieves relevant chunks, generates a grounded answer, and shows sources + a validation indicator.

## Testing

```bash
pytest tests/ -v
```

## Evaluation

Populate `data/evaluation/eval_dataset.json` with labeled `{question, ground_truth}` samples, then run:

```python
from src.evaluation.ragas_eval import run_evaluation
from src.evaluation.evaluation_dataset import load_eval_dataset

results = run_evaluation(load_eval_dataset())
print(results)
```

Or open `notebooks/evaluation.ipynb` for an interactive walkthrough.

## Resetting the local vector store

To wipe all ingested documents and start fresh, just delete the persistence directory while the app isn't running:

```bash
rm -rf data/chroma_db
```

It will be recreated automatically the next time you ingest a document.

## Environment variables reference

All settings are loaded from `.env` (see `src/config/settings.py`); only `GROQ_API_KEY` is required.

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | API key for Groq (serves the open-source 20B LLM) | — (required) |
| `GROQ_MODEL_NAME` | Model name on Groq | `openai/gpt-oss-20b` |
| `HF_TOKEN` | HuggingFace token for model downloads (optional) | — |
| `EMBEDDING_MODEL_NAME` | Sentence-transformers embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIRECTORY` | Local on-disk path for the Chroma index | `./data/chroma_db` |
| `CHROMA_COLLECTION_NAME` | Chroma collection name | `enterprise_documents` |
| `APP_ENV` | Environment name | `development` |
| `API_HOST` / `API_PORT` | FastAPI bind address | `0.0.0.0` / `8000` |
| `STREAMLIT_SERVER_PORT` | Streamlit port | `8501` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (characters) | `800` / `120` |
| `TOP_K_RETRIEVAL` | Candidates retrieved per sub-query | `8` |
| `TOP_K_RERANK` | Chunks kept after reranking | `4` |

## License

Open source — MIT License (adjust as needed for your organization).
