"""
Prometheus metrics: exposes counters/histograms for request volume, latency,
and pipeline errors. Scrape `/metrics` (wired up in `app/main.py`) with
Prometheus/Grafana for production observability.
"""

from prometheus_client import Counter, Histogram

from src.config.constants import METRICS_NAMESPACE

rag_requests_total = Counter(
    f"{METRICS_NAMESPACE}_requests_total", "Total number of RAG queries received"
)

rag_errors_total = Counter(
    f"{METRICS_NAMESPACE}_errors_total", "Total number of RAG pipeline errors", ["stage"]
)

rag_request_latency_seconds = Histogram(
    f"{METRICS_NAMESPACE}_request_latency_seconds", "End-to-end RAG request latency"
)

retrieval_latency_seconds = Histogram(
    f"{METRICS_NAMESPACE}_retrieval_latency_seconds", "Retrieval-stage latency"
)

llm_latency_seconds = Histogram(
    f"{METRICS_NAMESPACE}_llm_latency_seconds", "LLM generation-stage latency"
)
