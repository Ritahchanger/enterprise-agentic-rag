"""
PII filter: detects and redacts personally identifiable information (emails,
phone numbers, SSNs, credit cards, etc.) using Microsoft Presidio, so
sensitive data never leaks into logs, generated answers, or evaluation sets.
"""

from functools import lru_cache

from src.utils.logger import logger


@lru_cache
def _get_presidio_engines():
    """Lazily load Presidio's analyzer/anonymizer (heavier optional dependency)."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    return AnalyzerEngine(), AnonymizerEngine()


def redact_pii(text: str) -> str:
    """
    Detect and replace PII entities in `text` with placeholder tags
    (e.g. "<PERSON>", "<EMAIL_ADDRESS>"). Fails open (returns original text)
    if Presidio isn't available, so a missing optional dependency never
    breaks the core RAG pipeline.
    """
    if not text:
        return text

    try:
        analyzer, anonymizer = _get_presidio_engines()
        results = analyzer.analyze(text=text, language="en")
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except Exception as e:  # noqa: BLE001 - PII redaction must never crash the app
        logger.warning(f"PII redaction skipped due to error: {e}")
        return text
