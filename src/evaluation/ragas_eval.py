"""
RAGAS-based end-to-end evaluation: measures faithfulness, answer relevancy,
context precision, and context recall for the RAG pipeline against a labeled
evaluation dataset (see `src/evaluation/evaluation_dataset.py`).
"""

from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from src.graph.rag_graph import run_rag_pipeline
from src.utils.logger import logger


def build_ragas_dataset(eval_samples: List[Dict]) -> Dataset:
    """
    Run the RAG pipeline over each labeled sample and assemble a HuggingFace
    Dataset in the shape RAGAS expects: question / answer / contexts / ground_truth.
    """
    rows = []
    for sample in eval_samples:
        state = run_rag_pipeline(sample["question"])
        rows.append(
            {
                "question": sample["question"],
                "answer": state.get("final_answer", ""),
                "contexts": [c["content"] for c in state.get("chunks", [])],
                "ground_truth": sample["ground_truth"],
            }
        )
    return Dataset.from_list(rows)


def run_evaluation(eval_samples: List[Dict]):
    """Compute RAGAS metrics for a batch of labeled Q&A samples."""
    logger.info(f"Running RAGAS evaluation on {len(eval_samples)} sample(s)")
    dataset = build_ragas_dataset(eval_samples)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    logger.info(f"Evaluation results: {result}")
    return result
