"""
Loads/saves the labeled evaluation dataset (question, ground_truth, and
optionally the relevant chunk_ids) used by ragas_eval.py and retrieval_metrics.py.
Stored as JSON under `data/evaluation/`.
"""

import json
from pathlib import Path
from typing import Dict, List

from src.utils.logger import logger

DEFAULT_EVAL_PATH = Path("data/evaluation/eval_dataset.json")


def load_eval_dataset(path: Path = DEFAULT_EVAL_PATH) -> List[Dict]:
    """Load labeled Q&A evaluation samples from disk."""
    if not path.exists():
        logger.warning(f"No evaluation dataset found at {path}, returning empty list")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_eval_dataset(samples: List[Dict], path: Path = DEFAULT_EVAL_PATH) -> None:
    """Persist labeled Q&A evaluation samples to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved {len(samples)} evaluation sample(s) to {path}")
