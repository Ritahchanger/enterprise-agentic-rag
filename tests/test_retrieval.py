"""Unit tests for retrieval metrics and hybrid score fusion logic."""

from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["a", "c"]
    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = ["a", "c", "z"]
    assert round(recall_at_k(retrieved, relevant, k=3), 2) == round(2 / 3, 2)


def test_mean_reciprocal_rank():
    retrieved = ["x", "y", "a"]
    relevant = ["a"]
    assert mean_reciprocal_rank(retrieved, relevant) == pytest_approx(1 / 3)


def pytest_approx(value: float, tol: float = 1e-6):
    """Tiny helper to avoid pulling in pytest.approx for a single comparison."""

    class _Approx:
        def __eq__(self, other):
            return abs(other - value) < tol

    return _Approx()
