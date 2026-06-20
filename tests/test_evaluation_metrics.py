"""Regression tests for evaluation metric accounting."""

from __future__ import annotations

try:
    from .evaluation_metrics import not_supported_recall, supported_precision
except ImportError:  # pragma: no cover - used when run as a script
    from evaluation_metrics import not_supported_recall, supported_precision


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported"]
    predicted = ["Supported", "Supported"]

    assert supported_precision(expected, predicted) == 0.5


def test_supported_precision_zero_without_supported_predictions() -> None:
    expected = ["Supported", "Not Supported"]
    predicted = ["Partial", "Not Supported"]

    assert supported_precision(expected, predicted) == 0.0


def test_not_supported_recall_uses_gold_not_supported_rows() -> None:
    expected = ["Supported", "Not Supported", "Not Supported"]
    predicted = ["Supported", "Supported", "Not Supported"]

    assert not_supported_recall(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    test_supported_precision_zero_without_supported_predictions()
    test_not_supported_recall_uses_gold_not_supported_rows()
    print("ok: evaluation metric accounting")
