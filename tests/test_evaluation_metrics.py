"""Regression checks for safety-critical evaluation metrics."""

from __future__ import annotations

from evaluate_examples import supported_precision
from generate_report import _supported_precision


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Partial"]

    assert supported_precision(expected, predicted) == 0.5
    assert _supported_precision(expected, predicted) == 0.5


def test_supported_precision_is_zero_without_supported_predictions() -> None:
    expected = ["Supported", "Not Supported"]
    predicted = ["Partial", "Not Supported"]

    assert supported_precision(expected, predicted) == 0.0
    assert _supported_precision(expected, predicted) == 0.0


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    test_supported_precision_is_zero_without_supported_predictions()
    print("ok: Supported precision counts predicted-Supported false positives")
