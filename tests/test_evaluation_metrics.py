"""Regression tests for evaluation metric definitions."""

from __future__ import annotations

from metrics_utils import predicted_supported_precision


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial", "Supported"]
    predicted = ["Supported", "Supported", "Partial", "Partial"]

    assert predicted_supported_precision(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: evaluation metrics")
