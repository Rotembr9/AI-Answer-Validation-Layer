"""Regression tests for evaluation/reporting metric calculations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import evaluate_examples, generate_report  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Partial"]

    assert evaluate_examples.supported_precision(expected, predicted) == 0.5
    assert generate_report._supported_precision(expected, predicted) == 0.5


def test_supported_precision_zero_when_nothing_predicted_supported() -> None:
    expected = ["Supported", "Not Supported"]
    predicted = ["Partial", "Not Supported"]

    assert evaluate_examples.supported_precision(expected, predicted) == 0.0
    assert generate_report._supported_precision(expected, predicted) == 0.0


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    test_supported_precision_zero_when_nothing_predicted_supported()
    print("ok: Supported precision counts false Supported predictions")
