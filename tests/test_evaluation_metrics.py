"""Regression tests for evaluation metric calculations.

Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import evaluate_examples, generate_report  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Not Supported", "Partial", "Supported"]
    predicted = ["Supported", "Supported", "Supported", "Partial"]

    assert evaluate_examples._supported_precision(expected, predicted) == 1 / 3
    assert generate_report._supported_precision(expected, predicted) == 1 / 3


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: Supported precision counts predicted-Supported false positives")
