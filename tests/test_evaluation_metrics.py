"""
Regression tests for evaluation/reporting metrics.

Run:
    python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import evaluate_examples  # noqa: E402
import generate_report  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = [
        "Supported",
        "Supported",
        "Supported",
        "Not Supported",
        "Partial",
    ]
    predicted = [
        "Supported",
        "Supported",
        "Partial",
        "Supported",
        "Supported",
    ]

    # True precision is TP / predicted Supported = 2 / (2 TP + 2 FP).
    # The old bug calculated gold-Supported recall: 2 / (2 TP + 1 FN).
    assert evaluate_examples._supported_precision(expected, predicted) == 0.5
    assert generate_report._supported_precision(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: Supported precision counts false Supported predictions")
