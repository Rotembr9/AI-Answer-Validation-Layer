"""
Regression tests for evaluation metric helpers.
Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import evaluate_examples, generate_report  # noqa: E402


def test_supported_precision_counts_predicted_supported_only() -> None:
    expected = ["Supported", "Partial", "Not Supported", "Supported"]
    predicted = ["Supported", "Supported", "Partial", "Partial"]

    # One true positive and one false positive. The Supported -> Partial miss is
    # a recall error, not part of the precision denominator.
    assert evaluate_examples.supported_precision_counts(expected, predicted) == (1, 1)
    assert generate_report._supported_precision_counts(expected, predicted) == (1, 1)


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_only()
    print("ok: Supported precision counts predicted-Supported false positives")
