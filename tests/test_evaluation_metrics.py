"""Regression tests for evaluation/reporting metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import (
    not_supported_recall_score,
    supported_precision_score,
)


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Not Supported", "Supported", "Partial"]
    predicted = ["Supported", "Supported", "Partial", "Partial"]

    # One true Supported prediction and one false Supported prediction.
    # The missed Supported row affects recall, not precision.
    assert supported_precision_score(expected, predicted) == 0.5


def test_not_supported_recall_counts_actual_not_supported_rows() -> None:
    expected = ["Not Supported", "Not Supported", "Supported"]
    predicted = ["Not Supported", "Partial", "Supported"]

    assert not_supported_recall_score(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    test_not_supported_recall_counts_actual_not_supported_rows()
    print("ok: evaluation metric regressions passed")
