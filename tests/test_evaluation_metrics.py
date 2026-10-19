"""Regression tests for safety-oriented evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from evaluation_metrics import supported_precision


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Partial", "Supported", "Supported"]

    assert supported_precision(expected, predicted) == 1 / 3


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: evaluation metrics")
