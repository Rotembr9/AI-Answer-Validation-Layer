"""Regression tests for evaluation/report safety metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import evaluate_examples  # noqa: E402
import generate_report  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Partial"]

    assert evaluate_examples.supported_precision_score(expected, predicted) == 0.5
    assert generate_report.supported_precision_score(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: supported precision counts false Supported predictions")
