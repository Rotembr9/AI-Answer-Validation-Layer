"""Focused tests for headline evaluation metrics.

Run:
    python tests/test_metric_calculations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import evaluate_examples as eval_cli  # noqa: E402
import generate_report as report  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Partial"]

    assert eval_cli.supported_precision(expected, predicted) == 0.5
    assert report._supported_precision(expected, predicted) == 0.5


def test_supported_precision_is_not_supported_recall() -> None:
    expected = ["Supported", "Supported", "Not Supported"]
    predicted = ["Supported", "Partial", "Not Supported"]

    assert eval_cli.supported_precision(expected, predicted) == 1.0
    assert report._supported_precision(expected, predicted) == 1.0


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    test_supported_precision_is_not_supported_recall()
    print("ok: Supported precision metrics count false Supported predictions")
