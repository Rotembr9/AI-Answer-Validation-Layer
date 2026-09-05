"""Regression checks for evaluation/reporting metrics.

Run:
    python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision
from tests.generate_report import _supported_precision


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Partial"]

    assert supported_precision(expected, predicted) == 0.5
    assert _supported_precision(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: supported precision counts false Supported predictions")
