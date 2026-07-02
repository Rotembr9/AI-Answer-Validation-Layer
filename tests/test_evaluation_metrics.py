"""
Regression tests for evaluation/reporting aggregate metrics.
Run: python -m pytest tests/test_evaluation_metrics.py -q
   or: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision_score  # noqa: E402
from tests.generate_report import _supported_precision_score  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial", "Supported"]
    predicted = ["Supported", "Supported", "Partial", "Partial"]

    # One true Supported prediction and one false Supported prediction.
    assert supported_precision_score(expected, predicted) == 0.5
    assert _supported_precision_score(expected, predicted) == 0.5


def test_supported_precision_zero_when_no_supported_predictions() -> None:
    expected = ["Supported", "Not Supported"]
    predicted = ["Partial", "Not Supported"]

    assert supported_precision_score(expected, predicted) == 0.0
    assert _supported_precision_score(expected, predicted) == 0.0


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    test_supported_precision_zero_when_no_supported_predictions()
    print("ok: Supported precision counts predicted Supported rows")
