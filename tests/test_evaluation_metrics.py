"""
Regression checks for product-facing evaluation metric definitions.
Run: python3 tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision as cli_supported_precision  # noqa: E402
from tests.generate_report import supported_precision as report_supported_precision  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Supported", "Partial", "Not Supported"]
    predicted = ["Supported", "Partial", "Supported", "Not Supported"]

    # One correct Supported prediction and one false Supported prediction.
    assert cli_supported_precision(expected, predicted) == 0.5
    assert report_supported_precision(expected, predicted) == 0.5


def test_supported_precision_is_perfect_when_no_false_supported_predictions() -> None:
    expected = ["Supported", "Supported", "Partial", "Not Supported"]
    predicted = ["Supported", "Partial", "Partial", "Not Supported"]

    # Missing a gold Supported row is recall loss, not Supported precision loss.
    assert cli_supported_precision(expected, predicted) == 1.0
    assert report_supported_precision(expected, predicted) == 1.0


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    test_supported_precision_is_perfect_when_no_false_supported_predictions()
    print("ok: evaluation metric regressions passed")
