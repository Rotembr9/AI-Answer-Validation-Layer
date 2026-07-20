"""
Regression tests for evaluation/reporting metrics.
Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision_score  # noqa: E402
from tests.generate_report import _supported_precision_score  # noqa: E402


def test_supported_precision_counts_only_false_supported_predictions() -> None:
    expected = ["Supported", "Supported", "Partial", "Not Supported"]
    predicted = ["Supported", "Partial", "Supported", "Not Supported"]

    # One true Supported prediction and one false Supported prediction. The missed
    # gold Supported row is a recall miss and must not lower Supported precision.
    assert supported_precision_score(expected, predicted) == 0.5
    assert _supported_precision_score(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_only_false_supported_predictions()
    print("ok: Supported precision counts predicted-Supported false positives")
