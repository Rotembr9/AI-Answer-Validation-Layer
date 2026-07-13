"""Regression tests for evaluation metric definitions.

Run:
    python3 tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from evaluate_examples import supported_precision  # noqa: E402
from generate_report import _supported_precision  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Supported"]

    assert supported_precision(expected, predicted) == 1 / 3
    assert _supported_precision(expected, predicted) == 1 / 3


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: Supported precision counts false Supported predictions")
