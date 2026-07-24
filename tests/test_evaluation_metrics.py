"""Regression tests for evaluation/reporting metric definitions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from evaluate_examples import supported_precision  # noqa: E402
from generate_report import _supported_precision  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Supported", "Partial", "Not Supported"]
    predicted = ["Supported", "Partial", "Supported", "Not Supported"]

    # One true Supported prediction and one false Supported prediction.
    assert supported_precision(expected, predicted) == 0.5
    assert _supported_precision(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: supported precision counts predicted Supported false positives")
