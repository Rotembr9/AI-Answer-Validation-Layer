"""Regression tests for evaluation/reporting metric semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import evaluate_examples as cli_eval  # noqa: E402
from tests import generate_report as report_eval  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Partial", "Supported", "Supported"]

    # One correct Supported prediction out of three predicted Supported rows.
    assert cli_eval.supported_precision(expected, predicted) == 1 / 3
    assert report_eval._supported_precision(expected, predicted) == 1 / 3


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: evaluation metrics")
