"""Regression tests for evaluation metric calculations."""

from __future__ import annotations

from evaluate_examples import _supported_precision as cli_supported_precision
from generate_report import _supported_precision as report_supported_precision


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial", "Supported"]
    predicted = ["Supported", "Supported", "Supported", "Partial"]

    assert cli_supported_precision(expected, predicted) == 1 / 3
    assert report_supported_precision(expected, predicted) == 1 / 3


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: Supported precision counts false Supported predictions")
