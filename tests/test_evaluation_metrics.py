"""Regression tests for evaluator/report metric calculations."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision as cli_supported_precision  # noqa: E402
from tests.generate_report import _supported_precision as report_supported_precision  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Supported", "Supported"]

    assert math.isclose(cli_supported_precision(expected, predicted), 1 / 3)
    assert math.isclose(report_supported_precision(expected, predicted), 1 / 3)


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: Supported precision counts false Supported predictions")
