"""Regression tests for evaluation/reporting metric definitions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.generate_report import _run_dataset  # noqa: E402


def test_supported_precision_counts_predicted_supported_rows() -> None:
    dr = _run_dataset("data/examples_holdout.json")
    predicted_supported = [row for row in dr.rows if row["predicted"] == "Supported"]

    expected_precision = sum(
        1 for row in predicted_supported if row["expected"] == "Supported"
    ) / len(predicted_supported)

    assert dr.supported_precision == expected_precision
    assert dr.supported_precision == 1.0


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_rows()
    print("ok: supported precision counts predicted Supported rows")
