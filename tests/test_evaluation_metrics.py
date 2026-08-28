"""Regression tests for evaluation metric semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from generate_report import _run_dataset  # noqa: E402


def test_supported_precision_counts_predicted_supported_rows() -> None:
    holdout = _run_dataset("data/examples_holdout.json")
    assert holdout.any_false_supported == 0
    assert holdout.supported_precision == 1.0


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_rows()
    print("ok: Supported precision counts predicted Supported rows")
