"""Regression tests for safety-oriented evaluation metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from generate_report import _run_dataset  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    """Precision must not drop when gold Supported rows are conservatively Partial."""
    result = _run_dataset("data/examples_holdout.json")
    assert result.any_false_supported == 0, result.high_risk
    assert result.supported_precision == 1.0, result


def test_evaluate_examples_reports_predicted_supported_precision() -> None:
    result = subprocess.run(
        [sys.executable, "tests/evaluate_examples.py", "data/examples_holdout.json"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "Supported precision: 1.0000" in result.stdout, result.stdout


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    test_evaluate_examples_reports_predicted_supported_precision()
    print("ok: evaluation metrics count predicted-Supported precision")
