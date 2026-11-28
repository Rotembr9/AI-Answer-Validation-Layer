"""Regression tests for evaluator safety metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import generate_report  # noqa: E402


def test_supported_precision_counts_predicted_supported_precision() -> None:
    """Holdout has zero false Supported rows, so predicted-Supported precision is 1.0."""
    result = generate_report._run_dataset("data/examples_holdout.json")
    assert result.any_false_supported == 0
    assert result.supported_precision == 1.0


def test_cli_supported_precision_matches_predicted_supported_precision() -> None:
    out = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "tests" / "evaluate_examples.py"),
            str(ROOT / "data" / "examples_holdout.json"),
        ],
        cwd=ROOT,
        text=True,
    )
    assert "Supported precision: 1.0000" in out


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_precision()
    test_cli_supported_precision_matches_predicted_supported_precision()
    print("ok: supported precision counts predicted Supported precision")
