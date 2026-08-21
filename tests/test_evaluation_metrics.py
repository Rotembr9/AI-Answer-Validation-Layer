"""
Regression coverage for evaluation/reporting metrics.
Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from generate_report import _run_dataset  # noqa: E402


def test_report_supported_precision_counts_predicted_supported() -> None:
    holdout = _run_dataset("data/examples_holdout.json")
    assert holdout.any_false_supported == 0
    assert holdout.supported_precision == 1.0


def test_cli_supported_precision_counts_predicted_supported() -> None:
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests" / "evaluate_examples.py"),
            str(ROOT / "data" / "examples_holdout.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Supported precision: 1.0000" in p.stdout


if __name__ == "__main__":
    test_report_supported_precision_counts_predicted_supported()
    test_cli_supported_precision_counts_predicted_supported()
    print("ok: Supported precision counts predicted Supported outcomes")
