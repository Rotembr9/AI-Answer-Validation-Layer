"""Regression tests for evaluation/reporting metric definitions."""

from __future__ import annotations

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import evaluate_examples, generate_report  # noqa: E402


def _run_evaluate_examples(dataset: str) -> str:
    old_argv = sys.argv[:]
    buf = StringIO()
    try:
        sys.argv = ["evaluate_examples.py", dataset]
        with redirect_stdout(buf):
            evaluate_examples.main()
    finally:
        sys.argv = old_argv
    return buf.getvalue()


def test_cli_supported_precision_counts_predicted_supported() -> None:
    out = _run_evaluate_examples("data/examples_holdout.json")
    assert "False Supported cases (predicted Supported, gold not Supported):\n  (none)" in out
    assert "Supported precision: 1.0000" in out


def test_report_supported_precision_counts_predicted_supported() -> None:
    holdout = generate_report._run_dataset("data/examples_holdout.json")
    assert holdout.any_false_supported == 0
    assert holdout.supported_precision == 1.0
