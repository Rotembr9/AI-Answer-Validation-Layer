"""Regression tests for evaluation/report metrics.

Run:
    python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import generate_report  # noqa: E402


def test_supported_precision_uses_predicted_supported_denominator() -> None:
    """Holdout has 7 true Supported predictions and no false Supported predictions."""
    result = generate_report._run_dataset("data/examples_holdout.json")
    assert result.confusion[("Supported", "Supported")] == 7, result.confusion
    assert result.supported_precision == 1.0, result.supported_precision


if __name__ == "__main__":
    test_supported_precision_uses_predicted_supported_denominator()
    print("ok: supported precision uses predicted Supported denominator")
