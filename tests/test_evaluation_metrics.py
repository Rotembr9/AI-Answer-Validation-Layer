"""
Regression checks for evaluation metrics.

Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from generate_report import _evaluate  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    result = _evaluate(ROOT / "data" / "examples_holdout.json")
    assert result.strict_false_supported == 0, result.high_risk
    assert result.any_false_supported == 0, result.high_risk
    assert result.supported_precision == 1.0, result.supported_precision


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: Supported precision counts predicted-Supported false positives")
