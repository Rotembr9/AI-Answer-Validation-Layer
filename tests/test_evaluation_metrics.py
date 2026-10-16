"""Regression tests for evaluator safety metrics.

Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from evaluate_examples import supported_precision_score  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Partial", "Supported", "Supported"]
    predicted = ["Supported", "Supported", "Partial", "Partial"]

    # One true predicted-Supported and one false predicted-Supported -> precision 0.5.
    assert supported_precision_score(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: supported precision counts predicted-Supported false positives")
