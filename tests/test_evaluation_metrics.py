"""Regression tests for evaluation metric semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision_score  # noqa: E402


def test_supported_precision_uses_predicted_supported_denominator() -> None:
    expected = ["Supported", "Supported", "Partial", "Not Supported"]
    predicted = ["Supported", "Partial", "Supported", "Not Supported"]

    assert supported_precision_score(expected, predicted) == 0.5


if __name__ == "__main__":
    test_supported_precision_uses_predicted_supported_denominator()
    print("ok: Supported precision uses predicted Supported denominator")
