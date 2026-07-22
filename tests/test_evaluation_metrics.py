from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import evaluate_examples  # noqa: E402
import generate_report  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Supported", "Partial", "Not Supported"]
    predicted = ["Supported", "Partial", "Supported", "Supported"]

    assert evaluate_examples.supported_precision(expected, predicted) == 1 / 3
    assert generate_report.supported_precision(expected, predicted) == 1 / 3


def test_supported_precision_ignores_supported_false_negatives() -> None:
    expected = ["Supported", "Supported", "Partial"]
    predicted = ["Supported", "Partial", "Partial"]

    assert evaluate_examples.supported_precision(expected, predicted) == 1.0
    assert generate_report.supported_precision(expected, predicted) == 1.0


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    test_supported_precision_ignores_supported_false_negatives()
    print("ok: supported precision metrics")
