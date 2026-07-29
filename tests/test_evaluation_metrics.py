from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.evaluate_examples import supported_precision  # noqa: E402
from tests.generate_report import _supported_precision  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Supported", "Not Supported", "Partial"]
    predicted = ["Supported", "Partial", "Supported", "Supported"]

    # One true Supported prediction and two false Supported predictions.
    assert supported_precision(expected, predicted) == 1 / 3
    assert _supported_precision(expected, predicted) == 1 / 3


def test_supported_precision_ignores_conservative_supported_downgrades() -> None:
    expected = ["Supported", "Supported", "Not Supported"]
    predicted = ["Supported", "Partial", "Not Supported"]

    assert supported_precision(expected, predicted) == 1.0
    assert _supported_precision(expected, predicted) == 1.0


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    test_supported_precision_ignores_conservative_supported_downgrades()
    print("ok: supported precision counts predicted Supported labels")
