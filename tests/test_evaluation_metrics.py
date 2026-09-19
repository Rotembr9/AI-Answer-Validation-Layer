"""Regression checks for safety-report metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.generate_report import _run_dataset  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    result = _run_dataset("tests/fixtures/metrics_precision.json")
    assert result.supported_precision == 0.5
    assert result.any_false_supported == 1
    assert result.strict_false_supported == 0


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: Supported precision counts predicted Supported false positives")
