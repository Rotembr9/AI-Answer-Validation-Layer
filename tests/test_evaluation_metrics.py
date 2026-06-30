"""
Regression checks for evaluation/reporting metrics.

Run: python3 tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tests.generate_report as report  # noqa: E402


def test_supported_precision_counts_false_supported_predictions() -> None:
    """Precision must count gold Partial/NS rows predicted as Supported as false positives."""
    old_loader = report.load_examples_json
    old_validate = report.validate
    predictions = iter(["Supported", "Supported"])

    def fake_loader(_path: Path):
        return "doc", [
            {
                "id": "S",
                "question": "q1",
                "answer": "a1",
                "expected_label": "Supported",
            },
            {
                "id": "P",
                "question": "q2",
                "answer": "a2",
                "expected_label": "Partial",
            },
        ]

    def fake_validate(_question: str, _answer: str, _document: str):
        return {
            "verdict": next(predictions),
            "confidence": 1.0,
            "reason": "synthetic",
            "evidence": [],
        }

    try:
        report.load_examples_json = fake_loader
        report.validate = fake_validate

        result = report._run_dataset("data/examples.json")

        assert result.supported_precision == 0.5, result
        assert result.any_false_supported == 1, result
    finally:
        report.load_examples_json = old_loader
        report.validate = old_validate


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    print("ok: evaluation metrics")
