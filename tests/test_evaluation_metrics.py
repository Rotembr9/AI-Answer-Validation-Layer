"""Regression coverage for evaluation summary metrics.

Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import generate_report  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    examples = [
        {
            "id": "S",
            "question": "supported?",
            "answer": "tp",
            "expected_label": "Supported",
        },
        {
            "id": "P",
            "question": "partial?",
            "answer": "fp",
            "expected_label": "Partial",
        },
        {
            "id": "S_MISS",
            "question": "supported miss?",
            "answer": "fn",
            "expected_label": "Supported",
        },
    ]
    verdicts = {
        "tp": "Supported",
        "fp": "Supported",
        "fn": "Partial",
    }

    original_loader = generate_report.load_examples_json
    original_validate = generate_report.validate

    def fake_loader(_path: Path) -> tuple[str, list[dict[str, Any]]]:
        return "doc", examples

    def fake_validate(_question: str, answer: str, _document: str) -> dict[str, Any]:
        return {
            "verdict": verdicts[answer],
            "confidence": 1.0,
            "reason": "synthetic",
            "evidence": [],
        }

    generate_report.load_examples_json = fake_loader
    generate_report.validate = fake_validate
    try:
        result = generate_report._run_dataset("synthetic.json")
    finally:
        generate_report.load_examples_json = original_loader
        generate_report.validate = original_validate

    assert result.supported_precision == 0.5, result
    assert result.any_false_supported == 1, result


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: supported precision counts predicted Supported false positives")
