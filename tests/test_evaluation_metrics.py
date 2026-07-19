"""Regression tests for evaluation/reporting safety metrics.

Run: python tests/test_evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import evaluate_examples  # noqa: E402
import generate_report  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    expected = ["Supported", "Not Supported", "Partial", "Supported"]
    predicted = ["Supported", "Supported", "Supported", "Partial"]

    score = evaluate_examples.supported_precision_score(expected, predicted)

    assert score == 1 / 3, score


def test_report_supported_precision_counts_predicted_supported_false_positives() -> None:
    examples = [
        {"id": "S", "question": "q1", "answer": "a1", "expected_label": "Supported"},
        {"id": "N", "question": "q2", "answer": "a2", "expected_label": "Not Supported"},
        {"id": "P", "question": "q3", "answer": "a3", "expected_label": "Partial"},
    ]
    predictions = {
        "q1": "Supported",
        "q2": "Supported",
        "q3": "Partial",
    }
    original_loader = generate_report.load_examples_json
    original_validate = generate_report.validate

    def fake_loader(_path: Path):
        return "doc", examples

    def fake_validate(question: str, _answer: str, _document: str):
        return {
            "verdict": predictions[question],
            "confidence": 0.9,
            "reason": "test",
            "evidence": [],
        }

    try:
        generate_report.load_examples_json = fake_loader
        generate_report.validate = fake_validate
        result = generate_report._run_dataset("ignored.json")
    finally:
        generate_report.load_examples_json = original_loader
        generate_report.validate = original_validate

    assert result.supported_precision == 1 / 2, result
    assert result.strict_false_supported == 1, result
    assert result.any_false_supported == 1, result


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    test_report_supported_precision_counts_predicted_supported_false_positives()
    print("ok: Supported precision counts predicted Supported false positives")
