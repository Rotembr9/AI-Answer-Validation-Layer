"""Regression tests for evaluation/report metric definitions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tests.generate_report as report  # noqa: E402


def _run_report_with_predictions(expected: list[str], predicted: list[str]) -> float:
    examples = [
        {
            "id": f"T{i}",
            "question": f"Question {i}",
            "answer": f"Answer {i}",
            "expected_label": exp,
        }
        for i, exp in enumerate(expected)
    ]
    predictions = {ex["answer"]: pred for ex, pred in zip(examples, predicted)}

    original_loader = report.load_examples_json
    original_validate = report.validate

    def fake_loader(_path: Path) -> tuple[str, list[dict]]:
        return "source document", examples

    def fake_validate(_question: str, answer: str, _source_document: str) -> dict:
        return {
            "verdict": predictions[answer],
            "confidence": 1.0,
            "reason": "test",
            "evidence": [],
        }

    try:
        report.load_examples_json = fake_loader
        report.validate = fake_validate
        return report._run_dataset("data/examples.json").supported_precision
    finally:
        report.load_examples_json = original_loader
        report.validate = original_validate


def test_supported_precision_ignores_missed_supported_rows() -> None:
    precision = _run_report_with_predictions(
        ["Supported", "Supported", "Partial"],
        ["Supported", "Partial", "Partial"],
    )
    assert precision == 1.0


def test_supported_precision_counts_false_supported_rows() -> None:
    precision = _run_report_with_predictions(
        ["Supported", "Partial"],
        ["Supported", "Supported"],
    )
    assert precision == 0.5


if __name__ == "__main__":
    test_supported_precision_ignores_missed_supported_rows()
    test_supported_precision_counts_false_supported_rows()
    print("ok: Supported precision counts predicted-Supported false positives")
