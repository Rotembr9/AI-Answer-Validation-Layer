"""Regression tests for safety-reporting metrics."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import evaluate_examples, generate_report
from tests.metric_helpers import supported_prediction_precision


def test_supported_precision_counts_false_supported_predictions() -> None:
    expected = ["Supported", "Not Supported", "Partial", "Supported"]
    predicted = ["Supported", "Supported", "Supported", "Partial"]

    assert supported_prediction_precision(expected, predicted) == 1 / 3
    assert evaluate_examples.supported_prediction_precision(expected, predicted) == 1 / 3


def test_report_dataset_uses_predicted_supported_precision() -> None:
    examples = [
        {"id": "ok", "question": "q1", "answer": "Supported", "expected_label": "Supported"},
        {"id": "bad-ns", "question": "q2", "answer": "Supported", "expected_label": "Not Supported"},
        {"id": "bad-p", "question": "q3", "answer": "Supported", "expected_label": "Partial"},
    ]

    def fake_validate(_question: str, answer: str, _source_document: str) -> dict:
        return {
            "verdict": answer,
            "confidence": 0.9,
            "reason": "test",
            "evidence": [],
        }

    with (
        patch.object(generate_report, "load_examples_json", return_value=("doc", examples)),
        patch.object(generate_report, "validate", side_effect=fake_validate),
    ):
        result = generate_report._run_dataset("data/examples.json")

    assert result.supported_precision == 1 / 3
    assert result.strict_false_supported == 1
    assert result.any_false_supported == 2


if __name__ == "__main__":
    test_supported_precision_counts_false_supported_predictions()
    test_report_dataset_uses_predicted_supported_precision()
    print("ok: supported precision counts false Supported predictions")
