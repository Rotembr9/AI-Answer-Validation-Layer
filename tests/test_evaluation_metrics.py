"""Regression tests for evaluation/reporting metric calculations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import generate_report as gr  # noqa: E402


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    examples = [
        {"id": "S-OK", "question": "q1", "answer": "pred:supported", "expected_label": "Supported"},
        {"id": "N-FP", "question": "q2", "answer": "pred:supported", "expected_label": "Not Supported"},
        {"id": "P-FP", "question": "q3", "answer": "pred:supported", "expected_label": "Partial"},
        {"id": "S-FN", "question": "q4", "answer": "pred:partial", "expected_label": "Supported"},
    ]

    old_load = gr.load_examples_json
    old_validate = gr.validate
    try:
        gr.load_examples_json = lambda _path: ("source", examples)

        def fake_validate(_question: str, answer: str, _source: str) -> dict:
            verdict = "Supported" if answer == "pred:supported" else "Partial"
            return {"verdict": verdict, "confidence": 0.9, "reason": "", "evidence": []}

        gr.validate = fake_validate
        result = gr._run_dataset("unused.json")
    finally:
        gr.load_examples_json = old_load
        gr.validate = old_validate

    assert result.supported_precision == 1 / 3
    assert result.any_false_supported == 2


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: supported precision counts predicted Supported false positives")
