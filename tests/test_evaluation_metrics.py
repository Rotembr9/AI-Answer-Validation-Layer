"""Regression tests for safety-critical evaluation metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import generate_report  # noqa: E402


def _write_precision_fixture() -> Path:
    path = ROOT / "tests" / "_tmp_precision_fixture.json"
    policy_line = "Remote work: Employees may work remotely up to 3 days per week without extra approval."
    path.write_text(
        json.dumps(
            {
                "source_document": policy_line,
                "examples": [
                    {
                        "id": "TP",
                        "question": "How many remote days are allowed without approval?",
                        "answer": policy_line,
                        "expected_label": "Supported",
                    },
                    {
                        "id": "FP",
                        "question": "How many remote days are allowed without approval?",
                        "answer": policy_line,
                        "expected_label": "Not Supported",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_supported_precision_counts_predicted_supported_false_positives() -> None:
    fixture = _write_precision_fixture()
    try:
        result = generate_report._run_dataset("tests/_tmp_precision_fixture.json")
        assert result.supported_precision == 0.5

        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "evaluate_examples.py"), str(fixture)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Supported precision: 0.5000" in proc.stdout
    finally:
        fixture.unlink(missing_ok=True)


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_false_positives()
    print("ok: Supported precision counts predicted-Supported false positives")
