"""Regression coverage for safety-facing evaluation metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.generate_report import _run_dataset  # noqa: E402


def _write_false_supported_dataset(path: Path) -> None:
    source = "Remote policy: Employees may work remotely up to 3 days per week without extra approval."
    supported_answer = "Employees may work remotely up to 3 days per week without extra approval."
    path.write_text(
        json.dumps(
            {
                "source_document": source,
                "examples": [
                    {
                        "id": "S",
                        "question": "What remote work is allowed?",
                        "answer": supported_answer,
                        "expected_label": "Supported",
                    },
                    {
                        "id": "N",
                        "question": "What remote work is allowed?",
                        "answer": supported_answer,
                        "expected_label": "Not Supported",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_supported_precision_counts_predicted_supported_rows() -> None:
    path = ROOT / "reports" / "test_metrics_false_supported.json"
    try:
        _write_false_supported_dataset(path)

        report_result = _run_dataset(str(path.relative_to(ROOT)))
        assert report_result.supported_precision == 0.5
        assert report_result.strict_false_supported == 1
        assert report_result.any_false_supported == 1

        cli = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "evaluate_examples.py"), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Supported precision: 0.5000" in cli.stdout
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_supported_precision_counts_predicted_supported_rows()
    print("ok: Supported precision counts false Supported predictions")
