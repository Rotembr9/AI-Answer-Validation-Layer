"""
Guarantee holdout cases H-N08 and H-P10 are never classified as Supported.
Run: python -m pytest tests/test_safety_gates.py -q
   or: python tests/test_safety_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validator import load_examples_json, validate  # noqa: E402


def _doc() -> str:
    d, _ = load_examples_json(ROOT / "data" / "examples.json")
    return d


def _load_h(id_: str) -> tuple[str, str]:
    _, exs = load_examples_json(ROOT / "data" / "examples_holdout.json")
    for e in exs:
        if e["id"] == id_:
            return e["question"], e["answer"]
    raise KeyError(id_)


def test_h_n08_never_supported() -> None:
    q, a = _load_h("H-N08")
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_h_p10_never_supported() -> None:
    q, a = _load_h("H-P10")
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_incomplete_eligibility_is_partial_not_supported() -> None:
    """Positive-only eligibility answer omits doc exclusivity → Partial (not Supported)."""
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time employees are eligible."
    doc = "Full-time staff only; contractors are not eligible"
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_benefit_scoped_eligibility_omissions_are_not_supported() -> None:
    for question in (
        "What about the remote stipend?",
        "Which employees get the stipend?",
    ):
        r = validate(question, "Full-time employees are eligible.", _doc())
        assert r["verdict"] == "Partial", r


def test_non_urgent_sla_wrong_windows_are_not_supported() -> None:
    question = "What is the first-response SLA for non-urgent tickets?"
    for answer in (
        "Nonurgent tickets receive a first response within 4 business hours.",
        "First response within 4 business hours.",
        "First response within 3 business days.",
    ):
        r = validate(question, answer, _doc())
        assert r["verdict"] == "Not Supported", r


def test_priority_urgent_day_scale_answers_are_not_supported() -> None:
    for question, answer in (
        (
            "What is the Priority 1 first-response window?",
            "Priority 1 tickets receive a first response within one business day.",
        ),
        (
            "What is the P1 first-response window?",
            "P1 tickets receive a first response within one business day.",
        ),
    ):
        r = validate(question, answer, _doc())
        assert r["verdict"] == "Not Supported", r


def test_remote_without_approval_day_count_mismatch_is_not_supported() -> None:
    question = "How many remote days are allowed each week without approval?"
    for answer in (
        "You can work remotely up to 4 days per week without extra approval.",
        "You can work remotely up to 2 days per week without extra approval.",
        "You can work remotely up to one day per week without extra approval.",
    ):
        r = validate(question, answer, _doc())
        assert r["verdict"] == "Not Supported", r


def test_mixed_sla_answer_is_checked_per_clause() -> None:
    correct = validate(
        "What are the support SLAs?",
        (
            "Non-urgent tickets receive a first response within 2 business days; "
            "urgent Severity 1 tickets require a first response within 4 business hours."
        ),
        _doc(),
    )
    assert correct["verdict"] == "Supported", correct

    wrong_urgent = validate(
        "What are the support SLAs?",
        (
            "Non-urgent tickets receive a first response within 2 business days; "
            "urgent Severity 1 tickets require a first response within one business day."
        ),
        _doc(),
    )
    assert wrong_urgent["verdict"] == "Not Supported", wrong_urgent


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_benefit_scoped_eligibility_omissions_are_not_supported()
    test_non_urgent_sla_wrong_windows_are_not_supported()
    test_priority_urgent_day_scale_answers_are_not_supported()
    test_remote_without_approval_day_count_mismatch_is_not_supported()
    test_mixed_sla_answer_is_checked_per_clause()
    print("ok: safety gates reject confirmed false-Supported regressions")
