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


def _assert_never_supported(question: str, answer: str, doc: str | None = None) -> None:
    r = validate(question, answer, doc or _doc())
    assert r["verdict"] != "Supported", r


def test_nonurgent_sla_paraphrases_never_supported() -> None:
    _assert_never_supported(
        "How quickly must non-urgent tickets get a first reply?",
        "Nonurgent tickets receive a first response within 4 business hours.",
    )
    _assert_never_supported(
        "How quickly must non-urgent tickets get a first reply?",
        "Non-urgent tickets get a first response within one business day.",
    )


def test_priority_and_mixed_urgent_day_scale_never_supported() -> None:
    _assert_never_supported(
        "What is the first-response target for a Priority 1 support ticket?",
        "Priority 1 tickets get a first response within 2 business days.",
    )
    _assert_never_supported(
        "What are the first-response targets for non-urgent and urgent tickets?",
        "Non-urgent tickets get 2 business days; urgent Severity 1 tickets get one full business day.",
    )


def test_remote_no_approval_cap_mismatches_never_supported() -> None:
    _assert_never_supported(
        "How many remote days per week are allowed without extra approval?",
        "You can work remotely up to one day per week without extra approval.",
    )
    _assert_never_supported(
        "How many remote days are allowed each week without approval?",
        "You can work remotely up to four days per week without any approval.",
    )


def test_urgent_sla_without_severity_phrase_never_supported() -> None:
    _assert_never_supported(
        "What is the first-response target for urgent tickets?",
        "Urgent tickets require a first response within one full business day.",
        "Urgent tickets require a first response within 4 business hours; "
        "non-urgent tickets within 2 business days.",
    )


def test_urgent_sla_denial_paraphrase_never_supported() -> None:
    _assert_never_supported(
        "Ticket priorities explained?",
        "Non-urgent tickets get a first response in 2 business days; "
        "urgent Severity 1 tickets are faster but the doc does not specify the urgent SLA.",
    )


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_nonurgent_sla_paraphrases_never_supported()
    test_priority_and_mixed_urgent_day_scale_never_supported()
    test_remote_no_approval_cap_mismatches_never_supported()
    test_urgent_sla_without_severity_phrase_never_supported()
    test_urgent_sla_denial_paraphrase_never_supported()
    print("ok: safety gates block critical false Supported regressions")
