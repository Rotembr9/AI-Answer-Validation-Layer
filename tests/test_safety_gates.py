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


def test_priority_one_day_scale_sla_never_supported() -> None:
    q = "What first-response window applies to Priority 1 tickets?"
    doc = _doc()
    answers = [
        "Priority 1 tickets receive a first response within one business day.",
        "P1 tickets receive a first response within one business day.",
    ]
    for a in answers:
        r = validate(q, a, doc)
        assert r["verdict"] != "Supported", r


def test_mixed_sla_wrong_urgent_clause_never_supported() -> None:
    q = "Urgent vs non-urgent support SLAs?"
    answers = [
        (
            "Non-urgent tickets get a first response within 2 business days; "
            "urgent Severity 1 tickets get one business day."
        ),
        "Urgent and non-urgent tickets receive a first response within 2 business days.",
    ]
    for a in answers:
        r = validate(q, a, _doc())
        assert r["verdict"] != "Supported", r


def test_correct_mixed_sla_stays_supported() -> None:
    q = "Urgent vs non-urgent support SLAs?"
    a = (
        "Urgent tickets get a response within 4 business hours; "
        "non-urgent tickets get a response within 2 business days."
    )
    r = validate(q, a, _doc())
    assert r["verdict"] == "Supported", r


def test_closed_nonurgent_wrong_sla_never_supported() -> None:
    q = "What is the first-response SLA for nonurgent tickets?"
    a = "Nonurgent tickets receive a first response within 4 business hours."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_remote_approval_after_week_begins_never_supported() -> None:
    q = "What if I want a fourth remote day in one week?"
    a = "A fourth remote day requires written manager approval after that week begins."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_remote_cap_understatement_never_supported() -> None:
    q = "How many remote days per week need no manager sign-off?"
    a = "You can work remotely one day per week without extra approval."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_remote_permission_answer_not_penalized_by_unrelated_exclusivity() -> None:
    q = "Who can work remotely up to 3 days without approval?"
    a = "Employees may work remotely up to 3 days per week without extra approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_priority_one_day_scale_sla_never_supported()
    test_mixed_sla_wrong_urgent_clause_never_supported()
    test_correct_mixed_sla_stays_supported()
    test_closed_nonurgent_wrong_sla_never_supported()
    test_remote_approval_after_week_begins_never_supported()
    test_remote_cap_understatement_never_supported()
    test_remote_permission_answer_not_penalized_by_unrelated_exclusivity()
    print("ok: safety gates block false Supported regressions")
