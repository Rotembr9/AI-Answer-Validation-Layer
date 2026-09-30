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


def test_wrong_exception_group_keeps_eligibility_partial() -> None:
    """An unrelated exception must not hide the source's contractor exclusion."""
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time staff are eligible except interns."
    doc = "Full-time staff only; contractors are not eligible for the remote stipend."
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_named_exception_group_covers_eligibility_exclusion() -> None:
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time staff are eligible except contractors."
    doc = "Full-time staff only; contractors are not eligible for the remote stipend."
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


def test_closed_nonurgent_cannot_use_urgent_hours() -> None:
    q = "What is the first-response time for nonurgent tickets?"
    a = "Nonurgent tickets are answered within 4 business hours."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_priority_one_urgent_day_scale_never_supported() -> None:
    q = "What first-response window applies to Priority 1 tickets?"
    a = "Priority 1 tickets get a first response within one business day."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_mixed_sla_wrong_urgent_clause_never_supported() -> None:
    q = "What are the support SLAs?"
    a = "Non-urgent tickets get 2 business days; urgent Severity 1 tickets get one full business day."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_comma_mixed_sla_wrong_urgent_clause_never_supported() -> None:
    q = "What are the support SLAs?"
    a = "Non-urgent tickets get 2 business days, urgent Severity 1 tickets get one full business day."
    r = validate(q, a, _doc())
    assert r["verdict"] != "Supported", r


def test_urgent_day_scale_blocked_without_severity_literal() -> None:
    q = "How quickly do urgent tickets get a first response?"
    a = "Urgent tickets get a first response within one business day."
    doc = "Support response time: urgent tickets require a first response within 4 business hours."
    r = validate(q, a, doc)
    assert r["verdict"] != "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_wrong_exception_group_keeps_eligibility_partial()
    test_named_exception_group_covers_eligibility_exclusion()
    test_closed_nonurgent_cannot_use_urgent_hours()
    test_priority_one_urgent_day_scale_never_supported()
    test_mixed_sla_wrong_urgent_clause_never_supported()
    test_comma_mixed_sla_wrong_urgent_clause_never_supported()
    test_urgent_day_scale_blocked_without_severity_literal()
    print("ok: H-N08 and H-P10 are not Supported; exclusivity example is Partial")
