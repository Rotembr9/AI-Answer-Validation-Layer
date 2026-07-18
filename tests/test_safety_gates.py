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


def test_mixed_sla_wrong_urgent_clause_is_not_supported() -> None:
    q = "What are the support SLAs?"
    a = "Non-urgent tickets get 2 business days; urgent Severity 1 tickets get one full business day."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_mixed_sla_correct_clauses_remain_supported() -> None:
    q = "What are the support SLAs?"
    a = "Non-urgent tickets get 2 business days; urgent Severity 1 tickets get 4 business hours."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Supported", r


def test_urgent_sla_day_scale_without_severity_word_is_not_supported() -> None:
    q = "What first-response window applies to urgent tickets?"
    a = "Urgent tickets get a first response within one full business day."
    doc = "Urgent tickets require a first response within 4 business hours. Non-urgent tickets receive a first response within 2 business days."
    r = validate(q, a, doc)
    assert r["verdict"] == "Not Supported", r


def test_nonurgent_wrong_business_day_count_is_not_supported() -> None:
    q = "How quickly must non-urgent tickets get a first reply?"
    for a in (
        "Non-urgent tickets get a first response within one business day.",
        "Non-urgent tickets get a first response within 3 business days.",
    ):
        r = validate(q, a, _doc())
        assert r["verdict"] == "Not Supported", r


def test_remote_day_understatement_is_not_supported() -> None:
    q = "How many remote days per week?"
    a = "Employees may work remotely up to 1 day per week without approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_generic_stipend_eligibility_omission_is_partial() -> None:
    q = "How does the remote stipend work?"
    a = "Full-time employees are eligible."
    doc = "Full-time staff only; contractors are not eligible"
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_mixed_sla_wrong_urgent_clause_is_not_supported()
    test_mixed_sla_correct_clauses_remain_supported()
    test_urgent_sla_day_scale_without_severity_word_is_not_supported()
    test_nonurgent_wrong_business_day_count_is_not_supported()
    test_remote_day_understatement_is_not_supported()
    test_generic_stipend_eligibility_omission_is_partial()
    print("ok: safety gates block critical false Supported cases")
