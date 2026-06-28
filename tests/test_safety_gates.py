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


def test_wrong_sla_unit_numbers_are_not_supported() -> None:
    """Wrong same-unit SLA values must not be hidden by section / Severity numbers."""
    doc = _doc()
    cases = [
        (
            "How fast should urgent support tickets get a first response?",
            "Urgent Severity 1 tickets require a first response within 1 business hour.",
        ),
        (
            "How fast should urgent support tickets get a first response?",
            "Urgent Severity 1 tickets require a first response within 3 business hours.",
        ),
        (
            "How quickly must non-urgent tickets get a first reply?",
            "Non-urgent tickets receive a first response within one business day.",
        ),
    ]
    for q, a in cases:
        r = validate(q, a, doc)
        assert r["verdict"] == "Not Supported", r


def test_priority_one_day_scale_sla_is_not_supported() -> None:
    """Priority-1 phrasing still refers to the urgent SLA and must use hours, not days."""
    r = validate(
        "What first-response window applies to urgent tickets?",
        "Priority 1 tickets get a first response within one full business day.",
        _doc(),
    )
    assert r["verdict"] == "Not Supported", r


def test_wrong_remote_limit_number_is_not_supported() -> None:
    """A lower/incorrect cap is not Supported just because that number appears in the policy."""
    r = validate(
        "How many remote days are allowed each week without approval?",
        "You can work remotely up to 1 day per week without extra approval.",
        _doc(),
    )
    assert r["verdict"] == "Not Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_wrong_sla_unit_numbers_are_not_supported()
    test_priority_one_day_scale_sla_is_not_supported()
    test_wrong_remote_limit_number_is_not_supported()
    print(
        "ok: H-N08/H-P10 and numeric-unit contradictions are not Supported; "
        "exclusivity example is Partial"
    )
