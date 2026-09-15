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


def test_priority_one_day_scale_sla_never_supported() -> None:
    r = validate(
        "How fast do Priority 1 tickets get a first response?",
        "Priority 1 tickets get a first response within one full business day.",
        _doc(),
    )
    assert r["verdict"] != "Supported", r


def test_p1_day_scale_sla_never_supported() -> None:
    r = validate(
        "What is the P1 first response SLA?",
        "P1 support tickets receive a first response within 1 business day.",
        _doc(),
    )
    assert r["verdict"] != "Supported", r


def test_mixed_sla_wrong_urgent_span_never_supported() -> None:
    r = validate(
        "What are the response times?",
        (
            "Non-urgent tickets receive a first response within 2 business days; "
            "urgent tickets receive a first response within one business day."
        ),
        _doc(),
    )
    assert r["verdict"] != "Supported", r


def test_urgent_hour_sla_without_severity_literal_never_supported() -> None:
    doc = "Support response time: Urgent tickets require a first response within 4 business hours."
    r = validate(
        "How fast do urgent support tickets get a first response?",
        "Urgent tickets get a first response within one full business day.",
        doc,
    )
    assert r["verdict"] != "Supported", r


def test_wrong_remote_no_approval_cap_never_supported() -> None:
    r = validate(
        "How many remote days are allowed each week without approval?",
        "Employees may work remotely one day per week without approval.",
        _doc(),
    )
    assert r["verdict"] != "Supported", r


def test_incomplete_eligibility_is_partial_not_supported() -> None:
    """Positive-only eligibility answer omits doc exclusivity → Partial (not Supported)."""
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time employees are eligible."
    doc = "Full-time staff only; contractors are not eligible"
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_unrelated_exclusivity_marker_does_not_penalize_supported_answer() -> None:
    q = "Who may work remotely?"
    a = "Employees may work remotely up to 3 days per week."
    doc = (
        "Remote work: Employees may work remotely up to 3 days per week.\n"
        "Stipend: Full-time staff only; contractors are not eligible for the remote stipend."
    )
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_priority_one_day_scale_sla_never_supported()
    test_p1_day_scale_sla_never_supported()
    test_mixed_sla_wrong_urgent_span_never_supported()
    test_urgent_hour_sla_without_severity_literal_never_supported()
    test_wrong_remote_no_approval_cap_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_unrelated_exclusivity_marker_does_not_penalize_supported_answer()
    print("ok: safety gates block critical false Supported regressions")
