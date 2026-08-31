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


def test_wrong_remote_without_approval_caps_are_not_supported() -> None:
    doc = _doc()
    q = "How many remote days are allowed each week without approval?"
    for a in (
        "You can work remotely up to 4 days per week without approval.",
        "You can work remotely up to four days per week without approval.",
        "You can work remotely up to 2 days per week without approval.",
        "You can work remotely up to one day per week without approval.",
    ):
        r = validate(q, a, doc)
        assert r["verdict"] != "Supported", (a, r)


def test_nonurgent_sla_question_does_not_borrow_urgent_window() -> None:
    doc = _doc()
    q = "What is the first-response SLA for non-urgent tickets?"
    for a in (
        "First response within 4 business hours.",
        "First response within 3 business days.",
    ):
        r = validate(q, a, doc)
        assert r["verdict"] != "Supported", (a, r)


def test_question_scoped_nonurgent_sla_allows_full_correct_answer() -> None:
    doc = _doc()
    q = "What is the first-response SLA for non-urgent tickets?"
    a = "Non-urgent tickets receive a first response within 2 business days; urgent tickets are within 4 business hours."
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


def test_benefit_scoped_eligibility_omission_is_partial() -> None:
    doc = _doc()
    for q in (
        "What about the remote stipend?",
        "Which employees get the stipend?",
    ):
        r = validate(q, "Full-time employees are eligible.", doc)
        assert r["verdict"] == "Partial", (q, r)


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_wrong_remote_without_approval_caps_are_not_supported()
    test_nonurgent_sla_question_does_not_borrow_urgent_window()
    test_question_scoped_nonurgent_sla_allows_full_correct_answer()
    test_benefit_scoped_eligibility_omission_is_partial()
    print("ok: safety-gate regressions are not Supported")
