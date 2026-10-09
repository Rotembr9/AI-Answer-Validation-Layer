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


def test_wrong_exception_does_not_cover_eligibility_exclusion() -> None:
    q = "Who is eligible for the remote work stipend?"
    doc = "Full-time staff only; contractors are not eligible for the remote stipend."
    for answer in (
        "Full-time staff are eligible except interns.",
        "Only full-time staff are eligible except interns.",
    ):
        r = validate(q, answer, doc)
        assert r["verdict"] == "Partial", r


def test_contractor_exclusion_paraphrase_is_supported() -> None:
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time staff are eligible; contractors are excluded."
    doc = "Full-time staff only; contractors are not eligible for the remote stipend."
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


def test_contractor_not_being_eligible_is_not_a_contradiction() -> None:
    q = "Who gets remote stipend money?"
    a = (
        "Full-time employees can receive it; the policy also references contractors "
        "not being eligible but does not describe hybrid roles."
    )
    doc = "Full-time staff only; contractors are not eligible for the remote stipend."
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_unrelated_exclusion_line_does_not_penalize_remote_answer() -> None:
    q = "How many remote days are allowed each week without approval?"
    a = "Employees are allowed to work remotely up to 3 days per week without extra approval."
    doc = "\n".join(
        [
            "Full-time staff only; contractors are not eligible for the remote stipend.",
            "Employees may work remotely up to 3 days per week without extra approval.",
        ]
    )
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_wrong_exception_does_not_cover_eligibility_exclusion()
    test_contractor_exclusion_paraphrase_is_supported()
    test_contractor_not_being_eligible_is_not_a_contradiction()
    test_unrelated_exclusion_line_does_not_penalize_remote_answer()
    print("ok: safety gates passed")
