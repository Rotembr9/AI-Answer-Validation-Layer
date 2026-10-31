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


def test_unmatched_exception_does_not_cover_source_exclusion() -> None:
    """An exception for an unrelated group must not hide the contractor exclusion."""
    q = "Who is eligible for the remote stipend?"
    doc = _doc()
    answers = [
        "Full-time staff are eligible for the remote stipend except interns.",
        "Full-time staff are eligible for the remote stipend; it does not apply to interns.",
        "Only full-time staff are eligible for the remote stipend except interns.",
    ]
    for answer in answers:
        r = validate(q, answer, doc)
        assert r["verdict"] != "Supported", r


def test_contractor_exclusion_paraphrases_remain_supported() -> None:
    q = "Who is eligible for the remote stipend?"
    doc = _doc()
    answers = [
        "Full-time staff are eligible for the remote stipend except contractors.",
        "Full-time staff are eligible for the remote stipend; contractors are excluded.",
        "Full-time staff may receive the remote stipend; it does not apply to contractors.",
    ]
    for answer in answers:
        r = validate(q, answer, doc)
        assert r["verdict"] == "Supported", r


def test_unrelated_exception_does_not_hide_contractor_contradiction() -> None:
    q = "Are contractors eligible for the remote stipend?"
    a = "Yes, contractors are eligible for the remote stipend except interns."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_unmatched_exception_does_not_cover_source_exclusion()
    test_contractor_exclusion_paraphrases_remain_supported()
    test_unrelated_exception_does_not_hide_contractor_contradiction()
    print("ok: safety gates passed")
