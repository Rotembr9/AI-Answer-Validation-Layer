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


def test_wrong_exception_does_not_cover_contractor_exclusion() -> None:
    """An unrelated exception must not satisfy a source contractor exclusion."""
    q = "Who is eligible for the remote work stipend?"
    a = "Only full-time staff are eligible except interns."
    doc = "Remote stipend: Full-time staff only; contractors are not eligible."
    r = validate(q, a, doc)
    assert r["verdict"] != "Supported", r


def test_contractor_exclusion_paraphrases_are_supported() -> None:
    q = "Who is eligible for the remote work stipend?"
    doc = "Remote stipend: Full-time staff only; contractors are not eligible."
    answers = [
        "Only full-time staff are eligible; contractors are excluded.",
        "Full-time staff are eligible; contractors cannot receive the stipend.",
    ]
    for a in answers:
        r = validate(q, a, doc)
        assert r["verdict"] == "Supported", (a, r)


def test_exclusivity_penalty_uses_matched_evidence_line() -> None:
    q = "Who can work remotely without approval?"
    a = "Employees may work remotely up to 3 days per week without extra approval."
    doc = (
        "1. Eligible employees: Full-time staff only; contractors are not eligible for the remote stipend.\n"
        "2. Remote work: Employees may work remotely up to 3 days per week without extra approval."
    )
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_wrong_exception_does_not_cover_contractor_exclusion()
    test_contractor_exclusion_paraphrases_are_supported()
    test_exclusivity_penalty_uses_matched_evidence_line()
    print("ok: H-N08 and H-P10 are not Supported; exclusivity example is Partial")
