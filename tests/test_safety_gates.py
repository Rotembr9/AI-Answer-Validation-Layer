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

from validator import load_examples_json, validate, validate_with_debug  # noqa: E402


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


def test_exception_must_match_source_exclusion() -> None:
    q = "Who is eligible for the remote work stipend?"
    doc = "Full-time staff only; contractors are not eligible"

    generic_exception = validate(q, "Full-time staff are eligible except interns.", doc)
    assert generic_exception["verdict"] != "Supported", generic_exception

    contractor_exception = validate(q, "Full-time staff are eligible except contractors.", doc)
    assert contractor_exception["verdict"] == "Supported", contractor_exception


def test_exclusivity_penalty_uses_matched_evidence_line() -> None:
    q = "Who can work remotely without extra approval?"
    a = "Employees may work remotely up to 3 days per week without extra approval."
    r = validate_with_debug(q, a, _doc())
    assert r["verdict"] == "Supported", r
    assert r["_debug"]["exclusivity_omission_penalty"] == 0.0, r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_exception_must_match_source_exclusion()
    test_exclusivity_penalty_uses_matched_evidence_line()
    print("ok: safety gates passed")
