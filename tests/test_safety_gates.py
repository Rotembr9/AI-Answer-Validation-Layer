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


def test_section_number_does_not_ground_wrong_laptop_deadline() -> None:
    """The equipment section number ``6.`` is not a six-business-day return deadline."""
    q = "When do I return my laptop after leaving the company?"
    a = "Company laptops must be returned within 6 business days after employment ends."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_same_line_number_does_not_allow_wrong_remote_limit() -> None:
    """The 4th-day approval rule must not support four no-approval remote days."""
    q = "How many remote days are allowed each week without approval?"
    a = "You can work remotely up to 4 days per week without extra approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_section_number_does_not_ground_wrong_laptop_deadline()
    test_same_line_number_does_not_allow_wrong_remote_limit()
    print("ok: safety gate regression tests passed")
