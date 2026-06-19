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


def test_wrong_unit_number_from_section_header_is_not_supported() -> None:
    q = "How fast should urgent support tickets get a first response?"
    a = "Urgent Severity 1 tickets require a first response within 5 business hours."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_wrong_laptop_return_days_from_section_header_is_not_supported() -> None:
    q = "When do I return my laptop after leaving the company?"
    a = "Laptops must be returned within 6 business days after employment ends."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_too_many_remote_days_without_approval_is_not_supported() -> None:
    q = "How many remote days are allowed each week without approval?"
    a = "Employees may work remotely up to 4 days per week without extra approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_wrong_unit_number_from_section_header_is_not_supported()
    test_wrong_laptop_return_days_from_section_header_is_not_supported()
    test_too_many_remote_days_without_approval_is_not_supported()
    print("ok: safety gates block false Supported numeric and exclusivity cases")
