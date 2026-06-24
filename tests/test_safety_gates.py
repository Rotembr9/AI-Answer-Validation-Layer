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


def test_exception_words_do_not_hide_eligibility_omission() -> None:
    """A generic "exceptions" phrase is not coverage of an explicit exclusion."""
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time employees are eligible. No exceptions noted."
    doc = (
        "Remote work stipend eligibility: Full-time employees are eligible. "
        "Contractors are not eligible."
    )
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_part_time_exclusion_is_not_a_contradiction() -> None:
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time employees are eligible; part-time employees are not eligible."
    doc = "Full-time staff only; part-time employees are not eligible"
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


def test_urgent_day_scale_without_severity_label_never_supported() -> None:
    q = "What is the response window for urgent tickets?"
    a = "Urgent tickets require a first response within one business day."
    doc = (
        "Urgent tickets require a first response within 4 business hours. "
        "Effective January 1, 2025."
    )
    r = validate(q, a, doc)
    assert r["verdict"] != "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_exception_words_do_not_hide_eligibility_omission()
    test_part_time_exclusion_is_not_a_contradiction()
    test_urgent_day_scale_without_severity_label_never_supported()
    print("ok: H-N08 and H-P10 are not Supported; exclusivity example is Partial")
