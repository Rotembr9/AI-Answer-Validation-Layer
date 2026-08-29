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


def test_plural_eligibility_requirements_omission_not_supported() -> None:
    """Plural/prefix eligibility wording must still trigger the exclusivity omission gate."""
    q = "What are the eligibility requirements for the remote work stipend?"
    a = "Full-time employees are eligible for the remote work stipend."
    doc = (
        "Remote work stipend eligibility requirements: Full-time employees are eligible "
        "for the remote work stipend only; contractors are not eligible."
    )
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_understated_remote_day_cap_not_supported() -> None:
    """A lower remote-day cap is not supported just because its number appears elsewhere."""
    q = "How many remote days are allowed each week without approval?"
    a = "Employees may work remotely one day per week without approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_correct_remote_day_cap_with_employee_word_supported() -> None:
    """Remote cap questions must not inherit unrelated stipend eligibility exclusions."""
    q = "How many remote days are allowed each week without approval?"
    a = "Employees may work remotely up to 3 days per week without extra approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_plural_eligibility_requirements_omission_not_supported()
    test_understated_remote_day_cap_not_supported()
    test_correct_remote_day_cap_with_employee_word_supported()
    print("ok: H-N08, H-P10, exclusivity, and remote-cap safety checks passed")
