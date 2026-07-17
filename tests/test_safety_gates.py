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


def test_remote_one_day_cap_is_not_supported() -> None:
    q = "How many remote days are allowed each week without approval?"
    a = "You may work remotely one day per week without approval."
    r = validate(q, a, _doc())
    assert r["verdict"] == "Not Supported", r


def test_contractor_exclusion_paraphrase_is_not_rejected() -> None:
    q, a = _load_h("H-P07")
    r = validate(q, a, _doc())
    assert r["verdict"] == "Partial", r


def test_unrelated_expense_allowed_question_has_no_exclusivity_penalty() -> None:
    q = "Are employees allowed to submit expense reports for purchases?"
    a = "Employees must submit expense reports by the 15th day of the month following the purchase."
    r = validate_with_debug(q, a, _doc())
    assert r["_debug"]["exclusivity_omission_penalty"] == 0.0, r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_remote_one_day_cap_is_not_supported()
    test_contractor_exclusion_paraphrase_is_not_rejected()
    test_unrelated_expense_allowed_question_has_no_exclusivity_penalty()
    print("ok: safety-gate regressions passed")
