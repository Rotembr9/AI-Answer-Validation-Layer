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


def test_generic_exception_does_not_cover_contractor_exclusion() -> None:
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time staff are eligible except interns."
    doc = "Full-time staff only; contractors are not eligible."
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_split_exclusion_line_still_blocks_positive_only_eligibility() -> None:
    q = "Who is eligible for the remote work stipend?"
    a = "Full-time staff are eligible."
    doc = "Full-time staff receive the remote stipend.\nContractors are not eligible."
    r = validate(q, a, doc)
    assert r["verdict"] == "Partial", r


def test_contractor_exclusion_paraphrase_is_supported() -> None:
    q = "Who is eligible for the remote work stipend?"
    doc = "Full-time staff only; contractors are not eligible."
    supported_answers = [
        "Only full-time staff are eligible; contractors are excluded.",
        "Full-time staff are eligible except contractors.",
    ]
    for a in supported_answers:
        r = validate(q, a, doc)
        assert r["verdict"] == "Supported", r

    r = validate(
        q,
        "Full-time staff are eligible, but contractors can't receive the stipend.",
        doc,
    )
    assert r["verdict"] != "Not Supported", r


def test_unrelated_exclusion_does_not_downgrade_remote_work_answer() -> None:
    q = "Who can work remotely without additional approval?"
    a = "Employees may work remotely up to 3 days per week without extra approval."
    doc = (
        "Remote work: employees may work remotely up to 3 days per week without extra approval.\n"
        "Stipend: full-time staff only; contractors are not eligible."
    )
    r = validate(q, a, doc)
    assert r["verdict"] == "Supported", r


if __name__ == "__main__":
    test_h_n08_never_supported()
    test_h_p10_never_supported()
    test_incomplete_eligibility_is_partial_not_supported()
    test_generic_exception_does_not_cover_contractor_exclusion()
    test_split_exclusion_line_still_blocks_positive_only_eligibility()
    test_contractor_exclusion_paraphrase_is_supported()
    test_unrelated_exclusion_does_not_downgrade_remote_work_answer()
    print("ok: safety gates passed")
