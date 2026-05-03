"""Main validation entrypoint: explainable local signals only (no external APIs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import scoring as sc
from . import text_utils as tu


def _top_evidence(scored_lines: list[tuple[float, str]], k: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for score, line in scored_lines:
        if score <= 0:
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
        if len(out) >= k:
            break
    return out


def validate(question: str, answer: str, source_document: str) -> dict[str, Any]:
    """
    Validate ``answer`` against ``source_document`` for ``question``.

    Returns a JSON-serializable dict with verdict, confidence, reason, evidence.
    """
    r = validate_with_debug(question, answer, source_document)
    r.pop("_debug", None)
    return r


def validate_with_debug(question: str, answer: str, source_document: str) -> dict[str, Any]:
    """Like validate but adds subscores for tests and tuning."""
    lines = tu.split_document_lines(source_document)
    q_for_sim = f"{question}\n{answer}"
    best_cos, scored = tu.max_tfidf_cosine_to_lines(q_for_sim, lines)
    evidence = _top_evidence(scored, k=3)
    top_line = scored[0][1] if scored else ""
    kw = sc.keyword_match_score(question, answer, source_document, top_line)
    num_score, unknown_nums = sc.number_match_score(answer, source_document)
    contra = sc.contradiction_signals(answer, top_line, source_document, question)
    forbid_sup, safety_contra = sc.supported_safety_flags(answer, source_document)
    contra = max(contra, safety_contra)
    excl_pen = sc.incomplete_exclusivity_penalty(question, answer, source_document)
    contra = max(contra, excl_pen)

    def _numeric_conflict() -> bool:
        if not top_line:
            return False
        an = tu.extract_numeric_tokens(answer)
        en = tu.extract_numeric_tokens(top_line)
        if not an or not en:
            return False
        overlap = set(tu.tokenize(answer)) & set(tu.tokenize(top_line))
        if len(overlap) < 2:
            return False
        if an != en and an.isdisjoint(en):
            if max(an) >= 2 and max(en) >= 2:
                return True
        return False

    forced_ns = _numeric_conflict() and best_cos > 0.2
    confidence = sc.combine_scores(best_cos, kw, num_score, contra, unknown_nums)
    # Weak signal fix (S→P): full-document keyword overlap is often low when the answer
    # matches one evidence line; once numbers + line cosine align, nudge confidence slightly.
    if (
        not forbid_sup
        and not unknown_nums
        and num_score >= 0.999
        and best_cos >= 0.42
        and contra <= sc.MAX_CONTRA_FOR_SUPPORTED
    ):
        confidence = min(1.0, confidence + 0.022)
    if unknown_nums:
        confidence = min(confidence, 0.55)
    verdict, reason = sc.verdict_from_signals(
        confidence,
        best_cos,
        kw,
        contra,
        unknown_nums,
        forced_ns,
        forbid_supported=forbid_sup,
    )
    if verdict == "Supported":
        reason = (
            "Matched policy language and numeric limits in the cited evidence "
            f"(evidence similarity {best_cos:.2f}, keyword overlap {kw:.2f})."
        )
    elif verdict == "Partial":
        reason = (
            "Some statements align with the source, but not all details or conditions "
            f"are clearly satisfied (similarity {best_cos:.2f}, numbers score {num_score:.2f})."
        )
    elif unknown_nums and verdict == "Not Supported":
        reason = "One or more numbers in the answer do not appear in the source document."
    r = {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "reason": reason,
        "evidence": evidence,
        "_debug": {
            "evidence_similarity_score": round(best_cos, 4),
            "keyword_match_score": round(kw, 4),
            "number_match_score": round(num_score, 4),
            "contradiction_penalty": round(contra, 4),
            "unknown_numbers_in_answer": unknown_nums,
            "forbid_supported": forbid_sup,
            "safety_contra_boost": round(safety_contra, 4),
            "exclusivity_omission_penalty": round(excl_pen, 4),
        },
    }
    return r


def load_examples_json(path: Path | str) -> tuple[str, list[dict[str, Any]]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data["source_document"], list(data["examples"])
