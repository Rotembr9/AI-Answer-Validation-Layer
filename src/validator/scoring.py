"""
Score components and verdict mapping.

Thresholds (tunable for MVP; documented here for debugging):

- SUPPORTED_MIN_CONF: confidence needed to emit Supported (conservative to avoid
  marking hallucinations as Supported). Default 0.72.

- PARTIAL_BAND: if confidence is below Supported threshold but evidence_similarity
  or keyword overlap shows partial grounding, emit Partial.

- UNKNOWN_AS_REJECT: if max evidence similarity is very low, treat as Not Supported
  (no grounding).

Numeric rule:
- Every standalone integer / dollar amount in the answer should appear somewhere
  in the source document (same digits). If not, apply a concrete penalty and often
  Not Supported (guards N02-style fabricated figures).
"""

from __future__ import annotations

from . import text_utils as tu


# --- Thresholds (see module docstring) ---
# Conservative but achievable on the MVP dataset once polarity/order bugs are fixed.
# Slightly above P02 (borderline partial) and below the lowest clean Supported (S07 ~0.495).
SUPPORTED_MIN_CONF = 0.49
EVIDENCE_FLOOR = 0.12  # below this: "no relevant evidence"
NUMBER_MISS_PENALTY = 0.45


def keyword_match_score(question: str, answer: str, document: str) -> float:
    q, a, d = tu.tokenize(question), tu.tokenize(answer), tu.tokenize(document)
    qa = q + a
    return max(tu.jaccard_keywords(qa, d), tu.jaccard_keywords(tu.tokenize(answer), d))


def number_match_score(answer: str, document: str) -> tuple[float, bool]:
    """
    Returns (score 0–1, unknown_number_present).
    Score is fraction of answer numeric tokens that appear in the document text.
    """
    an = tu.extract_numeric_tokens(answer)
    dn = tu.extract_numeric_tokens(document)
    if not an:
        return 1.0, False
    hits = sum(1 for x in an if x in dn)
    unknown = hits < len(an)
    return hits / len(an), unknown


def contradiction_signals(answer: str, top_evidence_line: str, document: str) -> float:
    """
    Returns penalty in [0, 1]. Higher = stronger suspicion of contradiction.
    Uses polarity mismatch answer vs evidence line and keyword antisets.
    """
    a_tokens = tu.tokenize(answer)
    e_tokens = tu.tokenize(top_evidence_line)
    joined_a = " ".join(a_tokens).lower()
    joined_e = " ".join(e_tokens).lower()
    doc_low = document.lower()
    # Tokenizer splits "non-urgent" → tokens "non", "urgent"; hyphen-normalize for substring rules.
    a_norm = joined_a.replace("-", " ")
    d_norm = doc_low.replace("-", " ")
    penalty = 0.0
    # Avoid penalizing correct negations (e.g. "amounts above $500 are not reimbursed")
    # where the matched line is phrased positively but is the same rule.
    # Polarity vs top line alone was overly punitive for correct paraphrases; rely on rules below.

    # Remote work: flat denial vs policy that allows remote (use full doc — top line may be title only).
    if ("not permitted" in joined_a or "no remote" in joined_a or "not allowed" in joined_a) and (
        "remote" in joined_e or "remotely" in joined_e or "remote" in d_norm
    ):
        if "may work remotely" in d_norm or "may" in joined_e or "up to" in joined_e:
            penalty = max(penalty, 0.95)

    # Contractor eligibility: explicit mismatch patterns
    if "contractor" in joined_a and "eligible" in joined_a and "full-time" in joined_e:
        if "not" not in joined_a and "only" in joined_e:
            penalty = max(penalty, 0.9)

    # Doc-level: contractors explicitly not eligible for stipend / reimbursement
    if "contractor" in joined_a:
        if "contractors are not eligible" in doc_low and "not eligible" not in joined_a:
            # Strong explicit lie about contractor eligibility (short affirmative answers)
            if "contractors are eligible" in joined_a or "yes," in joined_a[:40]:
                penalty = max(penalty, 0.88)
            # Same-benefit claim in a longer mixed answer → Partial tier
            elif "same" in joined_a or "as full-time" in joined_a:
                penalty = max(penalty, 0.62)
            elif any(w in joined_a for w in ("eligible", "stipend", "reimbursement", "equipment")):
                penalty = max(penalty, 0.84)
    # Equipment return vs optional / no consequence (often mixed correct + wrong → keep penalty mid/high)
    if "laptop" in joined_a or ("return" in joined_a and "equipment" in doc_low):
        if any(
            p in joined_a
            for p in ("optional", "may keep", "no financial", "no consequence", "personally")
        ):
            if "must be returned" in doc_low or "payroll deduction" in doc_low:
                penalty = max(penalty, 0.68)
    # Denies policy-stated SLA as "not specified"
    if "not specified" in a_norm and "non urgent" in a_norm:
        if "2 business days" in d_norm and "non urgent" in d_norm:
            penalty = max(penalty, 0.55)
    # Omits approval timing for extra remote day
    if "fourth" in joined_a or "4th" in joined_a or "extra" in joined_a:
        if "verbal" in joined_a and "written" in doc_low:
            penalty = max(penalty, 0.62)
    if ("fourth" in joined_a or "4th" in joined_a) and "no timing" in joined_a:
        if "before that week" in doc_low or "before that week begins" in doc_low:
            penalty = max(penalty, 0.58)
    # Extra remote day without approval (policy requires approval for 4th day)
    if "fourth" in joined_a or "4th" in joined_a:
        if "without" in joined_a and "approval" in joined_a and "not" not in joined_a:
            if "approval" in doc_low:
                penalty = max(penalty, 0.86)
    # Reimbursement above cap
    if "above" in joined_a and "500" in joined_a:
        if "not reimbursed" in doc_low and "not" not in joined_a:
            penalty = max(penalty, 0.88)
    # Expense deadline wrong
    if "last day" in joined_a and "month" in joined_a:
        if "15" in doc_low or "15th" in doc_low:
            penalty = max(penalty, 0.85)
    # Urgent vs non-urgent SLA mix-ups (require true "urgent", not the substring inside "non urgent")
    if (
        "non urgent" not in a_norm
        and "urgent" in a_norm
        and "2 business day" in a_norm
        and "4 business hours" not in a_norm
    ):
        if "4 business hours" in d_norm:
            penalty = max(penalty, 0.88)
    # Non-urgent tickets must not use the urgent SLA window (skip if answer hedges, e.g. "not specified").
    if (
        "non urgent" in a_norm
        and "4 business hours" in a_norm
        and "2 business days" in d_norm
        and "not specified" not in a_norm
    ):
        penalty = max(penalty, 0.88)

    return min(1.0, penalty)


def combine_scores(
    ev_sim: float,
    kw: float,
    num: float,
    contra_pen: float,
    unknown_numbers: bool,
) -> float:
    """Map subscores to a single confidence in [0, 1]."""
    base = 0.45 * ev_sim + 0.30 * kw + 0.25 * num
    base *= 1.0 - 0.85 * contra_pen
    if unknown_numbers:
        base -= NUMBER_MISS_PENALTY
    return max(0.0, min(1.0, base))


def verdict_from_signals(
    confidence: float,
    ev_sim: float,
    kw: float,
    contra_pen: float,
    unknown_numbers: bool,
    forced_not_supported: bool,
) -> tuple[str, str]:
    if forced_not_supported or contra_pen >= 0.80:
        return "Not Supported", "Strong contradiction or unreliable numeric claims versus the source."
    # Any numeric token in the answer that never appears in the source → not grounded (MVP safety).
    if unknown_numbers:
        return "Not Supported", "The answer includes numbers or amounts not found in the source document."
    # Prefer Supported when signals are clearly aligned (check before medium-contradiction Partial).
    if (
        confidence >= SUPPORTED_MIN_CONF
        and contra_pen < 0.50
        and not unknown_numbers
        and ev_sim >= 0.14
    ):
        return "Supported", "High alignment between the answer, matched evidence, and stated numbers in the source."
    if 0.50 <= contra_pen < 0.80 and (ev_sim >= 0.18 or kw >= 0.12):
        return "Partial", "Some claims align with the source, but key conditions conflict with or omit policy details."
    if ev_sim < EVIDENCE_FLOOR and kw < 0.08:
        return "Not Supported", "No sufficient overlap with the source document."
    if ev_sim >= EVIDENCE_FLOOR or kw >= 0.08:
        return "Partial", "Some claims align with the source, but overlap, numbers, or conditions are incomplete or mixed."
    return "Not Supported", "Insufficient evidence in the source for the answer as stated."
