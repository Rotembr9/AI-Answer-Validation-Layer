"""Local text helpers: tokenization, TF–IDF overlap similarity, number extraction."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

# Words shorter than this are dropped from overlap / TF–IDF (noise reduction).
_MIN_TOKEN_LEN = 2

_TOKEN_RE = re.compile(r"[a-z0-9$%]+", re.IGNORECASE)


def normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def tokenize(s: str) -> list[str]:
    raw = _TOKEN_RE.findall(s.lower())
    out: list[str] = []
    for t in raw:
        t = t.strip("$%").strip()
        if len(t) >= _MIN_TOKEN_LEN or t.isdigit():
            out.append(t)
    return out


def split_document_lines(document: str) -> list[str]:
    """Non-empty lines and sentence-like chunks for evidence retrieval."""
    lines: list[str] = []
    for block in document.replace("\r\n", "\n").split("\n"):
        b = block.strip()
        if b:
            lines.append(b)
    # Also split long single blocks on sentence boundaries
    if len(lines) == 1 and len(lines[0]) > 200:
        parts = re.split(r"(?<=[.!?])\s+", lines[0])
        lines = [p.strip() for p in parts if p.strip()]
    return lines if lines else [document.strip()]


# --- Numbers / amounts (for consistency checks) ---

_INT_WORD_BOUNDARY = re.compile(r"\b\d+\b")
_MONEY = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def extract_numeric_tokens(text: str) -> set[int]:
    """
    Extract standalone integers and dollar amounts (as whole dollars).
    Used to check whether numeric claims in an answer appear in the source.
    Normalizes ordinals (15th -> 15) for deadline-style phrases.
    """
    text = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    found: set[int] = set()
    for m in _INT_WORD_BOUNDARY.finditer(text):
        try:
            found.add(int(m.group(0)))
        except ValueError:
            pass
    for m in _MONEY.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            found.add(int(float(raw)))
        except ValueError:
            pass
    return found


# --- TF–IDF cosine (global idf over document lines + query for stability) ---


def _tf_idf_matrix(docs: list[list[str]]) -> tuple[list[Counter[str]], dict[str, float]]:
    """docs[0] is typically the query (answer); others are corpus lines."""
    dfs: Counter[str] = Counter()
    tfs: list[Counter[str]] = []
    for d in docs:
        tf = Counter(d)
        tfs.append(tf)
        dfs.update(set(tf.keys()))
    n = len(docs)
    idf: dict[str, float] = {}
    for term, df in dfs.items():
        idf[term] = math.log((1.0 + n) / (1.0 + df)) + 1.0
    return tfs, idf


def _vec_from_tf(tf: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    return {t: tf[t] * idf.get(t, 0.0) for t in tf}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0.0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def max_tfidf_cosine_to_lines(query: str, lines: Iterable[str]) -> tuple[float, list[tuple[float, str]]]:
    """
    Return (best_cosine, sorted list of (score, line)) comparing query to each line.
    Corpus for IDF = query tokens union all line tokens.
    """
    line_list = list(lines)
    if not line_list:
        return 0.0, []

    q_tokens = tokenize(query)
    doc_tokens = [tokenize(L) for L in line_list]
    docs = [q_tokens] + doc_tokens
    tfs, idf = _tf_idf_matrix(docs)
    q_vec = _vec_from_tf(tfs[0], idf)

    scored: list[tuple[float, str]] = []
    for i, line in enumerate(line_list):
        lv = _vec_from_tf(tfs[i + 1], idf)
        c = _cosine(q_vec, lv)
        scored.append((c, line))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0] if scored else 0.0
    return best, scored


def jaccard_keywords(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    sa, sb = set(a_tokens), set(b_tokens)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# Simple negation / polarity cues for contradiction heuristics
_NEGATIVE_HINTS = frozenset(
    [
        "not",
        "no",
        "never",
        "denied",
        "ineligible",
        "prohibited",
        "cannot",
        "can't",
        "must not",
        "without approval",
    ]
)
_POSITIVE_HINTS = frozenset(
    [
        "may",
        "allowed",
        "eligible",
        "permitted",
        "can",
        "yes",
        "up to",
        "requires",
        "must be",
    ]
)


def polarity_hint(tokens: list[str]) -> int:
    """Rough signal: -1 negative, 0 neutral, +1 positive."""
    joined = " ".join(tokens)
    n = sum(1 for h in _NEGATIVE_HINTS if h in joined)
    p = sum(1 for h in _POSITIVE_HINTS if h in joined)
    if n > p:
        return -1
    if p > n:
        return 1
    return 0
