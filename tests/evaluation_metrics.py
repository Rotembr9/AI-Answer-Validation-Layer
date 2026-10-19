"""Shared metric helpers for evaluation scripts."""

from __future__ import annotations


def supported_precision(expected: list[str], predicted: list[str]) -> float:
    """Precision for predicted Supported verdicts: TP / all predicted Supported."""
    supported_tp = sum(
        1 for exp, pred in zip(expected, predicted) if exp == "Supported" and pred == "Supported"
    )
    supported_fp = sum(
        1 for exp, pred in zip(expected, predicted) if exp != "Supported" and pred == "Supported"
    )
    denom = supported_tp + supported_fp
    return supported_tp / denom if denom else 0.0
