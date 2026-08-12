"""Shared metric helpers for validator evaluation scripts."""

from __future__ import annotations


def predicted_supported_precision(expected: list[str], predicted: list[str]) -> float:
    """Precision for the Supported label: correct predicted Supported / all predicted Supported."""
    predicted_supported = sum(1 for p in predicted if p == "Supported")
    if not predicted_supported:
        return 0.0
    true_supported = sum(
        1 for exp, pred in zip(expected, predicted)
        if exp == "Supported" and pred == "Supported"
    )
    return true_supported / predicted_supported
