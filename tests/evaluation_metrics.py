"""Shared metrics for validator evaluation scripts."""

from __future__ import annotations

LABEL_SUPPORTED = "Supported"
LABEL_NOT_SUPPORTED = "Not Supported"


def supported_precision(expected: list[str], predicted: list[str]) -> float:
    """
    Precision for Supported verdicts: of rows predicted Supported, how many were
    actually Supported.
    """
    predicted_supported = sum(1 for p in predicted if p == LABEL_SUPPORTED)
    if predicted_supported == 0:
        return 0.0
    true_supported = sum(
        1
        for exp, pred in zip(expected, predicted)
        if exp == LABEL_SUPPORTED and pred == LABEL_SUPPORTED
    )
    return true_supported / predicted_supported


def not_supported_recall(expected: list[str], predicted: list[str]) -> float:
    """Recall for gold Not Supported rows."""
    actual_not_supported = sum(1 for exp in expected if exp == LABEL_NOT_SUPPORTED)
    if actual_not_supported == 0:
        return 0.0
    correctly_rejected = sum(
        1
        for exp, pred in zip(expected, predicted)
        if exp == LABEL_NOT_SUPPORTED and pred == LABEL_NOT_SUPPORTED
    )
    return correctly_rejected / actual_not_supported
