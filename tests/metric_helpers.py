"""Shared metrics for evaluation scripts."""

from __future__ import annotations


def supported_prediction_precision(expected: list[str], predicted: list[str]) -> float:
    """Precision for the risky Supported prediction class."""
    true_supported_predictions = sum(
        1
        for exp, pred in zip(expected, predicted)
        if exp == "Supported" and pred == "Supported"
    )
    supported_predictions = sum(1 for pred in predicted if pred == "Supported")
    if not supported_predictions:
        return 0.0
    return true_supported_predictions / supported_predictions
