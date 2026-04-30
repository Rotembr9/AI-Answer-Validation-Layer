"""
Evaluate the MVP validator on the labeled dataset in data/examples.json.

Run from repo root:
    python tests/evaluate_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validator import load_examples_json, validate  # noqa: E402

LABELS = ("Supported", "Not Supported", "Partial")


def confusion_matrix(
    expected: list[str], predicted: list[str]
) -> dict[tuple[str, str], int]:
    m: dict[tuple[str, str], int] = {}
    for e, p in zip(expected, predicted):
        key = (e, p)
        m[key] = m.get(key, 0) + 1
    return m


def print_matrix(expected: list[str], predicted: list[str]) -> None:
    cells = confusion_matrix(expected, predicted)
    print("Confusion matrix (rows = expected, cols = predicted):")
    header = " " * 22 + "".join(f"{lab[:12]:>14}" for lab in LABELS)
    print(header)
    for row in LABELS:
        line = f"{row[:20]:20}"
        for col in LABELS:
            line += f"{cells.get((row, col), 0):>14}"
        print(line)


def main() -> None:
    doc_path = ROOT / "data" / "examples.json"
    source_document, examples = load_examples_json(doc_path)

    expected: list[str] = []
    predicted: list[str] = []
    failed: list[str] = []

    supported_tp = supported_fp = 0
    ns_actual = ns_tp = 0

    for ex in examples:
        exp = ex["expected_label"]
        r = validate(ex["question"], ex["answer"], source_document)
        pred = r["verdict"]
        expected.append(exp)
        predicted.append(pred)
        if pred != exp:
            failed.append(f"  {ex['id']}: expected {exp}, got {pred} (conf={r['confidence']})")

        if exp == "Supported":
            if pred == "Supported":
                supported_tp += 1
            elif pred != "Supported":
                supported_fp += 1
        if exp == "Not Supported":
            ns_actual += 1
            if pred == "Not Supported":
                ns_tp += 1

    n = len(examples)
    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    acc = correct / n if n else 0.0

    supported_precision = supported_tp / (supported_tp + supported_fp) if (supported_tp + supported_fp) else 0.0
    ns_recall = ns_tp / ns_actual if ns_actual else 0.0

    unsafe_supported = sum(
        1 for e, p in zip(expected, predicted) if e != "Supported" and p == "Supported"
    )
    ns_mistaken_as_supported = sum(
        1 for e, p in zip(expected, predicted) if e == "Not Supported" and p == "Supported"
    )

    print("Total examples:", n)
    print("Correct:", correct)
    print(f"Accuracy: {acc:.4f}")
    print(f"Supported precision: {supported_precision:.4f}")
    print(f"Not Supported recall: {ns_recall:.4f}")
    print(
        f"Non-Supported truth -> Supported (strict, should be 0): {ns_mistaken_as_supported}"
    )
    print(
        f"Any non-Supported label -> Supported (incl. Partial): {unsafe_supported}"
    )
    print()
    print_matrix(expected, predicted)
    print()
    print("Failed examples:")
    if failed:
        print("\n".join(failed))
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
