#!/usr/bin/env python3
"""
Interactive or non-interactive CLI demo.

Uses the bundled policy from data/examples.json automatically (no document prompt).

Interactive (terminal only):
    python run_demo.py

Non-interactive (recommended for scripts; avoids stdin hangs):
    python run_demo.py --question "..." --answer "..."

On Windows PowerShell, put the answer in single quotes if it contains $:
    python run_demo.py --question "What is the annual cap?" --answer 'Up to $500 per year for approved expenses.'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from validator import validate  # noqa: E402


def load_bundled_document() -> str:
    path = ROOT / "data" / "examples.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["source_document"]


def print_result(result: dict, width: int = 64) -> None:
    print()
    print("-" * width)
    print(f"  VERDICT:      {result['verdict']}")
    print(f"  CONFIDENCE:   {result['confidence']:.3f}")
    print("-" * width)
    print()
    print("  EXPLANATION:")
    for line in result["reason"].splitlines():
        print(f"    {line}")
    print()
    print("  EVIDENCE:")
    for i, block in enumerate(result["evidence"], 1):
        text = " ".join(block.split())
        print(f"    [{i}] {text}")
    print()
    print("=" * width)
    print()


def main() -> None:
    width = 64
    parser = argparse.ArgumentParser(
        description="Validate an AI answer against the bundled policy in data/examples.json.",
    )
    parser.add_argument(
        "--question",
        "-q",
        metavar="TEXT",
        help="Question (non-interactive mode; requires --answer)",
    )
    parser.add_argument(
        "--answer",
        "-a",
        metavar="TEXT",
        help="AI-generated answer (non-interactive mode; requires --question)",
    )
    args = parser.parse_args()

    doc = load_bundled_document()

    has_q = args.question is not None
    has_a = args.answer is not None
    if has_q ^ has_a:
        parser.error("non-interactive mode requires both --question and --answer")

    if has_q and has_a:
        question = args.question.strip()
        answer = args.answer.strip()
        if not question or not answer:
            print("Error: --question and --answer must be non-empty.", file=sys.stderr)
            sys.exit(2)
    elif not sys.stdin.isatty():
        print(
            "Error: stdin is not a terminal. Use non-interactive mode, for example:",
            file=sys.stderr,
        )
        print(
            '  python run_demo.py --question "..." --answer "..."',
            file=sys.stderr,
        )
        sys.exit(2)
    else:
        print()
        print("=" * width)
        print(" AI Answer Validation Layer - interactive demo".ljust(width))
        print("=" * width)
        print()
        print("Using bundled document: data/examples.json (remote work & expense policy).")
        print("No source text is requested; only question and answer below.")
        print()
        try:
            question = input("Question (required): ").strip()
            answer = input("Answer (required):   ").strip()
        except EOFError:
            print("\nNo input received. For non-interactive use:", file=sys.stderr)
            print(
                '  python run_demo.py --question "..." --answer "..."',
                file=sys.stderr,
            )
            sys.exit(2)

        if not question or not answer:
            print("\nError: both question and answer are required.")
            sys.exit(1)

        print()

    result = validate(question, answer, doc)

    if has_q and has_a:
        print("Document: bundled policy from data/examples.json")
        print()
    else:
        print("-" * width)
    print_result(result, width=width)


if __name__ == "__main__":
    main()
