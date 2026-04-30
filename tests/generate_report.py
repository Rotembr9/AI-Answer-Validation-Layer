"""
Generate a human-readable evaluation report (Markdown + optional HTML).

Run from repo root:
    python tests/generate_report.py

Outputs:
    reports/evaluation_report.md
    reports/evaluation_report.html

Does not import or modify the validator beyond calling validate() on labeled data.
"""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validator import load_examples_json, validate  # noqa: E402

LABELS = ("Supported", "Not Supported", "Partial")

SNAPSHOT_PATH = ROOT / "reports" / "last_run_metrics.json"


@dataclass
class DatasetResult:
    name: str
    path: str
    n: int
    correct: int
    accuracy: float
    supported_precision: float
    ns_recall: float
    strict_false_supported: int  # gold NS -> pred Supported
    any_false_supported: int  # gold != Supported -> pred Supported
    confusion: dict[tuple[str, str], int]
    rows: list[dict]  # per-example: id, question, answer, expected, predicted, confidence, reason, evidence
    failed: list[dict]
    high_risk: list[dict]  # NS or Partial gold -> Supported pred


def _confusion_matrix(
    expected: list[str], predicted: list[str]
) -> dict[tuple[str, str], int]:
    m: dict[tuple[str, str], int] = {}
    for e, p in zip(expected, predicted):
        m[(e, p)] = m.get((e, p), 0) + 1
    return m


def _format_matrix_md(cells: dict[tuple[str, str], int]) -> str:
    lines = [
        "| (gold → pred) | Supported | Not Supported | Partial |",
        "|-----------------|-----------|-----------------|---------|",
    ]
    for row in LABELS:
        line = f"| **{row}** |"
        for col in LABELS:
            line += f" {cells.get((row, col), 0)} |"
        lines.append(line)
    return "\n".join(lines)


def _run_dataset(rel_path: str) -> DatasetResult:
    path = ROOT / rel_path
    source_document, examples = load_examples_json(path)
    expected: list[str] = []
    predicted: list[str] = []
    supported_tp = supported_fp = 0
    ns_actual = ns_tp = 0
    rows: list[dict] = []
    for ex in examples:
        exp = ex["expected_label"]
        r = validate(ex["question"], ex["answer"], source_document)
        pred = r["verdict"]
        expected.append(exp)
        predicted.append(pred)
        rows.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "expected": exp,
                "predicted": pred,
                "confidence": r["confidence"],
                "reason": r["reason"],
                "evidence": r.get("evidence", []),
            }
        )
        if exp == "Supported":
            if pred == "Supported":
                supported_tp += 1
            else:
                supported_fp += 1
        if exp == "Not Supported":
            ns_actual += 1
            if pred == "Not Supported":
                ns_tp += 1

    n = len(examples)
    correct = sum(1 for e, p in zip(expected, predicted) if e == p)
    acc = correct / n if n else 0.0
    sp = (
        supported_tp / (supported_tp + supported_fp)
        if (supported_tp + supported_fp)
        else 0.0
    )
    ns_r = ns_tp / ns_actual if ns_actual else 0.0
    strict_fs = sum(
        1 for e, p in zip(expected, predicted) if e == "Not Supported" and p == "Supported"
    )
    any_fs = sum(
        1 for e, p in zip(expected, predicted) if e != "Supported" and p == "Supported"
    )
    failed = [row for row in rows if row["expected"] != row["predicted"]]
    high_risk = [
        row
        for row in rows
        if row["predicted"] == "Supported" and row["expected"] != "Supported"
    ]

    return DatasetResult(
        name=path.name,
        path=str(path.relative_to(ROOT)).replace("\\", "/"),
        n=n,
        correct=correct,
        accuracy=acc,
        supported_precision=sp,
        ns_recall=ns_r,
        strict_false_supported=strict_fs,
        any_false_supported=any_fs,
        confusion=_confusion_matrix(expected, predicted),
        rows=rows,
        failed=failed,
        high_risk=high_risk,
    )


def _snapshot_metrics(dr: DatasetResult) -> dict:
    return {
        "n": dr.n,
        "accuracy": round(dr.accuracy, 6),
        "supported_precision": round(dr.supported_precision, 6),
        "ns_recall": round(dr.ns_recall, 6),
        "strict_false_supported": dr.strict_false_supported,
        "any_false_supported": dr.any_false_supported,
    }


def _load_previous_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.is_file():
        return None
    try:
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _product_summary(core: DatasetResult, hold: DatasetResult) -> str:
    """
    Short product-facing block: demo readiness, limitations, next step.
    Uses only DatasetResult fields (no validator imports).
    """
    safe_demo = (
        core.strict_false_supported == 0
        and core.any_false_supported == 0
        and hold.strict_false_supported == 0
        and hold.any_false_supported == 0
    )
    total_mismatch = len(core.failed) + len(hold.failed)

    sup_main = core.confusion.get(("Supported", "Supported"), 0)
    sup_row = sum(core.confusion.get(("Supported", c), 0) for c in LABELS)
    sup_recall_main = sup_main / sup_row if sup_row else 0.0

    sup_h = hold.confusion.get(("Supported", "Supported"), 0)
    sup_row_h = sum(hold.confusion.get(("Supported", c), 0) for c in LABELS)
    sup_recall_hold = sup_h / sup_row_h if sup_row_h else 0.0

    lines = [
        "### Safe for demo?",
        "",
        "**"
        + (
            "Yes — for a cautious internal or pilot demo."
            if safe_demo
            else "No — resolve false *Supported* rows before customer-facing use."
        )
        + "**",
        "",
        "Criteria used: zero predictions of *Supported* when the reference label is "
        "*Not Supported* or *Partial* (strict and broad counts both zero on main and holdout).",
        "",
        "### Current limitations",
        "",
        f"- **Main set (`{core.path}`):** accuracy **{core.accuracy:.1%}** ({core.correct}/{core.n}); "
        f"gold *Supported* recall **{sup_recall_main:.1%}** ({sup_main}/{sup_row} correct as *Supported*).",
        f"- **Holdout (`{hold.path}`):** accuracy **{hold.accuracy:.1%}** ({hold.correct}/{hold.n}); "
        f"gold *Supported* recall **{sup_recall_hold:.1%}** ({sup_h}/{sup_row_h}); "
        f"*Not Supported* recall **{hold.ns_recall:.1%}**.",
        "",
    ]
    if total_mismatch:
        lines.append(
            f"- **Remaining labeled mismatches:** **{total_mismatch}** row(s) across both sets "
            "(see §C). Mostly *Supported*→*Partial*, *Partial*→*Not Supported*, or borderline "
            "evidence scores — not safety violations."
        )
    else:
        lines.append("- **Remaining labeled mismatches:** none — full agreement on both datasets.")

    lines.extend(
        [
            "",
            "### Next recommended improvement",
            "",
            "- **Ranking / evidence:** Improve alignment scores for paraphrases whose best matching "
            "policy line is correct but TF–IDF cosine is modest (reduces *Supported*→*Partial*).",
            "- **Optional:** Small offline embedding model for evidence retrieval (still no external API calls).",
            "- **UX:** Surface `reason` and `evidence` prominently so *Partial* feels actionable.",
            "",
        ]
    )
    return "\n".join(lines)


def _product_summary_html(core: DatasetResult, hold: DatasetResult) -> str:
    """HTML companion to _product_summary (keep text in sync)."""
    esc = html.escape
    safe_demo = (
        core.strict_false_supported == 0
        and core.any_false_supported == 0
        and hold.strict_false_supported == 0
        and hold.any_false_supported == 0
    )
    total_mismatch = len(core.failed) + len(hold.failed)
    sup_main = core.confusion.get(("Supported", "Supported"), 0)
    sup_row = sum(core.confusion.get(("Supported", c), 0) for c in LABELS)
    sup_recall_main = sup_main / sup_row if sup_row else 0.0
    sup_h = hold.confusion.get(("Supported", "Supported"), 0)
    sup_row_h = sum(hold.confusion.get(("Supported", c), 0) for c in LABELS)
    sup_recall_hold = sup_h / sup_row_h if sup_row_h else 0.0

    lim = (
        f"<strong>Remaining labeled mismatches:</strong> <strong>{total_mismatch}</strong> row(s) across both sets "
        f"(see §C). Mostly <em>Supported</em>→<em>Partial</em>, <em>Partial</em>→<em>Not Supported</em>, or borderline "
        f"evidence scores — not safety violations."
        if total_mismatch
        else "<strong>Remaining labeled mismatches:</strong> none — full agreement on both datasets."
    )

    demo_line = (
        "Yes — for a cautious internal or pilot demo."
        if safe_demo
        else "No — resolve false <em>Supported</em> rows before customer-facing use."
    )

    return (
        f"""
  <h2>Product summary</h2>
  <h3>Safe for demo?</h3>
  <p><strong>{demo_line}</strong></p>
  <p>Criteria used: zero predictions of <em>Supported</em> when the reference label is
    <em>Not Supported</em> or <em>Partial</em> (strict and broad counts both zero on main and holdout).</p>
  <h3>Current limitations</h3>
  <ul>
    <li><strong>Main set (<code>{esc(core.path)}</code>):</strong> accuracy <strong>{core.accuracy:.1%}</strong>
      ({core.correct}/{core.n}); gold <em>Supported</em> recall <strong>{sup_recall_main:.1%}</strong>
      ({sup_main}/{sup_row} correct as <em>Supported</em>).</li>
    <li><strong>Holdout (<code>{esc(hold.path)}</code>):</strong> accuracy <strong>{hold.accuracy:.1%}</strong>
      ({hold.correct}/{hold.n}); gold <em>Supported</em> recall <strong>{sup_recall_hold:.1%}</strong>
      ({sup_h}/{sup_row_h}); <em>Not Supported</em> recall <strong>{hold.ns_recall:.1%}</strong>.</li>
    <li>{lim}</li>
  </ul>
  <h3>Next recommended improvement</h3>
  <ul>
    <li><strong>Ranking / evidence:</strong> Improve alignment scores for paraphrases whose best matching
      policy line is correct but TF–IDF cosine is modest (reduces <em>Supported</em>→<em>Partial</em>).</li>
    <li><strong>Optional:</strong> Small offline embedding model for evidence retrieval (still no external API calls).</li>
    <li><strong>UX:</strong> Surface <code>reason</code> and <code>evidence</code> prominently so <em>Partial</em> feels actionable.</li>
  </ul>
"""
    )


def _plain_english(core: DatasetResult, hold: DatasetResult) -> str:
    parts = []
    safe = (
        core.strict_false_supported == 0
        and core.any_false_supported == 0
        and hold.strict_false_supported == 0
        and hold.any_false_supported == 0
    )
    if safe:
        parts.append(
            "**What is working well:** The validator did not label any example whose correct "
            "label is *Not Supported* or *Partial* as *Supported* on either dataset. "
            "That is the most important safety signal for a demo."
        )
    else:
        parts.append(
            "**Risk:** Some answers that should not be fully supported were marked *Supported*. "
            "Those rows need review before trusting the tool with customers."
        )

    parts.append(
        f"**Accuracy:** On the main labeled set ({core.name}), **{core.correct}/{core.n}** "
        f"({core.accuracy:.1%}) examples match the gold label. "
        f"On the holdout set ({hold.name}), **{hold.correct}/{hold.n}** "
        f"({hold.accuracy:.1%}) match."
    )

    if core.supported_precision < 0.85 or hold.supported_precision < 0.85:
        parts.append(
            "**Supported precision** (when the model says *Supported*, how often that is correct) "
            "is moderate on one or both sets — many gold *Supported* rows may show as *Partial* instead. "
            "That is conservative and safer than false *Supported*, but worth improving for UX."
        )

    if hold.ns_recall < 0.7:
        parts.append(
            "**Not Supported recall** on the holdout set is limited: several wrong answers are "
            "classified as *Partial* rather than *Not Supported*. Heuristics for contradictions "
            "and verbal numbers could be strengthened next."
        )

    parts.append(
        "**What to improve next:** Improve recall on clearly false numeric or SLA claims without "
        "raising false *Supported* rates; optionally add clearer explanations in the UI when "
        "the verdict is *Partial*."
    )
    return "\n\n".join(parts)


def build_markdown(
    core: DatasetResult,
    hold: DatasetResult,
    prev: dict | None,
    generated_at: str,
) -> str:
    total_risk = core.any_false_supported + hold.any_false_supported
    strict_total = core.strict_false_supported + hold.strict_false_supported
    demo_ok = strict_total == 0 and total_risk == 0

    lines: list[str] = []
    lines.append("# AI Answer Validation — Evaluation report")
    lines.append("")
    lines.append(f"*Generated: {generated_at} (UTC)*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## A. Executive summary")
    lines.append("")
    lines.append(
        f"- **Overall status:** "
        f"{'**Healthy for a cautious demo** — no high-risk false Supported labels on either dataset.' if demo_ok else '**Review required** — see Safety section for incorrect *Supported* predictions.'}"
    )
    lines.append(
        f"- **Safe enough for demo (product view):** {'Yes — no gold *Not Supported* or *Partial* row was predicted as *Supported*.' if demo_ok else 'Not yet — fix or caveat high-risk rows before external demo.'}"
    )
    lines.append(
        f"- **False *Supported* cases (strict — gold was *Not Supported*):** **{strict_total}** "
        f"(main: {core.strict_false_supported}, holdout: {hold.strict_false_supported})"
    )
    lines.append(
        f"- **False *Supported* cases (any non-Supported gold):** **{total_risk}** "
        f"(main: {core.any_false_supported}, holdout: {hold.any_false_supported})"
    )
    lines.append(
        "- **Main risk areas:** Incorrect *Supported* verdicts (if any); low holdout accuracy "
        "(mixed claims and verbal numbers); *Partial* vs *Not Supported* confusion on negative examples."
    )
    lines.append("")
    lines.append("### Changes since last report")
    lines.append("")
    if prev and "datasets" in prev:
        lines.append("Compared to the previous `reports/last_run_metrics.json` snapshot:")
        lines.append("")
        for key, dr in (("examples.json", core), ("examples_holdout.json", hold)):
            old_ds = prev["datasets"].get(key)
            lines.append(f"#### `{key}`")
            if not old_ds:
                lines.append("- *(no prior metrics for this file)*")
                lines.append("")
                continue
            oa, na = old_ds.get("accuracy"), dr.accuracy
            osf, nsf = old_ds.get("strict_false_supported"), dr.strict_false_supported
            oaf, naf = old_ds.get("any_false_supported"), dr.any_false_supported
            lines.append(f"- **Accuracy:** {na:.4f} (was {oa:.4f})")
            lines.append(
                f"- **Strict false Supported:** {nsf} (was {osf})"
            )
            lines.append(
                f"- **Any false Supported:** {naf} (was {oaf})"
            )
            lines.append("")
        lines.append(
            "The snapshot file is overwritten each run so the *next* report can compare to this one."
        )
    else:
        lines.append("*No previous snapshot found.* After this run, metrics are saved to `reports/last_run_metrics.json` for diffing next time.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## B. Metrics")
    lines.append("")
    for dr in (core, hold):
        lines.append(f"### `{dr.path}`")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total examples | {dr.n} |")
        lines.append(f"| Accuracy | {dr.accuracy:.4f} ({dr.correct}/{dr.n}) |")
        lines.append(f"| Supported precision | {dr.supported_precision:.4f} |")
        lines.append(f"| Not Supported recall | {dr.ns_recall:.4f} |")
        lines.append(f"| False Supported (strict: gold NS → pred Supported) | **{dr.strict_false_supported}** |")
        lines.append(f"| False Supported (any gold ≠ Supported → pred Supported) | **{dr.any_false_supported}** |")
        lines.append("")
        lines.append("**Confusion matrix** (rows = gold label, columns = predicted):")
        lines.append("")
        lines.append(_format_matrix_md(dr.confusion))
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Product summary")
    lines.append("")
    lines.append(_product_summary(core, hold))
    lines.append("---")
    lines.append("")
    lines.append("## C. Remaining mismatches only (gold ≠ predicted)")
    lines.append("")
    for dr in (core, hold):
        lines.append(f"### `{dr.path}` — {len(dr.failed)} mismatch(es)")
        lines.append("")
        if not dr.failed:
            lines.append("*None — all labels match.*")
            lines.append("")
            continue
        lines.append("| ID | Expected | Predicted | Conf | Question (short) |")
        lines.append("|----|----------|-----------|------|------------------|")
        for row in dr.failed:
            qshort = row["question"].replace("|", "\\|")[:80]
            if len(row["question"]) > 80:
                qshort += "…"
            lines.append(
                f"| {row['id']} | {row['expected']} | {row['predicted']} | {row['confidence']:.3f} | {qshort} |"
            )
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Full detail (question, answer, reason, evidence)</summary>")
        lines.append("")
        for row in dr.failed:
            lines.append(f"#### {row['id']}")
            lines.append("")
            lines.append(f"- **Expected:** {row['expected']}  \n- **Predicted:** {row['predicted']}  \n- **Confidence:** {row['confidence']}")
            lines.append("")
            lines.append(f"**Question:** {row['question']}")
            lines.append("")
            lines.append(f"**Answer:** {row['answer']}")
            lines.append("")
            lines.append(f"**Reason (model):** {row['reason']}")
            lines.append("")
            lines.append("**Evidence lines:**")
            for i, ev in enumerate(row["evidence"], 1):
                lines.append(f"{i}. {ev}")
            lines.append("")
        lines.append("</details>")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## D. Safety — high-risk predictions")
    lines.append("")
    lines.append(
        "**These are the worst mistakes for product trust:** predicting ***Supported*** when the "
        "reference label is ***Not Supported*** or ***Partial***."
    )
    lines.append("")
    all_risk = []
    for dr in (core, hold):
        for row in dr.high_risk:
            all_risk.append({**row, "_dataset": dr.path})
    if not all_risk:
        lines.append("### Result")
        lines.append("")
        lines.append("**None.** No high-risk *Supported* predictions on either dataset.")
        lines.append("")
    else:
        lines.append("### **HIGH RISK — manual review**")
        lines.append("")
        for row in all_risk:
            lines.append(
                f"- **{row['id']}** (`{row['_dataset']}`) — gold **{row['expected']}**, predicted **Supported** (conf {row['confidence']:.3f})"
            )
        lines.append("")
        lines.append("| Dataset | ID | Gold | Question | Answer (truncated) |")
        lines.append("|---------|-------|------|----------|---------------------|")
        for row in all_risk:
            ans = row["answer"].replace("|", "\\|")[:100]
            if len(row["answer"]) > 100:
                ans += "…"
            q = row["question"].replace("|", "\\|")[:60]
            lines.append(
                f"| {row['_dataset']} | {row['id']} | {row['expected']} | {q} | {ans} |"
            )
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## E. Plain-English interpretation")
    lines.append("")
    lines.append(_plain_english(core, hold))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*End of report.*")
    lines.append("")
    return "\n".join(lines)


def build_html(
    core: DatasetResult,
    hold: DatasetResult,
    prev: dict | None,
    generated_at: str,
) -> str:
    total_risk = core.any_false_supported + hold.any_false_supported
    strict_total = core.strict_false_supported + hold.strict_false_supported
    demo_ok = strict_total == 0 and total_risk == 0

    def esc(s: str) -> str:
        return html.escape(s)

    risk_section = ""
    all_risk: list[dict] = []
    for dr in (core, hold):
        for row in dr.high_risk:
            all_risk.append({**row, "_dataset": dr.path})
    if not all_risk:
        risk_section = "<p><strong>No high-risk rows.</strong> No Supported predictions where gold was Not Supported or Partial.</p>"
    else:
        rows_html = "".join(
            f"<tr class='risk'><td>{esc(r['_dataset'])}</td><td>{esc(r['id'])}</td>"
            f"<td>{esc(r['expected'])}</td><td>{r['confidence']:.3f}</td>"
            f"<td>{esc(r['question'][:120])}</td></tr>"
            for r in all_risk
        )
        risk_section = (
            "<p class='risk-banner'><strong>HIGH RISK — incorrect Supported</strong></p>"
            f"<table><thead><tr><th>Dataset</th><th>ID</th><th>Gold</th><th>Conf</th><th>Question</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    failed_blocks = ""
    for dr in (core, hold):
        if not dr.failed:
            failed_blocks += f"<h3>{esc(dr.path)}</h3><p>No mismatches.</p>"
            continue
        tb = "".join(
            f"<tr><td>{esc(x['id'])}</td><td>{esc(x['expected'])}</td><td>{esc(x['predicted'])}</td>"
            f"<td>{x['confidence']:.3f}</td><td>{esc(x['question'][:100])}</td></tr>"
            for x in dr.failed
        )
        failed_blocks += (
            f"<h3>{esc(dr.path)} ({len(dr.failed)} mismatches)</h3>"
            f"<table><thead><tr><th>ID</th><th>Expected</th><th>Predicted</th><th>Conf</th><th>Question</th></tr></thead>"
            f"<tbody>{tb}</tbody></table>"
        )

    metrics_tables = ""
    for dr in (core, hold):
        cm = "".join(
            f"<tr><th>{esc(row)}</th>"
            + "".join(f"<td>{dr.confusion.get((row, col), 0)}</td>" for col in LABELS)
            + "</tr>"
            for row in LABELS
        )
        metrics_tables += f"""
        <h3>{esc(dr.path)}</h3>
        <table class="metrics">
          <tr><td>Total examples</td><td>{dr.n}</td></tr>
          <tr><td>Accuracy</td><td>{dr.accuracy:.4f} ({dr.correct}/{dr.n})</td></tr>
          <tr><td>Supported precision</td><td>{dr.supported_precision:.4f}</td></tr>
          <tr><td>Not Supported recall</td><td>{dr.ns_recall:.4f}</td></tr>
          <tr><td>False Supported (strict)</td><td class="{'ok' if dr.strict_false_supported == 0 else 'risk'}">{dr.strict_false_supported}</td></tr>
          <tr><td>False Supported (any)</td><td class="{'ok' if dr.any_false_supported == 0 else 'risk'}">{dr.any_false_supported}</td></tr>
        </table>
        <p>Confusion matrix (gold → predicted)</p>
        <table><thead><tr><th></th>{''.join(f'<th>{esc(c)}</th>' for c in LABELS)}</tr></thead>
        <tbody>{cm}</tbody></table>
        """

    prev_html = "<p>No prior snapshot.</p>"
    if prev and "datasets" in prev:
        prev_html = "<pre>" + esc(json.dumps(prev["datasets"], indent=2)) + "</pre><p>Compare numerically to tables below.</p>"

    status_class = "ok" if demo_ok else "warn"
    status_text = (
        "Healthy for a cautious demo — no false Supported on negative gold labels."
        if demo_ok
        else "Review required before external demo."
    )

    interp_html = "".join(
        f"<p>{esc(p)}</p>" for p in _plain_english(core, hold).split("\n\n")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Evaluation report</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    h1 {{ font-size: 1.35rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .metrics td:first-child {{ font-weight: 600; width: 14rem; }}
    .ok {{ color: #0d6832; font-weight: bold; }}
    .warn {{ color: #b06000; font-weight: bold; }}
    .risk {{ background: #ffe6e6; }}
    .risk-banner {{ color: #b00020; font-weight: bold; font-size: 1.05rem; }}
    pre {{ background: #f4f4f4; padding: 1rem; overflow: auto; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>AI Answer Validation — Evaluation report</h1>
  <p><em>Generated {esc(generated_at)} UTC</em></p>

  <h2>A. Executive summary</h2>
  <p class="{status_class}"><strong>Overall:</strong> {esc(status_text)}</p>
  <ul>
    <li><strong>Strict false Supported (gold Not Supported → Supported):</strong> {strict_total}</li>
    <li><strong>Any false Supported (gold not Supported → Supported):</strong> {total_risk}</li>
  </ul>

  <h3>Changes since last run</h3>
  {prev_html}

  <h2>B. Metrics</h2>
  {metrics_tables}
  {_product_summary_html(core, hold)}
  <h2>C. Remaining mismatches only</h2>
  {failed_blocks}

  <h2>D. Safety</h2>
  {risk_section}

  <h2>E. Plain-English interpretation</h2>
  {interp_html}
</body>
</html>
"""


def main() -> None:
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = _load_previous_snapshot()

    core = _run_dataset("data/examples.json")
    hold = _run_dataset("data/examples_holdout.json")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    md = build_markdown(core, hold, prev, generated_at)
    html_out = build_html(core, hold, prev, generated_at)

    (out_dir / "evaluation_report.md").write_text(md, encoding="utf-8")
    (out_dir / "evaluation_report.html").write_text(html_out, encoding="utf-8")

    snapshot = {
        "generated_at": generated_at,
        "datasets": {
            "examples.json": _snapshot_metrics(core),
            "examples_holdout.json": _snapshot_metrics(hold),
        },
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print(f"Wrote {out_dir / 'evaluation_report.md'}")
    print(f"Wrote {out_dir / 'evaluation_report.html'}")
    print(f"Wrote {SNAPSHOT_PATH.relative_to(ROOT)} (for next-run comparison)")


if __name__ == "__main__":
    main()
