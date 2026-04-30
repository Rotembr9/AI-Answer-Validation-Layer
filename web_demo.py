"""
Minimal local web UI (optional). Requires Flask only for this file.

    pip install -r requirements-web.txt
    python web_demo.py

Open http://127.0.0.1:5000 — no external APIs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, request, render_template_string  # noqa: E402

from validator import validate  # noqa: E402

app = Flask(__name__)

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AI Answer Validation (local MVP)</title>
  <style>
    body { font-family: system-ui, Segoe UI, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }
    h1 { font-size: 1.25rem; }
    label { display: block; margin-top: 1rem; font-weight: 600; font-size: 0.9rem; }
    textarea { width: 100%; box-sizing: border-box; padding: 0.5rem; font-size: 0.95rem; }
    textarea.doc { min-height: 12rem; }
    textarea.short { min-height: 5rem; }
    button { margin-top: 1rem; padding: 0.45rem 1rem; cursor: pointer; }
    pre { background: #f1f3f5; padding: 1rem; overflow: auto; font-size: 0.85rem; border-radius: 4px; }
    .hint { color: #495057; font-size: 0.85rem; margin-top: 0.25rem; }
  </style>
</head>
<body>
  <h1>AI Answer Validation Layer</h1>
  <p class="hint">Runs entirely on your machine. Paste a source document or keep the default from data/examples.json.</p>
  <form method="post" action="/">
    <label for="q">Question</label>
    <textarea class="short" id="q" name="question" required>{{ question }}</textarea>
    <label for="a">AI answer</label>
    <textarea class="short" id="a" name="answer" required>{{ answer }}</textarea>
    <label for="d">Source document</label>
    <textarea class="doc" id="d" name="document" required>{{ document }}</textarea>
    <button type="submit">Validate</button>
  </form>
{% if result %}
  <h2 style="margin-top:1.5rem;font-size:1.1rem;">Result (JSON)</h2>
  <pre>{{ result }}</pre>
{% endif %}
</body>
</html>
"""


def load_bundled_document() -> str:
    path = ROOT / "data" / "examples.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["source_document"]


@app.route("/", methods=["GET", "POST"])
def index():
    default_doc = load_bundled_document()
    if request.method == "GET":
        return render_template_string(
            PAGE, question="", answer="", document=default_doc, result=None
        )

    q = (request.form.get("question") or "").strip()
    a = (request.form.get("answer") or "").strip()
    d = (request.form.get("document") or "").strip()
    result_html = None
    if q and a and d:
        out = validate(q, a, d)
        result_html = json.dumps(out, indent=2, ensure_ascii=False)
    return render_template_string(
        PAGE, question=q, answer=a, document=d, result=result_html
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
