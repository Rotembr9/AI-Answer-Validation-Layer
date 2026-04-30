# AI Answer Validation Layer (MVP)

Local, explainable validation of an AI-generated answer against a short source document. The system does **not** generate answers and does **not** call external APIs or LLMs.

## Demo

**One-minute demo:** prove the labeled harness passes end-to-end.

```powershell
cd <repo-root>
$env:PYTHONPATH = "$PWD\src"
python tests/evaluate_examples.py
```

You should see **30** examples, **100%** accuracy on the bundled policy dataset, **Supported precision 1.0**, **Not Supported recall 1.0**, and **0** rows where a non-Supported gold label is predicted as Supported (strict NS→Supported).

**Try one wrong answer** (same policy as `data/examples.json`):

```powershell
$env:PYTHONPATH = "$PWD\src"
python -c "from pathlib import Path; import json; from validator import validate; doc=json.loads(Path('data/examples.json').read_text(encoding='utf-8'))['source_document']; print(validate('How many remote days without approval?','Five days, no approval needed.',doc))"
```

Expect **`Not Supported`** (the source allows **3** days, not 5) plus `evidence` lines from the document.

### CLI demo (`run_demo.py`)

The **source document** is always the bundled policy in `data/examples.json`. You are only asked for a **question** and an **answer** (nothing else).

**Option A — interactive** (run in a real terminal; you will see two labeled prompts):

```powershell
cd <repo-root>
python run_demo.py
```

**Option B — one command, no typing at prompts** (safe for PowerShell: use **single quotes** on the answer so `$` is not treated as a variable):

```powershell
cd <repo-root>
python run_demo.py --question "What is the annual cap?" --answer 'Up to $500 per year for approved expenses.'
```

Shorthand: `-q` / `-a`. If you run `run_demo.py` with **no** arguments while stdin is **not** a TTY (e.g. an empty pipe), the script **exits with an error** instead of waiting — use `--question` and `--answer` instead.

The script adds `src` to the import path automatically; no `PYTHONPATH` is required.

### Optional web UI (`web_demo.py`)

Single-page form: question, answer, and document (defaults to the bundled policy). Install Flask first:

```powershell
pip install -r requirements-web.txt
python web_demo.py
```

Open **http://127.0.0.1:5000**, click **Validate**, and read the JSON result (`verdict`, `confidence`, `reason`, `evidence`).

## Layout

- `data/examples.json` — one policy document and 30 labeled examples.
- `data/examples_holdout.json` — 30 **held-out** labeled examples (same document; different questions/answers).
- `src/validator/` — validation pipeline (`text_utils`, `scoring`, `validator`).
- `run_demo.py` — interactive CLI demo (bundled document).
- `web_demo.py` — optional Flask UI (same validator; paste any document).
- `requirements-web.txt` — Flask only, for `web_demo.py`.
- `tests/evaluate_examples.py` — runs all examples and prints accuracy metrics.
- `tests/generate_report.py` — builds `reports/evaluation_report.md` / `.html`.

## Run evaluation

From the project root (requires Python 3.10+). On Windows, if `python` is the Store stub, install Python from [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12`, then open a **new** terminal so `python` / `py` are on `PATH`.

```powershell
$env:PYTHONPATH = "$PWD\src"
python tests/evaluate_examples.py
```

Alternative (`PYTHONPATH` must include the repo root for `-m`):

```powershell
$env:PYTHONPATH = "$PWD"
python -m tests.evaluate_examples
```

Holdout evaluation (30 unseen labeled examples, same document):

```powershell
python tests/evaluate_examples.py data/examples_holdout.json
```

### Human-readable evaluation report (Markdown + HTML)

Runs both datasets and writes **`reports/evaluation_report.md`** and **`reports/evaluation_report.html`** (executive summary, metrics, failed rows with evidence, safety highlights, plain-English notes). Saves **`reports/last_run_metrics.json`** so the next run can summarize what changed.

```powershell
$env:PYTHONPATH = "$PWD\src"
python tests/generate_report.py
```

Open `reports/evaluation_report.html` in a browser, or read the `.md` file in any editor.

## Use the validator in code

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))

from validator import validate

result = validate(
    question="What is the remote work limit?",
    answer="Up to 3 days per week without extra approval.",
    source_document="...",
)
# result -> { "verdict", "confidence", "reason", "evidence" }
```

## Approach (MVP)

- **Evidence similarity**: TF–IDF cosine between the question+answer text and each document line; top lines are `evidence`.
- **Keywords**: Jaccard overlap between question+answer tokens and the document.
- **Numbers**: integers and dollar amounts in the answer must appear in the source (after normalizing ordinals like “15th”).
- **Contradictions**: simple polarity cues and a few policy-specific patterns (eligibility, equipment return, SLA wording).

Thresholds and weights live in `src/validator/scoring.py` (see comments there).
