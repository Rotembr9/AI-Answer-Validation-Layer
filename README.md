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

## Layout

- `data/examples.json` — one policy document and 30 labeled examples.
- `src/validator/` — validation pipeline (`text_utils`, `scoring`, `validator`).
- `tests/evaluate_examples.py` — runs all examples and prints accuracy metrics.

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
