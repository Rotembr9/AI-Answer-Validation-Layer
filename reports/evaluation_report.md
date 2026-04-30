# AI Answer Validation — Evaluation report

*Generated: 2026-04-30 13:01:41 (UTC)*

---

## A. Executive summary

- **Overall status:** **Healthy for a cautious demo** — no high-risk false Supported labels on either dataset.
- **Safe enough for demo (product view):** Yes — no gold *Not Supported* or *Partial* row was predicted as *Supported*.
- **False *Supported* cases (strict — gold was *Not Supported*):** **0** (main: 0, holdout: 0)
- **False *Supported* cases (any non-Supported gold):** **0** (main: 0, holdout: 0)
- **Main risk areas:** Incorrect *Supported* verdicts (if any); low holdout accuracy (mixed claims and verbal numbers); *Partial* vs *Not Supported* confusion on negative examples.

### Changes since last report

Compared to the previous `reports/last_run_metrics.json` snapshot:

#### `examples.json`
- **Accuracy:** 1.0000 (was 0.8667)
- **Strict false Supported:** 0 (was 0)
- **Any false Supported:** 0 (was 0)

#### `examples_holdout.json`
- **Accuracy:** 0.8000 (was 0.4000)
- **Strict false Supported:** 0 (was 0)
- **Any false Supported:** 0 (was 0)

The snapshot file is overwritten each run so the *next* report can compare to this one.

---

## B. Metrics

### `data/examples.json`

| Metric | Value |
|--------|-------|
| Total examples | 30 |
| Accuracy | 1.0000 (30/30) |
| Supported precision | 1.0000 |
| Not Supported recall | 1.0000 |
| False Supported (strict: gold NS → pred Supported) | **0** |
| False Supported (any gold ≠ Supported → pred Supported) | **0** |

**Confusion matrix** (rows = gold label, columns = predicted):

| (gold → pred) | Supported | Not Supported | Partial |
|-----------------|-----------|-----------------|---------|
| **Supported** | 10 | 0 | 0 |
| **Not Supported** | 0 | 10 | 0 |
| **Partial** | 0 | 0 | 10 |

### `data/examples_holdout.json`

| Metric | Value |
|--------|-------|
| Total examples | 30 |
| Accuracy | 0.8000 (24/30) |
| Supported precision | 0.7000 |
| Not Supported recall | 1.0000 |
| False Supported (strict: gold NS → pred Supported) | **0** |
| False Supported (any gold ≠ Supported → pred Supported) | **0** |

**Confusion matrix** (rows = gold label, columns = predicted):

| (gold → pred) | Supported | Not Supported | Partial |
|-----------------|-----------|-----------------|---------|
| **Supported** | 7 | 0 | 3 |
| **Not Supported** | 0 | 10 | 0 |
| **Partial** | 0 | 3 | 7 |

---

## Product summary

### Safe for demo?

**Yes — for a cautious internal or pilot demo.**

Criteria used: zero predictions of *Supported* when the reference label is *Not Supported* or *Partial* (strict and broad counts both zero on main and holdout).

### Current limitations

- **Main set (`data/examples.json`):** accuracy **100.0%** (30/30); gold *Supported* recall **100.0%** (10/10 correct as *Supported*).
- **Holdout (`data/examples_holdout.json`):** accuracy **80.0%** (24/30); gold *Supported* recall **70.0%** (7/10); *Not Supported* recall **100.0%**.

- **Remaining labeled mismatches:** **6** row(s) across both sets (see §C). Mostly *Supported*→*Partial*, *Partial*→*Not Supported*, or borderline evidence scores — not safety violations.

### Next recommended improvement

- **Ranking / evidence:** Improve alignment scores for paraphrases whose best matching policy line is correct but TF–IDF cosine is modest (reduces *Supported*→*Partial*).
- **Optional:** Small offline embedding model for evidence retrieval (still no external API calls).
- **UX:** Surface `reason` and `evidence` prominently so *Partial* feels actionable.

---

## C. Remaining mismatches only (gold ≠ predicted)

### `data/examples.json` — 0 mismatch(es)

*None — all labels match.*

### `data/examples_holdout.json` — 6 mismatch(es)

| ID | Expected | Predicted | Conf | Question (short) |
|----|----------|-----------|------|------------------|
| H-S06 | Supported | Partial | 0.483 | What risk is mentioned if I do not return my company laptop after leaving? |
| H-S09 | Supported | Partial | 0.486 | How long do I have after my job ends to send back the company laptop? |
| H-S10 | Supported | Partial | 0.496 | Are purchases made in January reported in January? |
| H-P03 | Partial | Not Supported | 0.132 | What are the support SLAs? |
| H-P06 | Partial | Not Supported | 0.000 | Expense timing for Q1 purchases? |
| H-P07 | Partial | Not Supported | 0.138 | Who gets remote stipend money? |

<details>
<summary>Full detail (question, answer, reason, evidence)</summary>

#### H-S06

- **Expected:** Supported  
- **Predicted:** Partial  
- **Confidence:** 0.483

**Question:** What risk is mentioned if I do not return my company laptop after leaving?

**Answer:** The policy says payroll deduction up to the replacement cost may apply.

**Reason (model):** Some statements align with the source, but not all details or conditions are clearly satisfied (similarity 0.35, numbers score 1.00).

**Evidence lines:**
1. 6. Equipment: Company laptops remain company property and must be returned within 10 business days after employment ends; failure to return may result in payroll deduction up to the replacement cost.
2. 4. Submission deadline: Expense reports must be submitted by the 15th day of the month following the purchase. Reports received after the 15th are denied for that purchase period.
3. 2. Remote work: Employees may work remotely up to 3 days per week without extra approval. A 4th remote day in the same week requires written manager approval received before that week begins.

#### H-S09

- **Expected:** Supported  
- **Predicted:** Partial  
- **Confidence:** 0.486

**Question:** How long do I have after my job ends to send back the company laptop?

**Answer:** Within 10 business days after employment ends.

**Reason (model):** Some statements align with the source, but not all details or conditions are clearly satisfied (similarity 0.36, numbers score 1.00).

**Evidence lines:**
1. 6. Equipment: Company laptops remain company property and must be returned within 10 business days after employment ends; failure to return may result in payroll deduction up to the replacement cost.
2. 5. Support response time: Non-urgent tickets receive a first response within 2 business days; urgent (Severity 1) tickets require a first response within 4 business hours.
3. 4. Submission deadline: Expense reports must be submitted by the 15th day of the month following the purchase. Reports received after the 15th are denied for that purchase period.

#### H-S10

- **Expected:** Supported  
- **Predicted:** Partial  
- **Confidence:** 0.496

**Question:** Are purchases made in January reported in January?

**Answer:** Reports go to the following month; January purchases would be reported by the 15th of the next month.

**Reason (model):** Some statements align with the source, but not all details or conditions are clearly satisfied (similarity 0.37, numbers score 1.00).

**Evidence lines:**
1. 4. Submission deadline: Expense reports must be submitted by the 15th day of the month following the purchase. Reports received after the 15th are denied for that purchase period.
2. REMOTE WORK AND EXPENSE POLICY (Effective January 1, 2025)
3. 6. Equipment: Company laptops remain company property and must be returned within 10 business days after employment ends; failure to return may result in payroll deduction up to the replacement cost.

#### H-P03

- **Expected:** Partial  
- **Predicted:** Not Supported  
- **Confidence:** 0.132

**Question:** What are the support SLAs?

**Answer:** Severity 1 urgent tickets get 4 business hours; the policy does not define any timeframe for non-urgent tickets.

**Reason (model):** Strong contradiction or unreliable numeric claims versus the source.

**Evidence lines:**
1. 5. Support response time: Non-urgent tickets receive a first response within 2 business days; urgent (Severity 1) tickets require a first response within 4 business hours.
2. 4. Submission deadline: Expense reports must be submitted by the 15th day of the month following the purchase. Reports received after the 15th are denied for that purchase period.
3. 1. Eligible employees: Full-time staff only; contractors are not eligible for the remote stipend or annual equipment reimbursement.

#### H-P06

- **Expected:** Partial  
- **Predicted:** Not Supported  
- **Confidence:** 0.0

**Question:** Expense timing for Q1 purchases?

**Answer:** File by the 15th of the month after purchase; if you bought in March you could wait until April 30.

**Reason (model):** One or more numbers in the answer do not appear in the source document.

**Evidence lines:**
1. 4. Submission deadline: Expense reports must be submitted by the 15th day of the month following the purchase. Reports received after the 15th are denied for that purchase period.
2. 6. Equipment: Company laptops remain company property and must be returned within 10 business days after employment ends; failure to return may result in payroll deduction up to the replacement cost.
3. 1. Eligible employees: Full-time staff only; contractors are not eligible for the remote stipend or annual equipment reimbursement.

#### H-P07

- **Expected:** Partial  
- **Predicted:** Not Supported  
- **Confidence:** 0.138

**Question:** Who gets remote stipend money?

**Answer:** Full-time employees can receive it; the policy also references contractors not being eligible but does not describe hybrid roles.

**Reason (model):** Strong contradiction or unreliable numeric claims versus the source.

**Evidence lines:**
1. 1. Eligible employees: Full-time staff only; contractors are not eligible for the remote stipend or annual equipment reimbursement.
2. REMOTE WORK AND EXPENSE POLICY (Effective January 1, 2025)
3. 3. Annual home-office stipend: Up to $500 per calendar year for approved expenses; amounts above $500 are not reimbursed.

</details>

---

## D. Safety — high-risk predictions

**These are the worst mistakes for product trust:** predicting ***Supported*** when the reference label is ***Not Supported*** or ***Partial***.

### Result

**None.** No high-risk *Supported* predictions on either dataset.

---

## E. Plain-English interpretation

**What is working well:** The validator did not label any example whose correct label is *Not Supported* or *Partial* as *Supported* on either dataset. That is the most important safety signal for a demo.

**Accuracy:** On the main labeled set (examples.json), **30/30** (100.0%) examples match the gold label. On the holdout set (examples_holdout.json), **24/30** (80.0%) match.

**Supported precision** (when the model says *Supported*, how often that is correct) is moderate on one or both sets — many gold *Supported* rows may show as *Partial* instead. That is conservative and safer than false *Supported*, but worth improving for UX.

**What to improve next:** Improve recall on clearly false numeric or SLA claims without raising false *Supported* rates; optionally add clearer explanations in the UI when the verdict is *Partial*.

---

*End of report.*
