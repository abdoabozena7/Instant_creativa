# NG12 blind evaluation human adjudication

The frozen retrieval and generation system must remain unchanged until this review is complete. This is evidence adjudication, not patient-care guidance.

## Review artifact

Use `outputs/ng12_blind_adjudication_v1/ng12_human_adjudication_v1.xlsx`.

- `Review Guide` defines the decision rubric.
- `Case Review` contains the 44 end-to-end responses.
- `Claim Review` contains 134 model-proposed atomic claims and their cited evidence.
- `Evidence Index` contains every retrieved passage supplied to generation.
- `Metrics Summary` calculates official rates from the adjudicated `Final` columns only.

The model-proposed claim decomposition is not treated as a judgment. Reviewers must verify on `Case Review` that every material claim in the answer appears in `Claim Review`. If the claim list is incomplete, set `Final claim list complete` to `FALSE`; official claim metrics must not be reported until the missing claim is added and reviewed.

## Reviewer workflow

1. Reviewer 1 completes only `R1` columns.
2. Reviewer 2 independently completes only `R2` columns.
3. A third reviewer or consensus meeting resolves disagreements into `Final` columns.
4. `UNCERTAIN` is allowed during independent review but not in final scoring.
5. Do not use the provisional same-model verdicts as an answer key.
6. Use only the question, generated answer, supplied evidence, gold behavior, and current-versus-historical authority rule.

For every claim, evaluate two separate questions:

- `Final supported`: does the full atomic claim follow directly from any supplied evidence?
- `Final citation entails`: does at least one citation attached to that claim support its full meaning?

A claim can be supported somewhere in the evidence but still have an incorrect or missing citation. Added exclusions, workflow steps, certainty, universality, or clinical conclusions are unsupported unless the evidence explicitly supplies them.

## Produce official metrics

After adjudication, export the two worksheets as UTF-8 CSV files:

- `Case Review` → `data/eval/human_case_review_v1.csv`
- `Claim Review` → `data/eval/human_claim_review_v1.csv`

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\score_human_adjudication.py
```

The scorer fails closed if:

- there are not exactly 44 cases,
- any final required decision is blank or `UNCERTAIN`,
- case or claim identifiers are missing or duplicated,
- a cited claim is marked `NOT_APPLICABLE` for citation entailment, or
- any case has an incomplete claim list.

Only after `human_adjudication_metrics_v1.json` is produced should Experiments A, B, and C begin. Each experiment gets a new architecture freeze and changes only one subsystem.
