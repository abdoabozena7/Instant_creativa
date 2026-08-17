# NG12 evidence console

This project parses and reconciles two NICE NG12 PDFs into clean, auditable records, builds an authority-aware hybrid retriever, and serves a citation-grounded hackathon demo with FastAPI, React, and Ollama. The configured generation model is `gpt-oss:120b-cloud`; `nomic-embed-text:latest` supplies 768-dimensional retrieval embeddings.

The clinical scope is restricted to lung, colorectal, oesophageal, stomach/gastric, pancreatic, bladder, and renal cancer. Shared definitions, patient support, safety netting, diagnostic-process material, and symptom-index rows are retained where they help interpret those sites. Other cancer sites are excluded as retrieval metadata and recommendation units.

## Source authority

The source-priority rule is non-negotiable:

- `2026_current` is the primary and canonical source for current recommendation wording, thresholds, investigations, and referral actions.
- `2015_full` is supporting material for clinical questions, evidence, tables, methodology, rationale, and committee context.
- Historical recommendation boxes from the full guideline are retained in `records_clean.jsonl` for audit, marked `retrieval_eligible: false`, and never emitted as chunks.
- Potential wording or action differences are reported, not silently merged.

## Why these parsers

Programmatic and visual inspection found a usable native text layer in both PDFs, so OCR would add error without benefit.

- The 101-page current guideline has native text on every page, usable bookmarks/headings, intact recommendation IDs, and accurate physical page numbering. PyMuPDF is used for pages, headings, and prose. Its ruled symptom tables need better cell recovery, so pdfplumber is used only for those table pages.
- The 382-page full guideline has native text on 381 pages and 69 bookmarks that provide reliable chapter boundaries. It has no recommendation IDs and uses Word-style layout, recommendation boxes, update sidebars, and complex evidence tables. Layout-aware PyMuPDF spans recover the hierarchy and recommendation boxes; table titles plus original extracted line order are retained for supporting evidence tables. No heavier document framework was justified.

This hybrid is intentionally narrow: PyMuPDF does most of the work, and pdfplumber is introduced only where the inspected source structure benefits from it.

## Submission and artifact contract

A clean clone is intentionally self-contained for the reviewed demo. The evaluated runtime snapshot is committed even though it can be rebuilt:

| Path | Submission status | Why |
|---|---|---|
| `api/`, `src/`, `scripts/`, `frontend/src`, tests, docs, manifests | Tracked source | Required to inspect, build, test, and run the project |
| `data/parsed/` | Tracked generated snapshot | Contains the reviewed 440-chunk corpus, provenance records, and merge report |
| `data/index/` | Tracked generated snapshot | Contains the matching 768-dimensional dense matrix and integrity manifest |
| `data/eval/` | Tracked evaluation evidence | Preserves the frozen blind baseline and deterministic/semantic audit trail |
| `data/raw/*.pdf` | Local/generated; not tracked | Source PDFs may be redistributed separately only when competition/licensing rules permit |
| `.venv/`, `frontend/node_modules/`, `frontend/dist/` | Local generated | Rebuilt by the bootstrap command |
| `output/`, `outputs/`, `.playwright-cli/`, `tmp/` | Local generated | Screenshots, workbook previews, and temporary run artifacts are not needed by the app |
| `.env` | Local; ignored | Secrets and machine-specific overrides must never be committed |
| `.env.example` | Tracked | Documents the required model names and URLs without secrets |

The 440 chunks and dense index are shipped because together they are small, are the exact evaluated snapshot, and let an instructor run the demo without hidden PDFs or a corpus rebuild. Rebuilding remains available for verification or source updates, but it creates a new architecture snapshot and must not be described as the frozen v1 run.

## Project layout

```text
data/
  raw/
    ng12_current_2026.pdf  # optional local source, not tracked
    ng12_full_2015.pdf     # optional local source, not tracked
  parsed/
    pages_current.jsonl
    pages_full.jsonl
    records_clean.jsonl
    chunks.jsonl
    merge_report.json
  eval/
    retrieval_cases.jsonl
    retrieval_metrics.json
    evaluation_freeze.json
    evaluation_freeze_v2.json
    evaluation_freeze_v4.json
    evaluation_freeze_v6.json
    blind_questions_v1.jsonl
    blind_gold_v1.jsonl
    blind_run_v1.jsonl
    blind_e2e_report_v1.json
    blind_run_v2.jsonl
    blind_e2e_report_v2.json
    blind_run_v4.jsonl
    blind_e2e_report_v4.json
    blind_run_v6.jsonl
    blind_e2e_report_v6.json
    adjudication_packets_v1.jsonl
    adjudication_template_v1.csv
    multi_judge_manifest_v1.json
    multi_judge_tasks_v1.jsonl
    multi_judge_deterministic_v1.jsonl
  index/
    chunk_embeddings.npy
    index_manifest.json
api/
  main.py
src/ingestion/
  cleaner.py
  pdf_parser.py
  full_pdf_parser.py
  scope_filter.py
  normalize.py
  reconcile.py
  chunker.py
  report.py
  corpus_report.py
  models.py
src/retrieval/
  bm25.py
  engine.py
  generation.py
  ollama_client.py
  scope_guard.py
src/evaluation/
  judge_clients.py
  judge_prompts.py
  multi_judge.py
scripts/
  bootstrap.ps1
  verify_runtime_artifacts.py
  parse_ng12.py
  build_corpus.py
  build_retrieval_index.py
  evaluate_retrieval.py
  freeze_evaluation_architecture.py
  run_blind_e2e_evaluation.py
  provisional_claim_adjudication.py
  score_human_adjudication.py
  build_human_adjudication_workbook.mjs
  run_multi_judge_evaluation.py
  run_multi_judge_secure.ps1
outputs/                   # local generated review artifact, not tracked
  ng12_blind_adjudication_v1/
    ng12_human_adjudication_v1.xlsx
frontend/
  src/
  public/
  package.json
tests/
  test_parsing.py
  test_corpus.py
  test_retrieval.py
```

The earlier current-guideline-only outputs (`pages.jsonl`, `scoped_records.jsonl`, and `parsing_report.json`) remain reproducible with `parse_ng12.py`; the two-source corpus is the main output.

## Run

### Fresh clone prerequisites

- Git
- Python 3.11 or newer
- Node.js/npm
- Ollama reachable at `http://127.0.0.1:11434` unless `OLLAMA_BASE_URL` is overridden
- `nomic-embed-text:latest` for hybrid search
- `gpt-oss:120b-cloud` for generated answers; it is not needed for evidence-only `/api/search`

`.env.example` is a secret-free reference. The application reads process environment variables directly; it does not silently load a local `.env` file. Defaults already match the submitted snapshot. To override them in PowerShell:

```powershell
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
$env:OLLAMA_CHAT_MODEL = "gpt-oss:120b-cloud"
```

Ensure Ollama is running and the models are available:

```powershell
ollama pull nomic-embed-text:latest
ollama pull gpt-oss:120b-cloud
```

### Clone → setup → run

```powershell
git clone https://github.com/abdoabozena7/Instant_creativa.git
cd Instant_creativa
PowerShell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1

.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. FastAPI serves the production React build and `/api/*`.

Verify a query reaches traceable evidence from another PowerShell window:

```powershell
$body = @{ query = "renal cancer visible haematuria age 45"; mode = "hybrid"; top_k = 3 } | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/search" -ContentType "application/json" -Body $body
$result.results | Select-Object rank, chunk_id, recommendation_id, source_version, citation
```

Expected rank 1 is current recommendation `1.6.6`, with stable chunk ID `ng12_1.6.6_c01` and page/source provenance.

### Rebuild missing artifacts

The normal clone already contains the reviewed artifacts. If only the dense index is missing, the same bootstrap command rebuilds it from the tracked chunks using Ollama. If the chunk corpus is also missing, supply both source PDFs explicitly:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 `
  -CurrentPdf "C:\path\to\ng12-current-2026.pdf" `
  -FullPdf "C:\path\to\ng12-full-2015.pdf"
```

`scripts/verify_runtime_artifacts.py` checks the chunk count, manifest hash, embedding model, shape, normalization, and a canonical smoke query. For UI development, run `npm run dev` inside `frontend/`; Vite proxies API calls to port 8000.

The app opens in **Presentation** mode: ten keyboard- and scroll-controlled scenes explain source authority, parsing, structured evidence, hybrid retrieval, safety gates, the blind baseline, observed failures, adjudication, and isolated experiments. Use Arrow keys, Page Up/Page Down, Space, or the chapter rail. The final scene opens the live retrieval console; the top navigation can jump directly to Retrieval or Evaluation at any time.

The corpus build copies the two inputs to `data/raw/` under stable names, parses the complete PDFs before filtering, and writes the five requested parsing outputs. The index builder embeds all 440 chunks through Ollama and writes a normalized NumPy matrix. With only 440 candidates, exact cosine search is fast and completely auditable, so a vector database would add complexity without a measured benefit.

## Retrieval and API

The retriever exposes three modes over the same corpus:

- BM25 for exact symptoms, thresholds, investigation names, and recommendation IDs.
- Dense exact-cosine retrieval using `nomic-embed-text:latest`.
- Hybrid retrieval using a measured 55% normalized BM25 / 45% dense score.

The final score also exposes deterministic adjustments for exact recommendation IDs, current canonical authority, evidence intent, explicit cancer-site matches, and site coherence inferred from a strong canonical result. These components are returned in `score_detail`; the safety policy is not hidden in a proprietary reranker.

FastAPI endpoints:

- `GET /api/health` — corpus, index, Ollama, and model readiness.
- `POST /api/search` — ranked evidence without generation.
- `POST /api/answer` — retrieval followed by citation-constrained `gpt-oss:120b-cloud` generation.
- `GET /api/metrics` — corpus, evaluation, reconciliation, and runtime telemetry.
- `GET /api/chunks/{chunk_id}` — direct stable-chunk inspection.
- `GET /docs` — interactive OpenAPI documentation.

Excluded-site-only questions are refused before embedding, retrieval, or model generation. Mixed queries search only the in-scope corpus and return a scope warning. Generated evidence labels are normalized and checked against the supplied context before the UI displays citation status.

## Output schemas

`pages_current.jsonl` and `pages_full.jsonl` contain one record for every physical PDF page, including blank pages, with source version/type, authority, physical page, and cleaned source text.

`records_clean.jsonl` is the normalized, pre-chunk corpus. Important fields include:

```json
{
  "record_id": "ng12_1.1.2",
  "document": "NICE NG12",
  "source_file": "ng12_current_2026.pdf",
  "source_version": "2026_current",
  "source_type": "current_guideline",
  "authority_priority": "primary",
  "page": 9,
  "page_end": 9,
  "section": "Lung cancer",
  "subsection": null,
  "cancer_sites": ["lung"],
  "content_type": "recommendation",
  "recommendation_id": "1.1.2",
  "text": "original NICE wording...",
  "canonical_recommendation": true,
  "related_recommendation_ids": [],
  "supporting_record_ids": ["..."],
  "retrieval_eligible": true,
  "conflict_status": null,
  "metadata": {}
}
```

`chunks.jsonl` adds stable chunk identity, chunk index/count, token count/encoding, and carries all citation and authority metadata forward. `merge_report.json` records source inspection, counts, deduplication, flagged differences, token distributions, examples, and known limitations.

## Measured result

The completed build produced:

- 101 current-guideline pages and 382 full-guideline pages.
- 445 scoped normalized records: 122 current and 323 supporting.
- 404 retrieval-eligible records; 41 historical recommendation copies are audit-only.
- 34 historical/current overlaps detected and 7 possible wording or action differences flagged.
- 440 chunks: 122 current and 318 supporting.
- 68 symptom-table chunks, 23 canonical recommendation chunks, 188 evidence-table chunks, 57 rationale chunks, 34 evidence chunks, plus shared and methodological content.
- `cl100k_base` chunk lengths: minimum 10, median 155, p90 531, p95 589, maximum 697, mean 230.16 tokens.

All 122 current records remain exact, unsplit logical units. Supporting records at or below 700 tokens remain intact. Longer supporting evidence tables split on extracted row/line boundaries; longer narrative evidence splits on sentence boundaries and is packed toward 550 tokens without overlap. These thresholds were chosen only after inspecting the actual record distribution: current recommendations maxed out well below the limit, while a small set of supporting tables and evidence passages were too large for a useful retrieval candidate.

## Validation

The automated suite currently has 85 passing tests. It checks page completeness and provenance, footer removal, exact scoped recommendation IDs, excluded sites, recommendation boundaries, multi-site symptom relationships, safety-netting and definitions, record/chunk identity, canonical authority, audit-only historical recommendations, reciprocal evidence links, token limits, output/report counts, embedding dimensions and normalization, scope and minimum-answerability guards, vague-assessment negatives, concrete-feature positive controls, BM25/hybrid regression behavior, citation normalization, primary-source ranking, FastAPI endpoints, metrics, production frontend serving, frozen-architecture integrity, blind/gold separation, balanced case coverage, preservation of historical failures, and isolated optional evaluation tooling.

Known limitations are explicit rather than hidden:

- The full guideline has no recommendation IDs, so historical boxes are mapped structurally and by text similarity.
- Its complex Word-style evidence tables do not become perfect semantic rows; titles and original line order are preserved so no medical content is rewritten.
- One native-text page is blank. Two additional cleaned pages contain only removable update/presentation furniture.
- Three historical recommendations do not have a safe one-to-one current match, mainly because the colorectal recommendations were substantially reorganized.
- The two PDFs complement one another structurally, but not without version differences. The current PDF provides the authoritative action; the older PDF provides context. Seven differences remain review flags and are never blended into current wording.

# Evaluation results

## Development retrieval benchmark

The original checked-in development set has 31 queries: 20 site recommendations, 3 shared-guidance questions, 2 symptom-table questions, 3 supporting-evidence questions, and 3 excluded-site questions. It was used while selecting the retrieval architecture, so its result must not be presented as independent validation:

- BM25: 96.4% Recall@1, 100% Recall@5, 98.2% MRR@10, 100% canonical top-1 accuracy.
- Dense: 85.7% Recall@1, 92.9% Recall@5, approximately 89% MRR@10, 84% canonical top-1 accuracy.
- Hybrid: 100% Recall@1, Recall@5, MRR@10, canonical top-1 accuracy, and excluded-site refusal accuracy.

Hybrid is therefore the frozen default. This is useful experimental evidence that combining lexical and dense retrieval helped this corpus; the 100% result is not the headline hackathon claim.

## Blind end-to-end baseline

Architecture files, the corpus, and the dense index were fingerprinted before the blind set was executed. The architecture SHA-256 is `c0abd7977cf0fb02a2e9664b0ae52ccd9cea32107f556fa2db038e16af4f82bd`. The 44 questions were stored without gold labels; the separate gold file was loaded only after every answer had been persisted. There are 11 cases for each of lung, colorectal, upper GI, and bladder/renal, including direct recommendations, age and threshold edges, multi-site symptoms, hard negatives, missing information, out-of-scope cases, and apparent 2015/2026 conflicts.

Deterministic results from the frozen run:

- Scope classification accuracy: 97.73%.
- Correct refusal rate: 80%; false refusal rate: 0%.
- Retrieval Recall@1: 75.68%; Recall@3 and Recall@5: 97.30%; MRR@6: 85.59% across 37 retrieval-scored cases.
- Current-guideline accuracy: 100% whenever the relevant current evidence was retrieved.
- Citation-label validity: 80%.
- End-to-end latency: 3.08 s P50 and 5.49 s P95.

The baseline exposed three concrete failure clusters:

- `BL-U10`: “gall bladder” was allowed because the scope guard also matched the in-scope substring “bladder”.
- `BL-L03`: the correct lung recommendation was rank 6 rather than inside top 5; generation still received it because the answer endpoint supplies six passages.
- Eight answers used non-canonical citation formatting such as Unicode spaces, bold labels, mixed brackets, or grouped labels. Those labels failed strict syntax validation even when evidence was present.

These findings are intentionally preserved as regression targets. The retriever, prompts, index, models, and scope guard were not changed after the freeze.

## Automated semantic adjudication

Deterministic metrics are never sent to an LLM. Scope accuracy, refusals, retrieval ranks, source priority, citation syntax, and latency remain code-scored. Semantic tasks are decomposed into claim support, citation entailment, overreach, completeness, and refusal quality.

The default `hackathon` mode uses one temperature-zero Gemini pass for each semantic task. A passing primary label is accepted; a primary failure is escalated to `gpt-oss:120b-cloud`. Any disagreement becomes `UNCERTAIN` and fails closed. `PARTIALLY_SUPPORTED` also fails. Claims with no citation deterministically fail citation entailment without an LLM call.

The prepared plan contains 334 LLM tasks and 27 deterministic missing-citation judgments. Hackathon mode requires 334 primary calls plus only the failed cases as second-judge calls, capped at 668. The retained `full` mode runs both judges three times for all tasks (2,004 calls) when deeper research-style stability analysis is justified. Both modes checkpoint every completed call and resume without repeating work. The Gemini key is read only from `GEMINI_API_KEY` or `GOOGLE_API_KEY`; secrets are never stored in project files or logs.

The reviewer-ready workbook remains available as the audit surface and optional human calibration set. It contains all 44 cases, 134 model-proposed claim units, cited passages, separate Reviewer 1/Reviewer 2 fields, adjudicated Final fields, and formula-driven metrics.

An optional same-model triage run using `gpt-oss:120b-cloud` flagged 24 cases for review and produced provisional signals: 90.65% citation accuracy, 79.10% claim support, 20.90% unsupported claims, and 81.82% behavior accuracy. These numbers are displayed as provisional only. The judge is not independent and made at least one known source-reading error, so they are neither clinical nor official evaluation results.

The frozen v1 evaluation artifacts remain preserved as the honest pre-fix baseline. V2 records the gall-bladder scope fix. During presentation rehearsal, two underspecified questions exposed a separate defect: the system inferred missing patient facts from retrieved eligibility criteria. The localized v4 fix adds a generic minimum-answerability gate for patient-specific decisions, strengthens the evidence-only prompt for partially specified questions, removes three measured non-clinical BM25 query terms, and normalizes two additional citation formats. It does not encode the screenshot questions or NG12 eligibility rules.

The full 44-case v4 regression run reports scope 100%, correct refusal 100%, false refusal 0%, Recall@1 75.68%, Recall@5 97.30%, MRR@6 86.04%, current-guideline accuracy 100%, and citation-label validity 100%. End-to-end latency was 3.57 s P50 and 9.40 s P95. Semantic judging and human review were not run. Because the same 44 cases had already been inspected, v4 is explicitly a versioned regression run, not a new independent blind evaluation.

Another rehearsal question—“A patient has some stomach issues, is this serious?”—showed that severity/uncertainty phrasing could bypass the decision-intent guard and let the model infer appetite loss. V6 generalizes the contract: patient-specific assessments containing only site, qualifier, or vague-description terms stop before retrieval/model; a concrete feature such as pain, vomiting, dysphagia, haematemesis, mass, or blood still proceeds. Twenty-one negative paraphrases and twelve positive controls protect the behavioral boundary; no NG12 eligibility table is encoded in the guard.

The full 44-case v6 regression run preserves scope 100%, correct refusal 100%, false refusal 0%, 37 retrieval-scored cases, Recall@1 75.68%, Recall@5 97.30%, MRR@6 86.04%, current-guideline accuracy 100%, and citation-label validity 100%. End-to-end latency was 4.19 s P50 and 9.47 s P95. Semantic judging and human review were not run; v6 is a regression run, not a new independent blind evaluation.

The dashboard and `/api/metrics` expose v6 as the current deterministic result. V1, v2, and v4 remain historical evidence and are never overwritten or relabeled. Optional semantic tooling is outside setup/runtime; future behavior changes require a new versioned freeze/run/report.

The Retrieval workspace presents ranked results as explicit clickable evidence rows. Selecting an `[E#]` citation or row opens the full source passage, current/supporting status, page-level provenance, stable chunk ID, and score explanation in one inspector; mobile selection scrolls to the same inspector. A reranker was not added: the independent development benchmark already gives hybrid 100% Recall@1 and MRR, so there is no measured headroom, while using the historical blind set to tune a reranker would invalidate its role as evaluation evidence.

The resulting metric must be described as **automated LLM-judged groundedness**, not clinical validation. The API and dashboard expose `multi_judge_report_v1.json` automatically when the complete report exists.

# Architecture status

Core architecture development is closed after the architecture and fresh-clone audits. No new feature, model, retrieval component, layer, or cleanup should be introduced before the instructor review.

Allowed changes from this point are only localized fixes for bugs reproduced during submission rehearsal or Instructor Defense rehearsal. Every such fix must retain the same regression discipline and create a new evaluation identity when it changes frozen architecture behavior.

The clean-clone path and live evidence flow have passed. Architecture development closed again after the measured v6 rehearsal fix; further changes still require a reproduced bug and a new versioned regression run.
