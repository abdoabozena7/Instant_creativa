# NG12 RAG Full Architecture Audit

Audit date: 17 August 2026  
Scope: repository source, generated schemas/artifacts, runtime, frontend integration, evaluation tooling, tests, dependencies, and submission structure.

## Executive verdict

The project has a strong, defensible hackathon architecture. Its best decisions are source-aware ingestion, structural exclusion of historical recommendations, retention of BM25, measured hybrid retrieval, exact in-memory cosine for a 440-chunk corpus, deterministic pre-generation scope checks, evidence-first prompting, and a correct separation between deterministic metrics and semantic judging.

The architecture is not “productionized,” and it should not be. The main complexity is concentrated in source-specific PDF extraction and optional evaluation tooling, where the source layout and experiment history justify most of it. The runtime remains small.

The audit found no P0 defect. Five clear P1 issues were small enough to fix safely: the gall-bladder overlap, fragile citation-label formatting, insufficient dense-index integrity validation, a non-reproducible submission package, and rehearsal-discovered inference from missing patient facts. The repository now tracks the reviewed 440-chunk corpus, matching dense index, evaluation evidence, source, tests, and documentation while excluding raw PDFs, secrets, dependencies, builds, and temporary outputs.

The project is ready for an instructor architecture review and clone-based submission. A clean-clone rehearsal outside the development workspace completed bootstrap, frontend build, API startup, search, evidence lookup, citation validation, and generated answering without hidden machine files; the current regression suite contains 63 tests.

## Audit method and evidence

The audit reconstructed behavior from imports and executable code before using README descriptions. It inspected all Python runtime/ingestion/evaluation modules, scripts, tests, frontend API/types and evidence rendering, dependency manifests, generated corpus/index/evaluation schemas, source artifact counts, and repository state.

Pre-change validation:

- 38/38 tests passed.
- React production build passed.
- Development retrieval reproduced: BM25 Recall@1 96.43%, Recall@5 100%, MRR@10 98.21%; dense 85.71%, 92.86%, 88.69%; hybrid 100%, 100%, 100%.
- Frozen blind v1 reproduced: scope 97.73%, correct refusal 80%, false refusal 0%, Recall@1 75.68%, Recall@5 97.30%, MRR@6 85.59%, current-guideline accuracy 100%, citation-label validity 80%.

Post-change validation:

- 63/63 tests pass after the rehearsal fix.
- React production build passes.
- Development retrieval rankings and metrics are unchanged; only nondeterministic latency samples moved.
- A full versioned v2 run over all 44 blind questions gives scope 100%, correct refusal 100%, and false refusal 0%.
- Re-normalizing stored v1 answer text gives 100% citation-label validity, but a full new v2 generation run produced two new non-canonical formats and measured 94.87%; this is reported rather than hidden.
- Full blind v2 confirms scope 100%, correct refusal 100%, false refusal 0%, unchanged Recall@1/5 and MRR, and 100% current-guideline accuracy. Frozen v1 artifacts remain unchanged and honestly labeled.
- Presentation rehearsal then exposed two underspecified questions that reached retrieval and let the generator treat evidence criteria as patient facts. The localized v4 fix uses a generic zero-feature decision gate, query-only filtering for three measured scaffold words, and stronger unknown-fact/modality/logic prompt contracts.
- The full v4 regression run reports scope 100%, correct refusal 100%, false refusal 0%, Recall@1 75.68%, Recall@5 97.30%, MRR@6 86.04%, current-guideline accuracy 100%, and citation-label validity 100%. It ran no semantic judge or human review and is not presented as a new independent blind set.

# End-to-End Data Flow Audit

## Concrete query trace

Question: “Should a 45-year-old with visible haematuria be referred for suspected renal cancer?”

| Step | Module/function | Input → output | State/ranking/safety effect |
|---|---|---|---|
| API validation | `api/main.py` / `AnswerRequest` | JSON → validated Pydantic request | Rejects short/oversized query and invalid mode/top-k |
| Scope | `scope_guard.assess_scope` | string → scope dictionary | Safety decision before embedding; returns selected renal site |
| Query preparation | `RetrievalEngine.search` | collapses whitespace; adds `search_query:` only for embedding | No clinical rewriting |
| Lexical signal | `BM25Index.scores` | query → 440 raw BM25 scores | Independent lexical ranking signal |
| Dense signal | `OllamaClient.embed`, NumPy matrix multiplication | query → normalized 768-vector → 440 cosine scores | Independent dense ranking signal; external model call |
| Fusion | `engine.search` | normalized BM25 + mapped cosine | Changes ranking using named 0.55/0.45 weights |
| Policy adjustment | `engine.search` | base score + deterministic boosts/penalties | Changes ranking for canonical authority, exact IDs, intent, methodology, explicit site, optional site coherence |
| Evidence selection | `engine.search` | sorted candidates → first `evidence_k=6` | Decides the only evidence generation can see |
| Evidence labeling | `generation.generate_grounded_answer` | ranked results → `[E1]…[E6]` blocks | Response-local IDs assigned before generation |
| Generation | `OllamaClient.chat` | system prompt + question + evidence → answer text | Semantic synthesis; residual hallucination point |
| Citation normalization | `normalize_citation_labels` | formatting variants → canonical `[E#]` | Deterministic formatting repair only |
| Citation validation | `validate_citation_labels` | canonical labels + evidence count → result | Fails invalid/missing labels; does not test entailment |
| API composition | `api/main.py` | generation + full retrieval → response dictionary | Adds safety note, total latency, telemetry |
| Frontend resolution | `api.ts`, `App.tsx` | response → answer/evidence panels | `[E#]` selects `results[#-1]`; inspector shows stable `chunk_id` |

Observed rank-1 object: chunk `ng12_1.6.6_c01`, record `ng12_1.6.6`, current 2026 primary recommendation, page 23. Base hybrid score `0.807042`; canonical boost `0.22`; explicit-site boost `0.03`; final `1.057042`.

## Directionality and state

The flow is directional: files → typed ingestion records → serialized chunks/index → immutable runtime engine → ranked evidence → generated text → validation → API/UI. No circular Python imports were found.

State enters at four places:

1. Offline files: parsed JSONL, embedding matrix, manifest, reports.
2. Process initialization: global `RetrievalEngine`, BM25 statistics, loaded matrix.
3. External runtime dependencies: Ollama query embedding and chat response.
4. Mutable demo telemetry: bounded in-memory deques/counters.

Ranking changes only in `src/retrieval/engine.py`: signal normalization/fusion, policy adjustment, optional site-coherence adjustment, sort/tie-break. The API and UI do not re-rank.

Safety decisions occur in `scope_guard.py` before query embedding, in ingestion through scope/canonical eligibility, in generation instructions, and in deterministic citation validation. These are different responsibilities rather than duplicate “safety layers.”

## Hidden coupling and provenance risks

- Search results, evidence, generated answers, and API responses are dictionaries with key conventions shared across Python and hand-written TypeScript interfaces. This is the largest implicit contract.
- Evidence labels are coupled to result order. That is acceptable per response, but only `chunk_id` is stable.
- Authority vocabulary (`2026_current`, `2015_full`, `primary`, `supporting`) is repeated in ingestion, retrieval, prompt, API, frontend, reports, and tests. The repetition is readable but requires synchronized edits.
- `authority_adjustment` includes non-authority intent and site-coherence effects. Values are exposed, but the name is broader than the contents.
- The UI contains static historical evaluation copy as well as artifact-driven numbers. It correctly describes frozen v1 today but must be kept explicitly historical after post-audit changes.
- Chunk serialization drops `heading_path`, record-level `metadata`, `conflict_status`, and original `source_text`. Runtime citation provenance remains intact through source/version/page/section/record/chunk IDs, but the evidence inspector cannot display every reconciliation diagnostic without looking up `records_clean.jsonl`.
- `api.main` constructs the engine at import. This is explicit fail-fast behavior, but it couples test/import success to local corpus presence.

No unnecessary request-to-component back-and-forth was found. Generation never calls retrieval, and validation never reconstructs provenance from answer prose.

# Ingestion Architecture Audit

## Stage assessment

| Stage | Assessment | Verdict |
|---|---|---|
| Raw current extraction | `pdf_parser.py` preserves every physical page and native text order | KEEP |
| Conservative cleaning | `cleaner.py` removes headers/footers and repairs layout markers without summarization | KEEP |
| Clinical/document scope filtering | `scope_filter.py` is separate from cleaning and uses headings/table cells | KEEP |
| Full-guideline extraction | Source-specific layout rules are isolated in `full_pdf_parser.py` | KEEP, despite file size |
| Current normalization | Explicit mapping into common schema | KEEP |
| Reconciliation | Separate module; links and flags without text merging | KEEP |
| Chunking | Runs only after reconciliation and respects logical units | KEEP |

The separation is correct. Reducing file count would blur distinct correctness claims.

## Canonical-source behavior

- `normalize_current_records()` assigns `2026_current`, `primary`, and canonical status to numbered current recommendations.
- Full-guideline records are always `2015_full`, `supporting`, and non-canonical.
- `_historical_record()` explicitly sets `retrieval_eligible=False`.
- `build_chunks()` skips every ineligible record. Tests assert all 41 historical recommendations are absent.
- Reconciliation links support records to current recommendation IDs and flags seven possible version differences without auto-resolving wording.

Historical recommendations therefore cannot accidentally become canonical at runtime. Supporting historical evidence can retrieve, which is intentional.

## Provenance and chunking

Recommendation IDs, physical page ranges, source files, source versions, record IDs, related recommendation IDs, and supporting record IDs survive through chunking. Current units are not split. Supporting content over 700 tokens is split using table-line or sentence/paragraph structure and packed toward 550 tokens without overlap. This is document-aware enough for the real corpus and avoids speculative clinical NLP.

Two ingestion concerns are P2, not immediate fixes:

- `ScopedRecord` contains several never-populated future-looking fields (`age_conditions`, `investigation`, `action_type`, `urgency`, and usually `symptoms`). They are omitted from normalized metadata when null, so they do not corrupt runtime data, but they make the stage schema wider than necessary.
- Page ranges and layout/font thresholds are stable source-layout constants rather than general configuration. A new NICE PDF layout requires parser review and test updates; silently “configuring” around that would be less safe.

# Data Model Audit

## Strengths

- Dataclasses with slots define clear page, scoped, normalized, and chunk stages.
- Stable identities distinguish page, record, chunk, recommendation, and response-local evidence rank.
- Authority and version semantics are unambiguous.
- Provenance fields needed by retrieval/UI are carried forward.
- Evaluation `case_id` consistently joins separated questions, gold, runs, packets, and diagnostics.

## Weaknesses

- Runtime objects after JSON loading are untyped dictionaries. A misspelled `source_version`, missing `authority_priority`, or frontend drift is detected only by tests/runtime access.
- `NormalizedRecord.text` and `source_text` duplicate one another for most records. The duplication supports audit semantics but has little current runtime value.
- `document`, `source_file`, `source_version`, `source_type`, and `authority_priority` are repeated in every chunk. This is deliberate denormalization for a tiny inspectable corpus; a relational model would add more complexity than it removes.
- Search-result and answer response schemas are not Pydantic response models. Adding a small response model could improve OpenAPI/test contracts, but converting all internal dictionaries is not justified before submission.
- Provisional claim decomposition is reused to construct independent judge tasks. Judge verdicts are independent; claim-boundary selection is not fully independent.

Recommendation: keep the four ingestion dataclasses. Do not introduce a domain-entity framework. If one schema cleanup is later allowed, type `EvidenceResult` and the public answer/search response—not the entire pipeline.

# Retrieval Architecture Audit

## Signal independence and normalization

BM25 and dense retrieval are genuinely independent: BM25 tokenizes retrieval text and computes corpus statistics; dense uses a separately built embedding matrix and query embedding. Dense does not consume BM25 ranks.

BM25 is max-normalized over allowed candidates. Dense cosine is linearly mapped from `[-1,1]` to `[0,1]`. Fusion is transparent and now uses named constants. This is understandable, though not statistically calibrated: the two normalized scales are comparable by construction, not by learned probability.

## Authority and current-guideline behavior

Canonical precedence is both:

- a build-time filter for historical recommendation copies; and
- a ranking policy for current recommendation/action questions.

This is appropriate. Filtering prevents unsafe historical action text; ranking keeps supporting evidence available. The adjustment total mixes semantic intent, authority, and coherence, so the architecture should describe it as a deterministic policy adjustment even though the current response field is `authority_adjustment`.

## Constants and top-k

- API search default: 8; frontend search: 8.
- API answer evidence default: 6; frontend answer: 6; blind run: 6.
- Engine clamps all searches to 1–20; API schemas enforce narrower answer 1–10.
- Hybrid weights and all boosts/penalties are module-level named constants after the audit.
- Reconciliation similarity thresholds remain unexplained numeric constants (`0.32`, `0.42`, `0.62`). They are offline diagnostics, not runtime ranking, but should be documented with examples if the parser is revised.

Top-k propagation is consistent for the live UI and blind run. The blind retrieval metric uses the returned six passages and reports MRR@6. The development script retrieves ten and reports MRR@10.

## Exact search verdict

Exact in-memory retrieval is the better hackathon design. The corpus is 440 chunks and the vector matrix is roughly 1.29 MB. Exact cosine is auditable, deterministic, trivially persisted, and much faster than generation. A vector database would solve no measured current problem. A reranker is likewise unjustified: blind Recall@5 is 97.3%, and the one miss was rank 6, already inside the generation evidence count.

# Safety / Scope Architecture Audit

The guard is in the correct layer and runs before expensive work. Token/phrase regexes are sufficient for the configured site-list decision; an LLM classifier is not justified by current evidence.

The known gall-bladder bug was real: `\bbladder\b` matched inside the phrase “gall bladder,” while the excluded pattern also matched, converting an excluded-only question into a mixed query. The fix removes excluded spans before testing selected patterns. A separately named bladder site survives, so true mixed questions still work.

Out-of-scope and unsupported evidence are correctly distinct:

- excluded-only → controlled scope refusal, no retrieval/generation;
- in-scope but metadata filters empty → no-evidence response / answer 404;
- evidence present but insufficient → model instructed to state insufficiency.

Remaining limitation: an unrelated question that contains neither a selected nor excluded site is treated as in scope. This permits symptom-only questions, which the corpus needs, but means the guard is not a general medical-domain classifier. Do not add an LLM. Expand deterministic hard-negative evaluation before changing behavior.

# Generation / Grounding Audit

The model receives only the selected result list. Every block has a preassigned ID, provenance citation, authority, content type, and text. Current and supporting sources are clearly distinguished in both metadata and system instructions.

The boundary is mostly clean:

- retrieval decides what evidence is supplied;
- generation synthesizes/explains;
- deterministic code handles scope, evidence IDs, formatting normalization, label-range validation, and response assembly;
- offline semantic evaluation handles support/entailment/overreach.

Residual risks are unavoidable model behaviors: unsupported inference, added certainty, synonym expansion, qualifier loss, and valid-but-wrong citation binding. The system prompt addresses these but cannot guarantee them. The API currently returns a malformed/empty answer with failed citation validation rather than inventing a deterministic medical answer, which is correct fail-closed behavior for grounding.

# Citation Architecture Audit

Desired flow is implemented:

`ranked EvidenceResult` → response-local `E#` assigned before prompt → model references ID → one deterministic normalizer/validator → UI uses the same result array → stable `chunk_id` and provenance displayed.

The previous implementation normalized only `【E1】`. Stored failures also contained zero-width characters, narrow no-break spaces, bold markers inside brackets, mixed closing brackets, and grouped `(E1; E6)` labels. These are now normalized centrally. Unknown labels remain visible and fail validation.

Citations are not reconstructed from section names or generated text. The human-readable citation is derived once from chunk metadata in the engine. The persistent link is `chunk_id`, not the displayed E-rank.

# Evaluation Architecture Audit

## Deterministic versus semantic

The split is clean:

- Deterministic: scope status, refusals, false refusals, Recall@K, MRR, current-source correctness, citation-label range/syntax, latency, artifact/freeze identity.
- Semantic: claim support, citation entailment, overreach, completeness, refusal quality.

The blind questions exclude gold fields; gold is loaded only after answers are persisted. That is a strong design for a hackathon.

## Judge architecture

The default `hackathon` mode matches the requested philosophy: one independent Gemini pass per semantic task; GPT-OSS only for failed primary tasks; disagreement becomes uncertain/fail-closed. Missing citations are scored deterministically. The exhaustive two-judge × three-pass path is isolated optional tooling and never enters runtime.

Over-complexity remains in the optional path: 334 tasks, task fingerprints, resumable checkpoints, retries, consensus logic, and full 2,004-call mode. It is acceptable only because it is explicitly optional. The core presentation should lead with deterministic blind metrics and lightweight judge limitations.

Architecture fingerprinting successfully detected the audit edits. Frozen v1 should remain historical. The current hard-coded v1 filenames make post-change versioning awkward; a future evaluation should write a new freeze/run/report version rather than overwrite v1.

# Backend Boundary Audit

FastAPI endpoints contain direct orchestration but little algorithmic logic. Retrieval ranking remains in `engine.py`; generation/citation logic remains in `generation.py`; Ollama HTTP details remain in `ollama_client.py`.

`/api/answer` does scope-aware retrieval, skips generation for refusal, handles no evidence, calls generation, updates telemetry, and composes the response. This is application logic inside the API module, but it is short and visible. A service layer would mostly move the same 50 lines and is not justified for this hackathon.

Global state is limited to an immutable engine and mutable demo telemetry. Initialization is explicit and now validates dense-index integrity. No dependency-injection framework should be added.

Error mapping is broad (`Exception` → 503), which prevents stack traces reaching the UI but can classify programmer defects as dependency failures. Narrower exception classes would be P2 polish; do not add a framework.

# Configuration Audit

## Genuine configuration

| Value | Current location | Assessment |
|---|---|---|
| Ollama URL, embed model, chat model | Environment variables in `OllamaClient` | Correct runtime configuration |
| Judge keys/models/URL | Environment in judge clients | Correct; no secret found in repository scan |
| Input PDFs/output directories | CLI arguments in build scripts | Correct build configuration |
| Retrieval mode, top-k, filters | API request / frontend request | Correct user/demo configuration |
| Evaluation mode/concurrency | Evaluation CLI | Correct optional-tool configuration |

## Stable domain/source constants

| Value | Why not global config |
|---|---|
| Seven allowed sites and excluded-site phrases | Defines evaluated product scope; changing it changes product behavior |
| `2026_current/primary` versus `2015_full/supporting` | Core authority invariant |
| Source page ranges/bookmark titles/font rules | Coupled to inspected PDFs; changes require parser review |
| Chunk target 550/max 700 and `cl100k_base` | Measured corpus-build decision; already parameterizable at function boundary |
| Prompt grounding rules | Versioned architecture behavior, not an operator toggle |
| BM25 0.55 / dense 0.45 and named policy constants | Frozen measured retrieval design; should change only through evaluation |

## Accidental or weakly documented constants

- Reconciliation similarity cutoffs `0.32`, `0.42`, `0.62` need evidence/examples.
- HTTP timeouts and retry counts are reasonable operational constants but scattered.
- Static frontend fallback numbers (`440`, `768d`, metric fallbacks) can drift from artifacts.
- Page 35 referral extraction and several page ranges are source constants but look like magic numbers without nearby named constants.

A giant configuration object would reduce clarity. Centralize only values changed by the same actor for the same reason.

# Dependency Audit

## Python

| Dependency | Use | Verdict |
|---|---|---|
| PyMuPDF | Native text, bookmarks, spans, positions | KEEP; core parser |
| pdfplumber | Ruled current symptom tables; report inspection | KEEP; narrow justified use |
| tiktoken | Deterministic token measurement/chunk limits | KEEP |
| NumPy | BM25 arrays, embeddings, exact cosine, validation | KEEP |
| httpx | Ollama and judge APIs, TestClient compatibility | KEEP |
| FastAPI/Pydantic | API and input validation | KEEP |
| uvicorn[standard] | Demo server | KEEP; `standard` extras are convenient but not essential |
| pytest | Test runner | KEEP; ideally separate dev requirements, P3 |

No overlapping RAG framework, vector database client, reranker, or unused ML framework exists.

## Frontend

React/ReactDOM, Vite/TypeScript, Framer Motion, and Lucide are all substantially used. The frontend dependency surface is appropriate for the existing presentation-heavy UI. No dependency replacement is warranted.

The workbook script imports environment-provided `@oai/artifact-tool`, which is not declared in `frontend/package.json`. Because workbook generation is optional and the artifact is checked in locally, classify this as optional-tool reproducibility debt, not a runtime dependency defect.

# Error Handling Audit

| Failure | Desired behavior | Actual / finding |
|---|---|---|
| PDF missing | Fail loudly before partial build | Correct |
| Native extraction/table invariant fails | Build diagnostics/tests fail | Correct for known source |
| Embedding model unavailable during query | Controlled 503; no semantic fallback | Correct |
| Dense files absent | Explicit BM25 fallback warning | Defensible demo degradation, not silent |
| Dense files stale/corrupt | Fail loudly | Fixed; hash/model/shape/finite/norm checks |
| Ollama chat unavailable | Controlled 503 | Correct |
| Empty retrieval after filters | Controlled, distinct from scope | Search warning; answer 404 |
| Malformed/empty model output | No invented answer; citation validation fails | Correct, though raw empty response still returns 200 |
| Invalid citation | Preserve text and fail validation | Correct |
| Unsupported question | State evidence insufficiency when evidence cannot answer | Prompt-driven; semantic quality remains evaluated |
| Invalid metadata | Build/tests or key access should fail | Mostly loud; runtime JSON is not schema-validated |
| Missing merge report on metrics endpoint | Metrics should degrade or controlled error | Currently unhandled 500; P2 because core answer path remains ready |

# Test Architecture Audit

Tests emphasize high-value invariants: complete pages, source provenance, exact recommendation selection, excluded recommendations, logical chunk integrity, historical exclusion, source links, embedding alignment/normalization, scope behavior, current ranking, citation mapping, API endpoints, blind/gold separation, deterministic baseline preservation, and judge fail-closed behavior.

Audit improvements added regression tests for:

- excluded phrase overlap and true mixed-site behavior;
- all observed citation formatting variants and invalid out-of-range labels;
- same-row-count stale dense index.

The frozen-v1 test was corrected to assert the honest historical invariant—stored run hash equals stored freeze hash—instead of requiring current code to remain forever identical to the historical architecture.

Remaining test gaps:

- unrelated-domain/no-site hard negatives;
- public response-schema validation between Python and TypeScript;
- generation contract for empty/malformed Ollama message;
- a single integration test tracing a result's `chunk_id` through answer citation to UI-facing provenance;
- index model-mismatch case (the runtime check exists; stale-hash behavior is tested).

Do not add tests for raw coverage count. Add only the integration contracts above when they protect a real change.

# Repository Structure Audit

The source folders mostly reveal the architecture: ingestion, retrieval/generation runtime, evaluation, API, frontend, scripts, tests, and data artifacts. Naming is understandable.

Findings:

- `src/retrieval` also owns generation, scope, and Ollama transport. Renaming it to `runtime` could be cleaner but would create broad import churn for little value. KEEP.
- `parse_ng12.py` and its `pages.jsonl/scoped_records.jsonl/parsing_report.json` are a documented earlier current-only pipeline. They duplicate part of `build_corpus.py` but remain useful parser diagnostics. OPTIONAL TOOLING; do not delete before submission unless the team no longer demonstrates parser phases.
- `output/`, `outputs/`, `.playwright-cli/`, frontend `*.tsbuildinfo`, compiled `vite.config.js/.d.ts`, and preview PNGs are local generated artifacts excluded by `.gitignore`; they are not submission dependencies.
- Evaluation artifacts live under `data/eval` and evaluation code under `src/evaluation/scripts`, separate from answer runtime. This boundary is good.
- The reviewed `data/parsed`, `data/index`, and `data/eval` snapshots are tracked. Raw PDFs remain local because they are unnecessary to run the submitted snapshot and may have redistribution constraints.
- `scripts/bootstrap.ps1` verifies shipped artifacts, rebuilds a missing dense index, or rebuilds the corpus when both source PDFs are supplied explicitly. README documents clone → setup → run → query → evidence.

# Over-Engineering Audit

| Component | Classification | Value / complexity / hackathon judgment |
|---|---|---|
| Full two-judge × three-pass evaluation | OPTIONAL TOOLING | Research-grade cost; useful only for stability analysis, not core proof |
| Lightweight primary + conditional second judge | KEEP | Matches semantic need and hackathon budget |
| Architecture fingerprinting | KEEP | Prevented post-hoc relabeling; filenames/version flow should be simplified later |
| Resumable judge checkpoints/retries | KEEP in optional tooling | Necessary once hundreds of external calls exist; irrelevant to runtime |
| Three retrieval modes | KEEP | Enables measured comparison and an evidence-only demo; default remains hybrid |
| Dense embedding index | KEEP | Measured complementary signal; tiny artifact |
| Exact cosine implementation | KEEP | Simpler than a vector database at 440 chunks |
| Evidence inspector endpoint/UI | KEEP | Materially improves provenance and instructor explainability |
| Extensive stage metadata | SIMPLIFY later | Core provenance is valuable; several null future fields can be removed |
| Reconciliation similarity and structural maps | KEEP | Real two-version problem; isolated offline and auditable |
| In-memory telemetry | KEEP | Small demo feedback; do not evolve into observability infrastructure |
| Configuration abstraction | KEEP current minimal approach | No global framework needed |
| Additional API/service/DI layers | REMOVE as proposal | Would move code without solving a current problem |
| Current/full parser reports | KEEP | Evidence that parsing choices and losses were inspected |

# Under-Engineering Audit

| Area | Fragility | Status |
|---|---|---|
| Substring/overlap site matching | “gall bladder” also selected bladder | FIXED with excluded-span-first phrase matching |
| Citation parsing | Unicode spaces, zero-width chars, bold/group/mixed brackets failed | FIXED with one normalizer and validator |
| Dense index/corpus contract | Same row count could conceal stale ordering/model/corruption | FIXED with manifest/hash/model/vector validation |
| Runtime data contracts | Dictionaries link retrieval, generation, API, and UI | REMAINS P2; type public evidence/response if changed |
| Scope semantics | No-site unrelated query is allowed | REMAINS P2; benchmark first, do not add an LLM reflexively |
| Policy score naming | `authority_adjustment` includes intent/coherence | REMAINS P2 clarity debt |
| Evaluation versioning | A fix must not overwrite its historical baseline | FIXED with explicit versioned freeze/run/report output and separate v2 artifacts |
| Artifact packaging | Reviewed corpus/index/evaluation snapshot is tracked; raw PDFs and disposable outputs are excluded | FIXED and clean-clone verified |
| Public metrics artifact errors | Required merge report read is not guarded | REMAINS P2 demo resilience issue |

# Architecture Scores

| Dimension | Score / 10 | Reason |
|---|---:|---|
| Simplicity | 8.4 | Small runtime, direct orchestration, no speculative infrastructure; source parsers/evaluation are necessarily larger |
| Separation of concerns | 8.3 | Clear ingestion stages and runtime/evaluation separation; API still owns short application orchestration |
| Modularity | 8.2 | Components are independently testable; dictionary contracts and broad `retrieval` package reduce the score slightly |
| Data provenance | 9.1 | Physical pages, source/version/authority, IDs, links, and chunks are preserved; some record-only diagnostics drop at chunking |
| Retrieval design | 8.8 | BM25 retained, dense measured, transparent hybrid/policy, exact search appropriate, index integrity validated |
| Safety architecture | 7.8 | Deterministic pre-generation guard and historical exclusion are strong; generic unrelated/no-site intent is not classified |
| Grounding architecture | 8.2 | Evidence-only prompt, preassigned IDs, deterministic validation; model overreach/entailment remain residual risks |
| Evaluation design | 7.8 | Strong blind/deterministic split and lightweight default; optional evaluator is complex and claim decomposition is not independent |
| Testability | 8.7 | High-value invariants and integration tests; public dictionary schema/no-site negatives remain gaps |
| Explainability | 8.8 | Score details, evidence inspector, source authority, reports, and measured alternatives are visible |
| Hackathon suitability | 9.2 | Technically strong without vector DB/reranker/framework creep; the small evaluated runtime snapshot is self-contained |

# Prioritized Findings

## P0 — could invalidate results / unsafe / broken architecture

None found.

## P1 — important before instructor review/submission

### P1-1 — Excluded phrase could also satisfy included phrase (implemented)

- **Problem:** “gall bladder” matched both excluded gall-bladder and included bladder patterns.
- **Why it matters:** The guard allowed an excluded-only clinical question to reach retrieval/generation; frozen correct refusal was 80%.
- **Smallest correct fix:** Remove all explicitly excluded spans before running selected-site patterns; preserve separately named included sites.
- **Files affected:** `src/retrieval/scope_guard.py`, `tests/test_retrieval.py`.
- **Expected impact:** Full blind v2: scope 97.73% → 100%; correct refusal 80% → 100%; false refusal remains 0%.
- **Regression risk:** Low; true mixed bladder/gall-bladder behavior is explicitly tested.

### P1-2 — Citation formatting variants failed deterministic binding (implemented)

- **Problem:** Only CJK corner brackets were normalized; Unicode whitespace, zero-width chars, bold labels, mixed brackets, and grouped labels failed.
- **Why it matters:** UI citation resolution and syntax metric understated valid evidence references; frozen label validity was 80%.
- **Smallest correct fix:** One deterministic group normalizer followed by one range validator; never reconstruct citations from prose.
- **Files affected:** `src/retrieval/generation.py`, `tests/test_retrieval.py`.
- **Expected impact:** Stored-v1 replay reached 100%; fresh v2 generation measured 94.87% because two newly observed formats were non-canonical. Entailment is unchanged and not claimed.
- **Regression risk:** Low; canonical labels remain unchanged and out-of-range labels still fail.

### P1-3 — Dense index could be stale with matching row count (implemented)

- **Problem:** Runtime validated only embedding row count.
- **Why it matters:** Reordered/edited chunks, model mismatch, NaNs, or non-normalized vectors could silently invalidate dense/hybrid rankings.
- **Smallest correct fix:** Validate the existing manifest's chunk SHA/model/rows/dimensions and vector finiteness/norm at startup.
- **Files affected:** `src/retrieval/engine.py`, `tests/test_retrieval.py`.
- **Expected impact:** No metric change; corrupted/stale configurations now fail loudly.
- **Regression risk:** Low; valid current artifacts pass, and changes now require the already-documented rebuild step.

### P1-4 — Repository was not a reproducible Git submission (implemented)

- **Problem:** Source and runtime artifacts were previously untracked/ignored, so a clone could not start without hidden local files.
- **Why it matters:** A judge must be able to reproduce the reviewed demo rather than reconstruct an undocumented corpus from unavailable PDFs.
- **Smallest correct fix:** Track the small reviewed corpus/index/evaluation snapshot; keep raw PDFs, secrets, dependencies, builds, and disposable outputs ignored; add a secret-free environment reference, verifier, bootstrap command, and README workflow.
- **Files affected:** Git index, `.gitignore`, `.gitattributes`, `.env.example`, `README.md`, `scripts/bootstrap.ps1`, `scripts/verify_runtime_artifacts.py`, runtime/evaluation artifacts.
- **Expected impact:** No model metric change; clean clone now reaches setup → run → query → evidence and preserves the evaluated snapshot.
- **Regression risk:** Low after rehearsal. Checkout line endings are pinned for every byte-hashed runtime file so Windows/Linux conversion cannot invalidate chunk or evaluation fingerprints; tests no longer depend on an ignored local workbook.

### P1-5 — Missing patient facts were inferred from retrieved criteria (implemented after rehearsal)

- **Problem:** Generic questions such as “Should this patient be referred for suspected cancer?” reached retrieval with no clinical facts. A plausible top passage then encouraged the model to turn eligibility conditions into asserted patient facts. Separately, BM25 query scaffolding in “What about the diabetes?” could outrank the diabetes recommendation with generic patient-support text.
- **Why it matters:** The answer could sound clinically decisive while being unsupported by the question, and the visible evidence trace could be topically wrong.
- **Smallest correct fix:** Stop only patient-specific decision requests containing zero clinical features before retrieval/model; keep partial questions for grounded generation; remove only the three measured BM25 scaffold terms; explicitly preserve unknown patient facts, recommendation modality, and AND/OR logic; normalize two observed citation variants.
- **Files affected:** `src/retrieval/scope_guard.py`, `src/retrieval/bm25.py`, `src/retrieval/engine.py`, `src/retrieval/generation.py`, `api/main.py`, and regression tests.
- **Expected impact:** Generic zero-feature decisions return controlled insufficient information with no evidence or model call; “What about the diabetes?” ranks the diabetes criterion first; partial and complete clinical questions remain answerable.
- **Regression risk:** Low but non-zero. Nine paraphrased negatives and five positive controls protect against phrase memorization; the full v4 run preserved scope, Recall@1/5, and current-guideline accuracy, while MRR moved 85.59% → 86.04% and citation-label validity 94.87% → 100%.

## P2 — worthwhile if time allows

- Add unrelated-domain/no-site hard-negative cases before changing guard semantics.
- Add small typed public response/evidence models if API/frontend contracts change.
- Rename/split policy score details so authority, intent, and coherence are not aggregated under one label.
- Version future freeze/run/report filenames instead of overwriting v1.
- Document reconciliation thresholds with concrete matched/unmatched examples.
- Make `/api/metrics` degrade cleanly when optional report files are missing/malformed.
- Remove never-populated scoped-record fields in a separately measured corpus rebuild.
- Add a provenance integration test from generated E-label through `chunk_id` and source page.

## P3 — polish only

- Separate pytest from runtime requirements.
- Ignore or clean frontend build metadata and compiled Vite config artifacts.
- Consolidate `output/` and `outputs/` naming.
- Format long lines and add linting only if the team already uses it; do not introduce a framework for submission.

# Final Executive Summary

**A. Already strong:** source-aware staged ingestion, explicit 2026 authority, audit-only historical recommendations, excellent provenance, measured BM25/dense/hybrid design, exact search appropriate to corpus size, deterministic guard before model calls, evidence-only prompting, and honest deterministic/semantic evaluation separation.

**B. Unnecessarily complex:** only the optional full 2×3 semantic evaluator and its checkpoints/fingerprints are research-grade. Keep them out of the core story; the default lightweight mode is appropriate.

**C. Too fragile/simple:** the phrase overlap, citation parser, index integrity checks, submission contract, and handling of underspecified patient decisions were too fragile and are fixed. Remaining P2 simplicity debt is dictionary-based runtime contracts and generic unrelated-domain behavior.

**D. Actually fixed:** excluded-span-first scope matching; a generic minimum-answerability gate; unknown-fact/modality/AND-OR generation contracts; minimal measured BM25 query filtering; centralized citation normalization/range validation; dense-index manifest/hash/model/vector validation; defect-specific tests; reproducible tracked runtime artifacts.

**E. Deliberately not changed:** UI design, clinical scope, corpus, chunk strategy, embedding/chat models, retrieval weights/strategy, top-k, vector storage, reranking, frameworks, agents, and optional evaluator architecture.

**F. Metric regression:** none in the full v4 regression run. Scope/refusal remained 100%/100% with 0% false refusal; Recall@1/5 remained 75.68%/97.30%; current-guideline accuracy remained 100%; MRR moved 85.59% → 86.04%; citation-label validity moved 94.87% → 100%. End-to-end latency was 3.57/9.40 s P50/P95. No semantic or clinical improvement is claimed because no judge or human review ran.

**G. Remaining P0/P1:** none. Later architecture changes are frozen unless a rehearsal exposes a concrete bug.

**H. Review readiness:** yes. The architecture and the submission path have both been exercised from a clean clone outside the development workspace.
