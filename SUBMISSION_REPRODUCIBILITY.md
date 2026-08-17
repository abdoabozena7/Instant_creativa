# Submission and Reproducibility Audit

Audit date: 17 August 2026  
Result: **PASS**

## Submission decision

The submission ships the exact reviewed runtime snapshot: 440 structured chunks, their provenance/merge records, the matching 768-dimensional dense index and manifest, and versioned v1/v2 evaluation evidence. Together these artifacts are small enough for Git and remove any dependency on hidden source PDFs or a machine-specific corpus build during judging.

The raw NICE PDFs are not required to run the snapshot and are not tracked. They may be supplied locally to rebuild the corpus when redistribution rules permit. A rebuild is a new corpus snapshot; it must be re-evaluated rather than relabeled as either versioned blind result.

| Category | Tracked | Generated/local |
|---|---|---|
| Application | `api/`, `src/`, `frontend/src`, package manifests | `.venv/`, `node_modules/`, `frontend/dist/`, TypeScript build info |
| Runtime data | `data/parsed/`, `data/index/` | `data/raw/*.pdf` |
| Evaluation | `data/eval/`, evaluation code/scripts | workbook previews, judge scratch outputs under `output/` or `outputs/` |
| Configuration | `.env.example`, configuration defaults in code | `.env`, process secrets and machine overrides |
| Review | README, architecture documents, tests | `.playwright-cli/`, `tmp/`, caches and screenshots |

## Environment contract

`.env.example` documents the supported values without credentials:

- Ollama URL: `http://127.0.0.1:11434`
- Embedding model: `nomic-embed-text:latest`
- Generation model: `gpt-oss:120b-cloud`
- Primary semantic-judge model: `gemini-3.6-flash`
- Conditional secondary judge: `gpt-oss:120b-cloud`

The demo runtime requires no API secret. Semantic evaluation may require a judge-provider key supplied only in the process environment. The application reads environment variables directly and does not silently load `.env`.

## One-command setup

From the repository root on Windows PowerShell:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

The command creates the Python environment, installs dependencies, verifies the shipped corpus/index contract, builds the frontend, and runs the tests. If only the dense index is absent it rebuilds it from tracked chunks. If the chunk corpus is absent, pass both source PDF paths as documented in README.

## Clean-clone rehearsal

The repository was cloned with `git clone --no-local` into a new directory under the Windows temporary folder, outside the development workspace. Before setup, the clone contained no raw PDFs, `.env`, virtual environment, `node_modules`, frontend build, output directory, or other ignored local state.

The README-only path then completed:

1. Bootstrap and artifact integrity verification.
2. Runtime load of exactly 440 chunks and the matching dense index.
3. React production build.
4. 41 passing tests.
5. FastAPI startup and ready health response.
6. Frontend HTTP response.
7. Hybrid search for `renal cancer visible haematuria age 45`.
8. Rank-1 evidence `ng12_1.6.6_c01`, recommendation `1.6.6`, source `2026_current`, physical page 23.
9. Evidence lookup resolving the same stable chunk and provenance.
10. Grounded answer generation with valid deterministic citation binding.
11. Clean tracked working tree after the rehearsal.

## Bugs exposed by rehearsal

- Windows converted JSONL line endings to CRLF, which correctly caused the dense-index manifest hash check to fail. `.gitattributes` now enforces LF for JSONL so committed chunk bytes remain identical across platforms.
- PowerShell did not automatically stop on a failed native verifier. The bootstrap now checks every native exit code and fails immediately.
- One test depended on a generated, ignored adjudication workbook. It now verifies the tracked workbook inputs and builder instead, keeping the generated workbook optional.

These were reproducibility bugs, not architecture or metric optimizations.

## Closure rule

Architecture development is closed after this successful rehearsal. Core refactoring, retrieval tuning, dependency changes, and speculative cleanup stop here. Further changes require a concrete bug observed during instructor rehearsal or the live demo, followed by proportionate regression testing.
