# NG12 Architecture Decisions

These are compact architecture decision records for the implemented system. Measurements distinguish the 31-query development set from the frozen 44-case blind run.

## Decision record

| Decision | Why we chose it | Alternative considered | Evidence / measurement | Tradeoff |
|---|---|---|---|---|
| Narrow NG12 scope: lung, colorectal, oesophageal, stomach, pancreatic, bladder, renal | A bounded evidence surface is explainable and can be evaluated honestly during a hackathon | Parse every cancer site or provide generic clinical QA | Corpus tests assert the exact selected recommendation IDs and exclude other sections; blind scope baseline was 97.73%, with the known phrase defect fixed after audit | The scope guard is a configured phrase guard, not general clinical intent classification |
| Two-source reconciliation | The current PDF has concise recommendations; the full guideline supplies rationale, evidence, and methodology | Use only the current PDF; blend both sources indiscriminately | 445 normalized records; 34 overlaps and 7 possible version differences are explicitly reported | Reconciliation has source-specific structural rules and requires audit when source layouts change |
| 2026 canonical precedence | Current actions, thresholds, and urgency must not be overridden by historical wording | Let semantic similarity decide authority | 41 historical recommendation copies remain audit-only and are absent from 440 chunks; blind current-guideline accuracy was 100% | Supporting 2015 context must always retain visible authority metadata |
| Raw extraction separated from cleaning and scope selection | Each transformation has a different correctness claim and can be tested independently | One compact parser that emits final chunks | Physical page tests, cleaning tests, scoped-ID tests, and corpus invariants fail at distinct stages | More modules than a single script, but each boundary is real and source-driven |
| PyMuPDF plus targeted pdfplumber use | Native text and bookmarks handle prose; ruled symptom tables need cell geometry | OCR; a general document framework; PyMuPDF alone | Current: native text on 101/101 pages. Full: native text on 381/382 pages. No malformed current symptom-table rows in build diagnostics | Two PDF libraries, but no OCR error/cost and each library has a narrow role |
| Section-aware chunking | Recommendation IDs, table rows, definitions, and source sections carry clinical meaning and provenance | Fixed-size token windows with overlap | All 122 current units remain unchanged; supporting chunks max at 697 tokens; 440 chunks total | Supporting sentence/table packing is source-specific and not a universal chunker |
| Keep BM25 | Exact recommendation IDs, thresholds, symptom words, and investigation names are central in NG12 | Dense-only retrieval | Development BM25: 96.43% Recall@1, 100% Recall@5, 98.21% MRR@10 | Lexical matching can miss paraphrases |
| Keep dense retrieval as an independent signal | Paraphrases and clinical intent can differ lexically even in a small corpus | BM25-only | Development dense alone was weaker (85.71% Recall@1, 92.86% Recall@5), but it contributed to the selected hybrid | Adds query-embedding dependency and roughly 45–75 ms local measured retrieval latency |
| Weighted hybrid default: 0.55 BM25 / 0.45 dense | Combine measured complementary signals without a hidden reranker | Dense-only, BM25-only, reciprocal-rank fusion | Development hybrid achieved 100% Recall@1/5 and MRR; v8 Recall@5 was 97.30% and MRR@6 86.04% | Weights and policy boosts were selected on development data and should not be presented as universal constants |
| Exact cosine over NumPy instead of a vector database | Only 440 normalized vectors; row-level scores are easy to inspect | FAISS or a hosted/local vector database | Exact search is a single matrix multiplication; development hybrid P50 remained well below generation latency | Linear search would become the first retrieval infrastructure limit if the corpus grew substantially |
| No reranker in the submitted runtime | A second ranking model must earn its latency/dependency cost on an untouched development target | Cross-encoder or LLM reranking of the hybrid candidate set | Development hybrid already has 100% Recall@1 and MRR, leaving no measurable headroom; the historical blind set is not a tuning set | Blind Recall@1 remains 75.68%; a reranker becomes defensible only with a new held-out development set showing repeatable gain |
| Deterministic scope guard before retrieval | Known site boundaries are explicit, cheap, testable, and do not require sending clinical text to another model | LLM scope classifier | Blind v1: 97.73% scope, 80% correct refusal, 0% false refusal. Full post-fix v2: 100%, 100%, 0% respectively | Phrase rules do not perform general clinical reasoning and require representative tests |
| Minimum-answerability gate before retrieval | A patient-specific decision or severity assessment with no concrete symptom cannot be grounded; retrieving a plausible criterion encourages the model to invent patient context | Encode every NG12 criterion; use an LLM classifier; let the generator guess | Tests cover 21 vague/empty paraphrases and 12 concrete-feature controls; v6 preserved all retrieval/scope/current-source metrics | The gate distinguishes concrete from vague input using a small transparent vocabulary, not a clinical ontology. Partial cases still rely on the model and remain fallible |
| Deterministic instruction-safety guard before clinical guards | Explicit control-plane instructions are not clinical questions and should never reach embedding, retrieval, or generation | Trust the prompt alone; use an LLM injection classifier; hide evidence only in the UI | 30 English/Arabic/Unicode attack variants blocked; 11 benign controls allowed; live API returns `model=null`, zero evidence; v8 preserved all 44-case deterministic metrics | Pattern rules cover explicit override, exfiltration, evidence/provenance and authority manipulation, not every possible adversarial paraphrase |
| Deterministic authority adjustments after signal fusion | Current recommendations should win action questions; evidence-intent questions should still retrieve rationale | Rely on embedding relevance alone; use a learned reranker | Development canonical top-1: hybrid 100%, dense 84%; blind current-guideline accuracy 100% | Current `authority_adjustment` total also contains intent/site-coherence policy, which is transparent but semantically broad |
| Grounded generation receives only selected evidence | Retrieval owns evidence selection; generation owns explanation/synthesis | Let the LLM search or answer from parametric memory | Prompt includes six labeled passages, source authority, unknown-fact/modality/AND-OR rules; generation is skipped for refusal and zero-feature decisions | Prompt constraints reduce but cannot eliminate overreach or wrong citation binding |
| Assign response-local evidence IDs before generation | The model can reference exact supplied blocks, while `chunk_id` preserves durable provenance | Reconstruct citations from generated prose | UI maps `[E#]` directly to the same ranked result; stable source identity remains the chunk ID | Reordering the result list changes E-labels, so ranks are not persistent identifiers |
| Deterministic citation normalization and validation | Formatting variants are syntax problems, not semantic reasoning problems | Ask the LLM to repair its own citations; accept any citation-looking prose | V1 validity was 80%, v2 was 94.87%, and v6 was 100% after also covering bracketed `Evidence E#` labels | Label validity proves only that an ID exists, not that the passage entails the claim |
| Withhold generation that fails the citation contract | Showing an uncited refusal or invalid answer beside retrieved passages falsely implies those passages support it | Display a warning badge; let the UI infer refusal text | A fake uncited model response is rejected deterministically; v8 citation-label validity remains 100% | This proves label binding only, not semantic entailment; valid-but-wrong claims still require semantic evaluation |
| Deterministic evaluation where possible | Scope, ranks, authority, label range, and latency have objective code-level answers | Send every metric to an LLM judge | 44 blind cases are independently stored from gold; deterministic metrics reproduce exactly | Gold design still determines what “relevant” means and must be reviewed |
| Lightweight LLM judge by default | Claim support, entailment, completeness, and overreach require semantic interpretation | Full human review only; two judges × three passes for every task | Default: one independent Gemini pass and GPT-OSS only for failed primary cases; missing citations fail deterministically | Automated judging is not clinical validation; its claim decomposition currently originates from same-model provisional tooling |
| Keep exhaustive multi-judge mode as optional tooling | It can study stability without burdening the demo/runtime | Delete all research-grade evaluation code | Full mode is isolated under `src/evaluation` and scripts, never imported by answering runtime | 2,004 calls and checkpoint logic are overkill for the core hackathon story |
| FastAPI as composition root with direct orchestration | The request sequence is short and readable | Dependency-injection framework, repositories, service containers | Endpoints call one retrieval engine and one generation function; 130 tests cover high-value invariants and end-to-end behavior | Output contracts remain dictionary/TypeScript conventions rather than shared generated schemas |
| Fail loudly on stale dense index | Same-row-count misalignment can silently invalidate rankings | Check only vector row count | Manifest already records chunk SHA, model, rows, dimensions; runtime now validates all plus normalization | Index/corpus changes require an explicit rebuild, which is desirable for a judged demo |

## Instructor Defense

### 1. Why hybrid instead of dense-only?

Dense was tested, not assumed superior. On the 31-query development set it achieved 85.71% Recall@1, 92.86% Recall@5, and 84% canonical top-1 accuracy. BM25 achieved 96.43%, 100%, and 100%; the 55/45 hybrid achieved 100% on those measures. NG12 contains exact thresholds, test names, and recommendation IDs where lexical evidence is unusually valuable, while dense retrieval still helps paraphrases.

**Likely follow-up: Why no reranker?** The development target already has 100% hybrid Recall@1 and MRR, so a reranker cannot demonstrate improvement there. Tuning on the 44-case historical blind set would be methodologically wrong. I would add one only after a new held-out development set shows a repeatable ranking gain that justifies its latency and dependency.

### 2. Why no vector database?

There are 440 vectors of 768 floats—about 1.29 MB. Exact normalized matrix multiplication is fast, deterministic, and exposes every score. A vector database would introduce serialization, indexing, service, and filtering complexity without improving the measured hackathon bottleneck, which is generation rather than cosine search.

### 3. Why keep the 2015 full guideline?

The 2026 current guideline gives authoritative actions but not all explanatory evidence, limitations, and committee rationale. The 2015 full document supplies that supporting context. It is valuable for “why/evidence” questions, provided it is never allowed to replace a current action.

### 4. How do you prevent historical recommendations from overriding 2026?

The prevention occurs structurally, not only in the prompt. Historical recommendation records are marked `canonical_recommendation=false` and `retrieval_eligible=false`; the chunker skips them, so they cannot enter retrieval or generation. Other 2015 evidence remains `supporting`, while current recommendations receive explicit authority policy and prompt labeling. Tests assert that all 41 historical recommendation records are absent from chunks.

### 5. Why deterministic scope guard rather than an LLM?

The configured boundary is a finite list of included and excluded sites. Regex phrase rules are cheaper, faster, reproducible, and run before embedding/generation. The blind set showed 0% false refusal; the one “gall bladder” overlap was a deterministic phrase bug and was fixed deterministically. An LLM would add latency and nondeterminism without evidence that semantic classification is needed for this boundary.

### 6. What exactly does section-aware chunking preserve?

Every current recommendation, symptom-table row, definition, and shared-guidance unit remains a single unchanged chunk. Supporting content is split only when oversized, using table-line/row boundaries for evidence tables and sentence/paragraph boundaries for narrative evidence and rationale. Each chunk retains source file/version, authority, page range, section, cancer sites, recommendation links, record ID, and stable chunk ID.

### 7. Where can hallucination still happen?

After retrieval. The model may over-generalize, add a synonym or workflow step, combine qualifiers incorrectly, or attach a valid evidence label to a claim the passage does not entail. The prompt and citation validator constrain this but do not prove semantics. That is why claim support, citation entailment, and overreach are evaluated separately offline.

### 8. Why is blind Recall@1 lower than Recall@5?

The corpus intentionally includes both canonical recommendations and closely related table/evidence/rationale chunks. Policy boosts help current actions, but paraphrased or ambiguity-heavy questions can place a related passage above the exact gold unit. The correct unit was within the first five in 36 of 37 scored blind cases, so evidence-set recall is strong even when first-rank precision is lower. Generation receives six passages.

### 9. What does the LLM judge prove, and what does it not prove?

It estimates semantic groundedness against the supplied evidence under a documented rubric: claim support, citation entailment, overreach, completeness, and refusal quality. It does not prove medical correctness, validate NICE guidance, establish patient safety, remove judge bias, or replace human adjudication. Deterministic retrieval/scope metrics never depend on it.

### 10. Which component changes first if the corpus grows 100×?

The exact dense scan and in-memory BM25 construction become the first infrastructure candidates for replacement or sharding after measurement. The authority model, provenance schema, scope gate, evidence IDs, grounded prompt, and citation validator should remain. A vector index would be justified then by measured latency/memory, not added pre-emptively now.
