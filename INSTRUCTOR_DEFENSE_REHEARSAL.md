# Instructor Defense Rehearsal

Use each first paragraph as a 20–40 second spoken answer. The follow-up is the likely challenge to rehearse next.

## 1. Why hybrid retrieval when BM25 is already strong?

BM25 is strong here because NG12 contains exact symptom terms, ages, thresholds, investigations, and recommendation IDs. We kept it and measured alternatives: development Recall@1 was 96.43% for BM25, 85.71% for dense, and 100% for the selected 55/45 hybrid. Dense is not treated as superior; it stays because it adds a different paraphrase signal and the combined system performed best on development while blind Recall@5 remained 97.30%.

**Likely follow-up:** Are the weights universal? No. They are transparent, configurable values chosen on this corpus's development set; the blind set is reported separately and was not used to manufacture a better number.

**Likely follow-up:** Why not add a reranker? The development hybrid already has 100% Recall@1 and MRR, so there is no measurable headroom on the legitimate tuning set. Using the historical blind cases to tune a reranker would weaken the evaluation. A reranker is justified only after a new held-out development set shows repeatable improvement worth the extra model and latency.

## 2. Why did you not use a vector database?

The corpus has only 440 normalized vectors of 768 floats, roughly 1.29 MB. Exact cosine is one inspectable NumPy matrix multiplication and returns the true score for every chunk. A vector database would add a service, index lifecycle, serialization, and filtering complexity without solving the actual latency bottleneck, which is model generation.

**Likely follow-up:** What changes at 100× scale? Measure memory and query latency first; an approximate/vector index becomes justified when exact scan violates a target, while provenance and evidence contracts can remain unchanged.

## 3. Why retain the 2015 full guideline at all?

The 2026 document is canonical for current actions, but the full 2015 guideline contains rationale, evidence tables, methodology, limitations, and committee context that can answer “why” questions. We therefore reconcile both sources and keep 2015 only as visibly labeled supporting context.

**Likely follow-up:** Isn't that clinically risky? Historical recommendation boxes are structurally marked non-canonical and non-retrievable; 41 such records are retained for audit but never emitted as chunks.

## 4. How do you prove 2015 cannot override 2026?

The guarantee is made before ranking and prompting. Historical recommendation records have `canonical_recommendation=false` and `retrieval_eligible=false`, and the chunker skips them. Tests assert their absence from all 440 chunks. Remaining 2015 evidence is labeled `supporting`; current recommendations are labeled `current` and receive explicit, inspectable policy priority.

**Likely follow-up:** Could the model still cite 2015 context? Yes, for explanation when supplied, but it is told not to present it as current action. Deterministic filtering prevents the most dangerous case: an obsolete recommendation entering evidence at all.

## 5. Why is scope classification deterministic rather than an LLM call?

The project boundary is a finite configured list of included and excluded cancer sites, so phrase/token matching is cheaper, faster, reproducible, and runs before embeddings or generation. The blind baseline had 0% false refusal. The observed gall-bladder/bladder collision was a concrete phrase-overlap bug; excluded spans are now removed before included-site matching. The full v6 regression run measured 100% scope, 100% correct refusal, and 0% false refusal.

**Likely follow-up:** What happens with “Should this patient be referred?” or “The patient has stomach issues—is this serious?” A separate minimum-answerability check returns “insufficient information” before retrieval or generation. It requires a concrete symptom/sign, not merely a site, age qualifier, or vague word. It does not memorize those sentences or encode NG12 eligibility rules; questions containing pain, vomiting, dysphagia, haematemesis, a mass, blood, or another concrete feature continue to grounded generation.

## 6. What does “section-aware chunking” actually preserve?

Current recommendations, symptom-table rows, definitions, and shared-guidance units remain intact rather than being split at arbitrary token counts. Oversized supporting material is split at table-row or sentence/paragraph boundaries. Each chunk retains source version, authority, physical pages, section, cancer sites, recommendation links, record ID, and stable chunk ID.

**Likely follow-up:** Is there still a size limit? Yes; supporting chunks are packed within the configured limit, with the built corpus's maximum at 697 tokens. Structure decides boundaries first; size controls only oversized supporting content.

## 7. Where can hallucination still happen?

After evidence selection, the generator can still over-generalize, combine qualifiers incorrectly, introduce an unsupported workflow step, or attach an existing citation label to a claim the passage does not entail. The prompt now explicitly treats absent patient facts as unknown and preserves modality plus AND/OR logic, but prompting cannot prove compliance. Deterministic validation proves labels exist, not entailment; semantic risk remains an offline evaluation concern.

**Likely follow-up:** Why not let validation reject every unsupported claim? Syntax and ID range are deterministic; semantic entailment is not. Putting an LLM judge in the live response path would add latency and another fallible model, so it remains evaluation tooling.

## 8. Why is blind Recall@1 only 75.7% while Recall@5 is 97.3%?

The corpus intentionally contains related canonical recommendations plus tables, evidence, and rationale. On paraphrased or ambiguous questions, a highly related unit can rank above the exact gold unit. In 36 of 37 scored blind retrieval cases the gold evidence was still in the first five, and generation receives six passages, so evidence-set coverage is strong even though first-rank precision leaves room for measured future work.

**Likely follow-up:** Why not tune until Recall@1 is higher? Because the blind set is an evaluation set, not a tuning target. Post-hoc tuning would weaken the credibility of the measurement and might damage grounding or authority behavior.

## 9. What does the LLM judge establish—and what does it not establish?

It estimates semantic properties that deterministic code cannot: claim support, citation entailment, completeness, overreach, and refusal quality against supplied evidence. The default is one independent primary judge, with a second only for failed or uncertain cases. It does not prove medical correctness, patient safety, NICE validity, absence of judge bias, or clinical readiness.

**Likely follow-up:** Why keep the multi-pass evaluator? Only as optional research tooling for stability analysis. It is isolated under evaluation code and never participates in query runtime or the default hackathon evaluation.

## 10. How can a judge reproduce exactly what you evaluated?

The repository ships the reviewed 440 chunks, matching dense matrix and integrity manifest, and versioned evaluation evidence. The raw PDFs and machine outputs are excluded. One bootstrap command verifies hashes/model/dimensions, installs dependencies, builds the frontend, and runs the regression suite. A clean clone outside the workspace reached API, UI, search, stable evidence provenance, and a citation-valid answer without hidden files.

**Likely follow-up:** Why ship generated artifacts instead of rebuilding? They are only about 8 MB, preserve the exact evaluated snapshot, and avoid requiring redistributable PDFs and Ollama embedding work during judging. Rebuild scripts remain available, but a rebuild must be treated as a newly evaluated snapshot.
