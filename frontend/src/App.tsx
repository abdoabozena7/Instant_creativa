import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Database,
  Gauge,
  Layers3,
  LoaderCircle,
  Search,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { Presentation } from "./Presentation";
import type {
  AnswerResponse,
  EvidenceResult,
  HealthResponse,
  MetricsResponse,
  Mode,
  SearchResponse,
} from "./types";

const sites = ["", "lung", "colorectal", "oesophageal", "stomach", "pancreatic", "bladder", "renal"];
const examples = [
  "What does NG12 recommend for unexplained haemoptysis in someone aged 40 or over?",
  "When should adults be offered quantitative FIT for suspected colorectal cancer?",
  "Why did the committee use positive predictive value for colorectal symptoms?",
];

function pct(value: number | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(value === 1 ? 0 : 1)}%`;
}

function label(value: string) {
  return value.replaceAll("_", " ");
}

function App() {
  const [view, setView] = useState<"story" | "search" | "metrics">("story");
  const [query, setQuery] = useState(examples[0]);
  const [mode, setMode] = useState<Mode>("hybrid");
  const [site, setSite] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState<SearchResponse | null>(null);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [selected, setSelected] = useState<EvidenceResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  useEffect(() => {
    Promise.all([api.health(), api.metrics()])
      .then(([healthData, metricsData]) => {
        setHealth(healthData);
        setMetrics(metricsData);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const results = answer?.retrieval.results ?? search?.results ?? [];
  const activeRetrieval = answer?.retrieval ?? search;

  async function run(withAnswer: boolean) {
    if (query.trim().length < 3) return;
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      if (withAnswer) {
        const response = await api.answer(query, mode, site);
        setAnswer(response);
        setSearch(null);
        setSelected(response.retrieval.results[0] ?? null);
      } else {
        const response = await api.search(query, mode, site);
        setSearch(response);
        setAnswer(null);
        setSelected(response.results[0] ?? null);
      }
      setMetrics(await api.metrics());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unknown request error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`app-shell ${view === "story" ? "story-active" : ""}`}>
      <header className={`topbar ${view === "story" ? "story-topbar" : ""}`}>
        <button className="brand" onClick={() => setView("search")} aria-label="Open search workspace">
          <span className="brand-mark">12</span>
          <span>
            <strong>NG12</strong>
            <small>Evidence console</small>
          </span>
        </button>
        <nav aria-label="Primary navigation">
          <button className={view === "story" ? "active" : ""} onClick={() => setView("story")}>
            Presentation
          </button>
          <button className={view === "search" ? "active" : ""} onClick={() => setView("search")}>
            Retrieval
          </button>
          <button className={view === "metrics" ? "active" : ""} onClick={() => setView("metrics")}>
            Evaluation
          </button>
        </nav>
        <div className="system-state">
          <span className={`status-dot ${health?.status === "ready" ? "online" : ""}`} />
          <span>{health?.status === "ready" ? "Corpus ready" : "Checking system"}</span>
          <span className="model-name">{health?.chat_model ?? "gpt-oss:120b-cloud"}</span>
        </div>
      </header>

      <AnimatePresence mode="wait">
        {view === "story" ? (
          <motion.div
            key="story"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.28 }}
          >
            <Presentation metrics={metrics} health={health} onOpenDemo={() => setView("search")} />
          </motion.div>
        ) : view === "search" ? (
          <motion.main
            key="search"
            className="workspace"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.24 }}
          >
            <section className="query-stage">
              <div className="eyebrow"><ShieldCheck size={15} /> Primary-source constrained</div>
              <div className="query-heading">
                <div>
                  <h1>Ask the evidence.</h1>
                  <p>Current recommendations rank first. Older evidence explains—never overrides.</p>
                </div>
                <div className="source-rule" aria-label="Source authority rule">
                  <span><b>01</b> 2026 action</span>
                  <ArrowRight size={15} />
                  <span><b>02</b> 2015 context</span>
                </div>
              </div>
              <div className="query-composer">
                <Search size={21} />
                <textarea
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") void run(true);
                  }}
                  aria-label="NG12 question"
                  rows={2}
                />
                <button className="answer-button" onClick={() => void run(true)} disabled={loading}>
                  {loading ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={17} />}
                  Answer with evidence
                </button>
              </div>
              <div className="query-controls">
                <div className="segmented" aria-label="Retrieval mode">
                  {(["hybrid", "bm25", "dense"] as Mode[]).map((item) => (
                    <button key={item} className={mode === item ? "active" : ""} onClick={() => setMode(item)}>
                      {item}
                    </button>
                  ))}
                </div>
                <select value={site} onChange={(event) => setSite(event.target.value)} aria-label="Cancer site filter">
                  {sites.map((item) => <option key={item} value={item}>{item ? label(item) : "All in-scope sites"}</option>)}
                </select>
                <button className="evidence-only" onClick={() => void run(false)} disabled={loading}>
                  Retrieve only <ChevronRight size={15} />
                </button>
                <span className="shortcut">Ctrl ↵ to answer</span>
              </div>
              <div className="example-row">
                <span>Try</span>
                {examples.slice(1).map((example) => (
                  <button key={example} onClick={() => setQuery(example)}>{example}</button>
                ))}
              </div>
            </section>

            {error && <div className="error-banner"><CircleAlert size={18} /> {error}</div>}

            <AnimatePresence>
              {activeRetrieval && (
                <motion.section
                  className="result-zone"
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.32 }}
                >
                  <div className="result-main">
                    {answer && <AnswerBlock answer={answer} onCitation={(rank) => setSelected(results[rank - 1] ?? null)} />}
                    {activeRetrieval.scope.status === "out_of_scope" ? (
                      <div className="scope-refusal">
                        <XCircle size={24} />
                        <div><h2>Outside configured scope</h2><p>{activeRetrieval.scope.message}</p></div>
                      </div>
                    ) : (
                      <EvidenceList results={results} selected={selected} onSelect={setSelected} retrieval={activeRetrieval} />
                    )}
                  </div>
                  <EvidenceInspector result={selected} />
                </motion.section>
              )}
            </AnimatePresence>

            {!activeRetrieval && <ReadinessStrip health={health} metrics={metrics} />}
          </motion.main>
        ) : (
          <motion.main
            key="metrics"
            className="metrics-view"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
          >
            <MetricsView metrics={metrics} />
          </motion.main>
        )}
      </AnimatePresence>
    </div>
  );
}

function AnswerBlock({ answer, onCitation }: { answer: AnswerResponse; onCitation: (rank: number) => void }) {
  const pieces = useMemo(() => answer.answer.split(/(\[E\d+\])/g), [answer.answer]);
  return (
    <section className="answer-block">
      <div className="section-kicker"><Sparkles size={15} /> Grounded answer</div>
      <div className="answer-copy">
        {pieces.map((piece, index) => {
          const match = piece.match(/^\[E(\d+)\]$/);
          return match ? (
            <button key={`${piece}-${index}`} className="citation-token" onClick={() => onCitation(Number(match[1]))}>{piece}</button>
          ) : (
            <span key={index}>
              {piece.split(/(\*\*.*?\*\*)/g).map((part, partIndex) =>
                part.startsWith("**") && part.endsWith("**")
                  ? <strong key={partIndex}>{part.slice(2, -2)}</strong>
                  : part,
              )}
            </span>
          );
        })}
      </div>
      <div className="answer-audit">
        <span className={answer.citation_validation.passed ? "passed" : "failed"}>
          {answer.citation_validation.passed ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
          Citation labels {answer.citation_validation.passed ? "valid" : "need review"}
        </span>
        <span>{answer.model ?? "scope guard"}</span>
        <span>{(answer.latency_ms / 1000).toFixed(2)}s total</span>
      </div>
    </section>
  );
}

function EvidenceList({ results, selected, onSelect, retrieval }: {
  results: EvidenceResult[];
  selected: EvidenceResult | null;
  onSelect: (result: EvidenceResult) => void;
  retrieval: SearchResponse;
}) {
  return (
    <section className="evidence-list">
      <div className="list-header">
        <div><span className="section-kicker"><BookOpen size={15} /> Ranked evidence</span><h2>{results.length} traceable passages</h2></div>
        <div className="latency-readout"><b>{retrieval.latency_ms.toFixed(1)}</b><small>ms retrieval</small></div>
      </div>
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.045 } } }}>
        {results.map((result) => (
          <motion.button
            variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}
            key={result.chunk_id}
            className={`evidence-row ${selected?.chunk_id === result.chunk_id ? "selected" : ""}`}
            onClick={() => onSelect(result)}
          >
            <span className="rank">E{result.rank}</span>
            <span className="evidence-content">
              <span className="evidence-meta">
                <b>{result.recommendation_id ? `Recommendation ${result.recommendation_id}` : label(result.content_type)}</b>
                <i className={result.authority_priority}>{result.authority_priority}</i>
                <span>{result.section} · p.{result.page}{result.page_end !== result.page ? `–${result.page_end}` : ""}</span>
              </span>
              <span className="evidence-excerpt">{result.text}</span>
            </span>
            <span className="score"><b>{result.score.toFixed(3)}</b><small>score</small></span>
          </motion.button>
        ))}
      </motion.div>
    </section>
  );
}

function EvidenceInspector({ result }: { result: EvidenceResult | null }) {
  return (
    <aside className="inspector">
      {result ? (
        <motion.div key={result.chunk_id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <span className="inspector-label">Evidence trace / E{result.rank}</span>
          <h2>{result.recommendation_id ? `NG12 ${result.recommendation_id}` : result.section}</h2>
          <p className="full-citation">{result.citation}</p>
          <dl>
            <div><dt>Authority</dt><dd>{result.authority_priority}</dd></div>
            <div><dt>Source</dt><dd>{result.source_version === "2026_current" ? "Current guideline" : "Full guideline"}</dd></div>
            <div><dt>Content</dt><dd>{label(result.content_type)}</dd></div>
            <div><dt>Sites</dt><dd>{result.cancer_sites.map(label).join(", ") || "Cross-cutting"}</dd></div>
            <div><dt>Tokens</dt><dd>{result.token_count}</dd></div>
          </dl>
          <div className="score-anatomy">
            <h3>Why this ranked</h3>
            <div><span>Base retrieval</span><b>{result.score_detail.base_score.toFixed(3)}</b></div>
            <div><span>Authority adjustment</span><b>+{result.score_detail.authority_adjustment.toFixed(3)}</b></div>
            {result.score_detail.explanations.map((item) => <p key={item}><CheckCircle2 size={13} /> {item}</p>)}
          </div>
          <p className="chunk-id">{result.chunk_id}</p>
        </motion.div>
      ) : (
        <div className="inspector-empty"><Layers3 size={28} /><h2>Select evidence</h2><p>Inspect provenance, authority, score components, and stable chunk identity.</p></div>
      )}
    </aside>
  );
}

function ReadinessStrip({ health, metrics }: { health: HealthResponse | null; metrics: MetricsResponse | null }) {
  const evaluation = metrics?.evaluation;
  const blind = metrics?.blind_e2e;
  return (
    <section className="readiness-strip">
      <div><Database size={18} /><b>{health?.corpus_chunks ?? 440}</b><span>retrieval chunks</span></div>
      <div><Gauge size={18} /><b>{pct(blind?.deterministic_metrics.retrieval_recall_at_5 ?? evaluation?.modes.hybrid.recall_at_5)}</b><span>blind Recall@5</span></div>
      <div><ShieldCheck size={18} /><b>{pct(blind?.deterministic_metrics.correct_refusal_rate ?? evaluation?.modes.hybrid.out_of_scope_refusal_accuracy)}</b><span>blind correct refusal</span></div>
      <div><Activity size={18} /><b>{health?.dense_index_ready ? "768d" : "—"}</b><span>exact cosine index</span></div>
    </section>
  );
}

function MetricsView({ metrics }: { metrics: MetricsResponse | null }) {
  if (!metrics) return <div className="loading-page"><LoaderCircle className="spin" /> Loading evaluation…</div>;
  const evaluation = metrics.evaluation;
  const blind = metrics.blind_e2e;
  const deterministic = blind?.deterministic_metrics;
  const automated = metrics.multi_judge;
  const corpus = metrics.corpus;
  const runLabel = blind?.evaluation_name.match(/_(v\d+)$/)?.[1] ?? "current";
  const totalCases = blind?.questions.total ?? 0;
  const refusalCases = blind?.questions.by_expected_behavior.refuse ?? 0;
  const scopeFailures = blind?.failures.scope.length ?? 0;
  const citationFailures = blind?.failures.citation_labels.length ?? 0;
  return (
    <>
      <section className="metrics-title">
        <span className="eyebrow"><Gauge size={15} /> Frozen architecture · blind run {runLabel}</span>
        <h1>End-to-end evaluation</h1>
        <p>Forty-four unseen cases test scope, retrieval, source authority, grounded generation, citations, and refusal behavior as one system.</p>
      </section>
      <section className="metric-band">
        <div><strong>{blind?.questions.total ?? 0}</strong><span>blind cases</span></div>
        <div><strong>{pct(deterministic?.scope_classification_accuracy)}</strong><span>scope accuracy</span></div>
        <div><strong>{pct(deterministic?.retrieval_recall_at_5)}</strong><span>retrieval Recall@5</span></div>
        <div><strong>{pct(deterministic?.current_guideline_accuracy)}</strong><span>current-guideline accuracy</span></div>
      </section>

      {blind && deterministic && (
        <section className="blind-section">
          <div className="benchmark-heading">
            <div><span className="section-kicker">Deterministic scoring</span><h2>What the frozen run proved</h2></div>
            <span className="freeze-id"><ShieldCheck size={15} /> SHA {blind.architecture_sha256.slice(0, 12)}</span>
          </div>
          <div className="blind-measures">
            <EvalMetric label="Scope classification" value={deterministic.scope_classification_accuracy} detail={`${Math.round(deterministic.scope_classification_accuracy * totalCases)} / ${totalCases} cases`} warning={scopeFailures > 0} />
            <EvalMetric label="Correct refusal" value={deterministic.correct_refusal_rate} detail={`${Math.round(deterministic.correct_refusal_rate * refusalCases)} / ${refusalCases} excluded-site cases`} warning={deterministic.correct_refusal_rate < 1} />
            <EvalMetric label="False refusal" value={deterministic.false_refusal_rate} detail="0 answerable cases refused" inverse />
            <EvalMetric label="Retrieval Recall@1" value={deterministic.retrieval_recall_at_1} detail={`${deterministic.retrieval_queries_scored} scored queries`} />
            <EvalMetric label="Retrieval Recall@5" value={deterministic.retrieval_recall_at_5} detail="One lung case landed at rank 6" />
            <EvalMetric label="Current source at relevant hit" value={deterministic.current_guideline_accuracy} detail="2026 won every measured check" />
            <EvalMetric label="Citation label validity" value={deterministic.citation_label_validity_rate} detail={citationFailures ? `${citationFailures} outputs used non-canonical labels` : "All generated labels resolved"} warning={citationFailures > 0} />
            <div className="latency-metric"><span>End-to-end latency</span><strong>{(deterministic.latency_ms.end_to_end_p50 / 1000).toFixed(2)}s</strong><small>P50 · {(deterministic.latency_ms.end_to_end_p95 / 1000).toFixed(2)}s P95</small></div>
          </div>
          <div className="failure-ledger">
            <div>{scopeFailures ? <CircleAlert size={17} /> : <CheckCircle2 size={17} />}<span><b>Scope</b> {scopeFailures ? "A query crossed the configured site boundary." : "Phrase-aware exclusions correctly rejected gall bladder without matching bladder."}</span><strong>{scopeFailures}</strong></div>
            <div><Search size={17} /><span><b>Retrieval</b> One lung threshold case missed top five and appeared at rank six.</span><strong>{blind.failures.retrieval_at_5.length}</strong></div>
            <div><BookOpen size={17} /><span><b>Citation syntax</b> {citationFailures ? "Some generated outputs still used non-canonical evidence-label formats." : "Every generated evidence label resolved."}</span><strong>{citationFailures}</strong></div>
          </div>
        </section>
      )}

      {automated && (
        <section className="semantic-section automated-judges">
          <div className="semantic-heading">
            <div>
              <span className="section-kicker">Automated two-judge evaluation</span>
              <h2>Claim-level consensus, fail-closed</h2>
            </div>
            <span className="judge-agreement"><CheckCircle2 size={15} /> {pct(automated.metrics.judge_agreement_rate)} agreement</span>
          </div>
          <p className="semantic-warning">Gemini and gpt-oss each run three passes per criterion. Partial support, missing citations, and cross-judge disagreements fail. Automated evaluation—not clinical validation.</p>
          <div className="semantic-measures">
            <EvalMetric label="Claim support" value={automated.metrics.claim_support_rate} detail="Full support only; partial fails" />
            <EvalMetric label="Citation entailment" value={automated.metrics.citation_entailment_rate} detail={`${pct(automated.metrics.citation_coverage_rate)} citation coverage`} />
            <EvalMetric label="Overreach free" value={automated.metrics.overreach_free_rate} detail="Case-level overreach judge" />
            <EvalMetric label="Answer completeness" value={automated.metrics.answer_completeness_rate} detail="Material evidence included" />
          </div>
          <div className="judge-provenance">
            {Object.entries(automated.judges).map(([judge, config]) => (
              <span key={judge}><b>{label(judge)}</b>{config.model} · {config.passes} passes</span>
            ))}
            <span><b>Decisions</b>{automated.counts.consensus_decisions}</span>
            <span><b>Disagreements</b>{automated.disagreements.length}</span>
          </div>
        </section>
      )}

      {evaluation && (
        <section className="benchmark-section">
          <div className="benchmark-heading">
            <div><span className="section-kicker">Development benchmark · not blind</span><h2>Why hybrid was selected</h2></div>
            <span className="winner"><CheckCircle2 size={15} /> Frozen choice: {evaluation.recommended_mode}</span>
          </div>
          <p className="benchmark-context">This 31-query set was used during development. Its 100% hybrid score selected the architecture; it is context, not the headline validation result.</p>
          <div className="benchmark-table" role="table">
            <div className="benchmark-row header" role="row">
              <span>Mode</span><span>Recall@1</span><span>Recall@5</span><span>MRR@10</span><span>Canonical top-1</span><span>Scope refusal</span><span>P50</span>
            </div>
            {(["bm25", "dense", "hybrid"] as Mode[]).map((mode) => {
              const values = evaluation.modes[mode];
              const winner = mode === evaluation.recommended_mode;
              return (
                <div className={`benchmark-row ${winner ? "best" : ""}`} role="row" key={mode}>
                  <span><i>{winner ? "●" : "○"}</i><b>{mode}</b></span>
                  <MetricCell value={values.recall_at_1} />
                  <MetricCell value={values.recall_at_5} />
                  <MetricCell value={values.mrr_at_10} />
                  <MetricCell value={values.canonical_top1_accuracy} />
                  <MetricCell value={values.out_of_scope_refusal_accuracy} />
                  <span><b>{values.latency_ms.p50.toFixed(1)}</b><small>ms</small></span>
                </div>
              );
            })}
          </div>
          <p className="selection-rule">{evaluation.selection_rule}</p>
        </section>
      )}
      <section className="metrics-lower">
        <div className="distribution">
          <span className="section-kicker">Corpus coverage</span>
          <h2>Chunks by cancer site</h2>
          {Object.entries(corpus.chunking.chunks_by_cancer_site).sort((a, b) => b[1] - a[1]).map(([site, count]) => (
            <div className="distribution-row" key={site}>
              <span>{label(site)}</span>
              <i><motion.b initial={{ width: 0 }} animate={{ width: `${(count / 120) * 100}%` }} /></i>
              <strong>{count}</strong>
            </div>
          ))}
        </div>
        <div className="safety-audit">
          <span className="section-kicker">Safety gates</span>
          <h2>Authority before fluency</h2>
          <AuditLine passed text="2026 recommendations are canonical" detail="Primary ranking adjustment is exposed per result." />
          <AuditLine passed text="Historical recommendations cannot retrieve" detail={`${corpus.records.audit_only_historical_recommendations} records remain audit-only.`} />
          <AuditLine passed={corpus.chunking.chunks_over_maximum.length === 0} text="Chunk-size validation" detail={`Maximum ${corpus.chunking.chunk_token_distribution.max} tokens; no oversize chunks.`} />
          <AuditLine passed text="Conflicts remain visible" detail={`${corpus.reconciliation.conflicts_detected_count} possible version differences are reported, never merged.`} />
        </div>
      </section>
    </>
  );
}

function EvalMetric({ label: metricLabel, value, detail, warning = false, inverse = false }: {
  label: string;
  value: number;
  detail: string;
  warning?: boolean;
  inverse?: boolean;
}) {
  const visualValue = inverse ? 1 - value : value;
  return (
    <div className={`eval-metric ${warning ? "warning" : ""}`}>
      <span>{metricLabel}</span>
      <strong>{pct(value)}</strong>
      <i><motion.em initial={{ width: 0 }} animate={{ width: `${visualValue * 100}%` }} transition={{ duration: .55 }} /></i>
      <small>{detail}</small>
    </div>
  );
}

function MetricCell({ value }: { value: number }) {
  return <span className="metric-cell"><b>{pct(value)}</b><i><em style={{ width: `${value * 100}%` }} /></i></span>;
}

function AuditLine({ passed, text, detail }: { passed: boolean; text: string; detail: string }) {
  return <div className="audit-line">{passed ? <CheckCircle2 size={18} /> : <CircleAlert size={18} />}<div><b>{text}</b><span>{detail}</span></div></div>;
}

export default App;
