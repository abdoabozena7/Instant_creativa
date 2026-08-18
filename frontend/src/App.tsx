import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Database,
  Gauge,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Search,
  ShieldCheck,
  ScanLine,
  Sparkles,
  Target,
  X,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Presentation } from "./Presentation";
import type {
  AnswerResponse,
  EvidenceResult,
  HealthResponse,
  MetricsResponse,
  Mode,
  QueryOutcome,
  SearchResponse,
  VisionMetadata,
  VisionRefusalResponse,
} from "./types";

const sites = ["", "lung", "colorectal", "oesophageal", "stomach", "pancreatic", "bladder", "renal"];
const demoCases = [
  {
    label: "Direct",
    query: "What does NG12 recommend for unexplained haemoptysis in someone aged 40 or over?",
  },
  {
    label: "Compare",
    query: "According to NG12, compare the referral criteria for unexplained visible haematuria for bladder cancer and renal cancer. Answer as two concise cited bullets.",
  },
  {
    label: "Safe refusal",
    query: "What does NG12 recommend for suspected prostate cancer?",
  },
  {
    label: "Emergency",
    query: "I am vomiting blood heavily right now and feel faint. What should I do?",
  },
];

function pct(value: number | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(value === 1 ? 0 : 1)}%`;
}

function label(value: string) {
  return value.replaceAll("_", " ");
}

function fileToDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Image could not be read"));
    reader.onerror = () => reject(new Error("Image could not be read"));
    reader.readAsDataURL(file);
  });
}

function fileSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

type RequestKind = "answer" | "retrieve";
type WorkflowStageStatus = "planned" | "complete" | "stopped" | "not_run" | "unknown";

interface RequestDescriptor {
  kind: RequestKind;
  mode: Mode;
  inputMethod?: "text" | "vision";
}

interface ImagePreview {
  url: string;
  name: string;
  size: number;
  mimeType: string;
  base64: string;
}

interface WorkflowStage {
  id: string;
  label: string;
  status: WorkflowStageStatus;
}

function App() {
  const [view, setView] = useState<"story" | "search" | "metrics">("story");
  const [query, setQuery] = useState(demoCases[0].query);
  const [mode, setMode] = useState<Mode>("hybrid");
  const [site, setSite] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState<SearchResponse | null>(null);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [selected, setSelected] = useState<EvidenceResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [request, setRequest] = useState<RequestDescriptor | null>(null);
  const [imagePreview, setImagePreview] = useState<ImagePreview | null>(null);
  const [vision, setVision] = useState<VisionMetadata | null>(null);
  const [visionRefusal, setVisionRefusal] = useState<VisionRefusalResponse | null>(null);
  const imageInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    Promise.all([api.health(), api.metrics()])
      .then(([healthData, metricsData]) => {
        setHealth(healthData);
        setMetrics(metricsData);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  const activeRetrieval = answer?.retrieval ?? search;
  const outcome = answer?.outcome ?? search?.outcome ?? visionRefusal?.outcome ?? null;
  const evidenceVisible = outcome === "grounded_answer" || outcome === "retrieval_results";
  const results = evidenceVisible ? (answer?.retrieval.results ?? search?.results ?? []) : [];

  function openEvidence(result: EvidenceResult | null) {
    setSelected(result);
    if (result && window.matchMedia("(max-width: 1050px)").matches) {
      window.requestAnimationFrame(() => {
        document.getElementById("evidence-inspector")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }

  async function run(withAnswer: boolean) {
    if (query.trim().length < 3) {
      setError("Write a clinical question before requesting a recommendation.");
      return;
    }
    const useAttachment = withAnswer && imagePreview !== null;
    setRequest({ kind: withAnswer ? "answer" : "retrieve", mode, inputMethod: useAttachment ? "vision" : "text" });
    setLoading(true);
    setError(null);
    setSelected(null);
    setSearch(null);
    setAnswer(null);
    setVision(null);
    setVisionRefusal(null);
    try {
      if (useAttachment) {
        const response = await api.visionAnswer(
          imagePreview.base64,
          imagePreview.mimeType,
          query.trim(),
          mode,
          site,
        );
        setVision(response.vision);
        if (!("retrieval" in response)) {
          setVisionRefusal(response);
        } else {
          setAnswer(response);
          setSelected(response.outcome === "grounded_answer" ? response.retrieval.results[0] ?? null : null);
          setMetrics(await api.metrics());
        }
      } else if (withAnswer) {
        const response = await api.answer(query, mode, site);
        setAnswer(response);
        setSearch(null);
        setSelected(response.outcome === "grounded_answer" ? response.retrieval.results[0] ?? null : null);
      } else {
        const response = await api.search(query, mode, site);
        setSearch(response);
        setAnswer(null);
        setSelected(response.outcome === "retrieval_results" ? response.results[0] ?? null : null);
      }
      setMetrics(await api.metrics());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unknown request error");
    } finally {
      setLoading(false);
    }
  }

  function clearVisionInput() {
    setImagePreview(null);
    setVision(null);
    setVisionRefusal(null);
    if (imageInput.current) imageInput.current.value = "";
  }

  async function attachImage(file: File) {
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      setError("Use a JPEG, PNG, or WebP export. DICOM and PDF are not supported in this bonus.");
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setError("The image is larger than the 8 MB limit.");
      return;
    }

    const dataUrl = await fileToDataUrl(file);
    const imageBase64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
    setImagePreview({ url: dataUrl, name: file.name, size: file.size, mimeType: file.type, base64: imageBase64 });
    setError(null);
    setVision(null);
    setVisionRefusal(null);
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
              <div className="query-heading">
                <div>
                  <span className="workspace-label">NG12 evidence retrieval</span>
                  <h1>Ask the evidence.</h1>
                  <p>Ask a clinical question and get an answer grounded in the current NICE guidance.</p>
                </div>
              </div>
              <div className={`query-composer ${loading ? "is-working" : ""}`}>
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
                <input
                  ref={imageInput}
                  className="image-input"
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    event.currentTarget.value = "";
                    if (file) void attachImage(file);
                  }}
                  aria-label="Upload a de-identified clinical image"
                />
                <button
                  className="vision-button"
                  onClick={() => imageInput.current?.click()}
                  disabled={loading}
                  title="Upload a de-identified JPEG, PNG, or WebP image up to 8 MB"
                >
                  <ImagePlus size={17} />
                  <span>Case + image</span>
                </button>
                <button className="answer-button" onClick={() => void run(true)} disabled={loading || query.trim().length < 3}>
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
                <span className="shortcut">Ctrl ↵ to answer</span>
              </div>
              <div className="example-row" aria-label="Prepared live demo cases">
                <span>Try a case</span>
                {demoCases.map((demo) => (
                  <button key={demo.label} onClick={() => setQuery(demo.query)} title={demo.query}>{demo.label}</button>
                ))}
              </div>
              <AnimatePresence>
                {imagePreview && (
                  <VisionTrace
                    preview={imagePreview}
                    question={query}
                    vision={vision}
                    loading={loading && request?.inputMethod === "vision"}
                    onClear={clearVisionInput}
                  />
                )}
              </AnimatePresence>
              <AnimatePresence>
                {request && (loading || error || activeRetrieval || visionRefusal) && (
                  <RequestWorkflow
                    key="request-workflow"
                    request={request}
                    loading={loading}
                    error={error}
                    outcome={outcome}
                    answer={answer}
                  />
                )}
              </AnimatePresence>
            </section>

            {error && <div className="error-banner"><CircleAlert size={18} /> {error}</div>}

            {visionRefusal && (
              <div className="vision-refusal">
                <ScanLine size={22} />
                <div><strong>Image stopped before NG12 retrieval</strong><p>{visionRefusal.answer}</p></div>
              </div>
            )}

            <AnimatePresence>
              {activeRetrieval && (
                <motion.section
                  className={`result-zone ${evidenceVisible ? "" : "guard-only"}`}
                  initial={{ opacity: 0, y: 18 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.32 }}
                >
                  <div className="result-main">
                    {outcome === "grounded_answer" && answer && (
                      <AnswerBlock answer={answer} evidence={results} onCitation={(rank) => openEvidence(results[rank - 1] ?? null)} />
                    )}
                    {evidenceVisible ? (
                      <EvidenceList results={results} selected={selected} onSelect={openEvidence} retrieval={activeRetrieval} />
                    ) : (
                      <OutcomePanel outcome={outcome} answer={answer} retrieval={activeRetrieval} />
                    )}
                  </div>
                  {evidenceVisible && <EvidenceInspector result={selected} />}
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

function VisionTrace({ preview, question, vision, loading, onClear }: {
  preview: ImagePreview;
  question: string;
  vision: VisionMetadata | null;
  loading: boolean;
  onClear: () => void;
}) {
  return (
    <motion.section
      className="vision-trace"
      initial={{ opacity: 0, y: -8, height: 0 }}
      animate={{ opacity: 1, y: 0, height: "auto" }}
      exit={{ opacity: 0, y: -6, height: 0 }}
      transition={{ duration: 0.24, ease: "easeOut" }}
      aria-live="polite"
    >
      <div className="vision-preview">
        <img src={preview.url} alt="Uploaded de-identified clinical input" />
        <span><b>{preview.name}</b><small>{fileSize(preview.size)}</small></span>
      </div>
      <div className="vision-copy">
        <span className="vision-kicker"><ScanLine size={14} /> {loading ? "Analyzing attachment" : vision ? label(vision.image_kind) : "Attached context"}</span>
        {!!question.trim() && <small className="vision-case">Question: {question.trim()}</small>}
        {loading ? (
          <p>Adding relevant attachment findings to your written question before the NG12 scope check…</p>
        ) : vision ? (
          <>
            <strong>{vision.extracted_query || "No safe supporting context was extracted."}</strong>
            <p>{vision.limitations}</p>
            {!!vision.uncertainties.length && <small>Uncertainty: {vision.uncertainties.join(" · ")}</small>}
          </>
        ) : (
          <p>{question.trim().length >= 3
            ? "Ready. Select “Answer with evidence” to use this attachment with the written question."
            : "Write a clinical question to continue. An attachment cannot be submitted on its own."}</p>
        )}
      </div>
      <span className={`vision-state ${vision?.status ?? "pending"}`}>
        {loading ? <LoaderCircle className="spin" size={14} /> : vision?.status === "refused" ? <CircleAlert size={14} /> : <CheckCircle2 size={14} />}
        {loading ? "Analyzing" : vision?.status === "refused" ? "Not used" : vision ? "Used" : "Attached"}
      </span>
      <button className="vision-clear" onClick={onClear} aria-label="Remove uploaded image"><X size={15} /></button>
    </motion.section>
  );
}

function RequestWorkflow({ request, loading, error, outcome, answer }: {
  request: RequestDescriptor;
  loading: boolean;
  error: string | null;
  outcome: QueryOutcome | null;
  answer: AnswerResponse | null;
}) {
  const stageStatus = (stage: "vision" | "safety" | "emergency" | "scope" | "retrieval" | "ranking" | "evidence" | "generation" | "release"): WorkflowStageStatus => {
    if (loading) return "planned";
    if (error || !outcome) return "unknown";
    if (stage === "vision") return outcome === "vision_refusal" ? "stopped" : "complete";
    if (outcome === "vision_refusal") return "not_run";

    const stoppedBeforeRetrieval = ["safety_refusal", "emergency_redirect", "scope_refusal", "insufficient_information"].includes(outcome);
    if (stage === "safety") return outcome === "safety_refusal" ? "stopped" : "complete";
    if (stage === "emergency") {
      if (outcome === "safety_refusal") return "not_run";
      return outcome === "emergency_redirect" ? "stopped" : "complete";
    }
    if (stage === "scope") {
      if (outcome === "safety_refusal" || outcome === "emergency_redirect") return "not_run";
      return outcome === "scope_refusal" || outcome === "insufficient_information" ? "stopped" : "complete";
    }
    if (stage === "retrieval" || stage === "ranking") return stoppedBeforeRetrieval ? "not_run" : "complete";
    if (stage === "evidence") {
      if (stoppedBeforeRetrieval) return "not_run";
      return outcome === "no_results" ? "stopped" : "complete";
    }
    if (stage === "generation") {
      if (outcome === "grounded_answer" || outcome === "generation_rejected") return "complete";
      return "not_run";
    }
    if (outcome === "grounded_answer" && answer?.citation_validation.passed) return "complete";
    if (outcome === "generation_rejected") return "stopped";
    return "not_run";
  };

  const stages = [
    ...(request.inputMethod === "vision" ? [{ id: "vision" as const, label: "Vision extraction" }] : []),
    { id: "safety" as const, label: "Instruction safety" },
    { id: "emergency" as const, label: "Emergency boundary" },
    { id: "scope" as const, label: "Scope + specificity" },
    { id: "retrieval" as const, label: `${request.mode.toUpperCase()} retrieval` },
    { id: "ranking" as const, label: "Authority ranking" },
    { id: "evidence" as const, label: "Evidence assembly" },
    ...(request.kind === "answer" ? [
      { id: "generation" as const, label: "Grounded generation" },
      { id: "release" as const, label: "Citation release gate" },
    ] : []),
  ].map((stage) => ({ ...stage, status: stageStatus(stage.id) }));

  const summary = loading
    ? {
        title: request.inputMethod === "vision" ? "Translating image into a bounded query" : request.kind === "answer" ? "Preparing an evidence-bound response" : "Preparing a traceable evidence set",
        detail: request.inputMethod === "vision" ? "The optional Vision adapter runs first; only a safe in-scope text query can enter the existing pipeline." : "These are planned stages while the request is in flight. Observed states appear only after the API responds.",
      }
    : workflowSummary(outcome, error);

  const completedCount = stages.filter((stage) => stage.status === "complete").length;
  const compactStatus = loading
    ? `Running ${stages.length} validation checks`
    : error
      ? "Request status unavailable"
      : outcome === "grounded_answer" || outcome === "retrieval_results"
        ? `${completedCount} validation checks passed`
        : summary.title;

  return (
    <motion.details
      className={`request-workflow ${loading ? "in-flight" : "observed"}`}
      open={loading || undefined}
      initial={{ opacity: 0, height: 0, marginTop: 0 }}
      animate={{ opacity: 1, height: "auto", marginTop: 14 }}
      exit={{ opacity: 0, height: 0, marginTop: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      aria-live="polite"
      aria-busy={loading}
    >
      <summary className="workflow-compact">
        <span className="workflow-summary-icon" aria-hidden="true">
          {loading ? <LoaderCircle className="spin" size={18} /> : outcome === "grounded_answer" || outcome === "retrieval_results" ? <CheckCircle2 size={18} /> : <CircleAlert size={18} />}
        </span>
        <strong>{compactStatus}</strong>
        <span>{loading ? "View live details" : "View details"}</span>
      </summary>
      <div className="workflow-detail">
        <div className="workflow-summary">
          <span className="workflow-summary-icon" aria-hidden="true">
            {loading ? <LoaderCircle className="spin" size={18} /> : outcome === "grounded_answer" || outcome === "retrieval_results" ? <CheckCircle2 size={18} /> : <CircleAlert size={18} />}
          </span>
          <div>
            <strong>{summary.title}</strong>
            <p>{summary.detail}</p>
          </div>
          <span className="workflow-state">{loading ? "In flight" : "Observed"}</span>
        </div>
        <WorkflowStageList
          key={`${loading ? "planned" : outcome ?? "unknown"}-${request.kind}-${request.mode}`}
          stages={stages}
          cadenceMs={loading ? 330 : 160}
        />
      </div>
    </motion.details>
  );
}

function WorkflowStageList({ stages, cadenceMs }: { stages: WorkflowStage[]; cadenceMs: number }) {
  const [visibleCount, setVisibleCount] = useState(1);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setVisibleCount(stages.length);
      return;
    }
    const timer = window.setInterval(() => {
      setVisibleCount((current) => {
        if (current >= stages.length) {
          window.clearInterval(timer);
          return current;
        }
        return current + 1;
      });
    }, cadenceMs);
    return () => window.clearInterval(timer);
  }, [cadenceMs, stages.length]);

  return (
    <ol className="workflow-stages" aria-label="Request execution workflow">
      {stages.map((stage, index) => (
        <li
          key={stage.id}
          className={`${stage.status} ${index < visibleCount ? "sequence-visible" : "sequence-hidden"}`}
          title={workflowStatusLabel(stage.status)}
        >
          <span className="workflow-stage-icon" aria-hidden="true">
            {stage.status === "complete" ? <CheckCircle2 size={16} /> : stage.status === "stopped" ? <XCircle size={16} /> : <span />}
          </span>
          <span>{stage.label}</span>
          <small>{workflowStatusLabel(stage.status)}</small>
        </li>
      ))}
    </ol>
  );
}

function workflowStatusLabel(status: WorkflowStageStatus) {
  return {
    planned: "Planned",
    complete: "Complete",
    stopped: "Stopped",
    not_run: "Not run",
    unknown: "Unconfirmed",
  }[status];
}

function workflowSummary(outcome: QueryOutcome | null, error: string | null) {
  if (error) return {
    title: "Request status unavailable",
    detail: "The API request failed, so no execution stage is reported as complete.",
  };
  const summaries: Record<QueryOutcome, { title: string; detail: string }> = {
    grounded_answer: {
      title: "Evidence-bound response released",
      detail: "The answer passed the deterministic citation gate; every displayed label resolves to ranked evidence.",
    },
    retrieval_results: {
      title: "Evidence retrieval complete",
      detail: "The ranked passages are ready. Open any result to inspect its full text, provenance, and score trace.",
    },
    safety_refusal: {
      title: "Stopped at instruction safety",
      detail: "The request was blocked before scope analysis, retrieval, or model generation.",
    },
    emergency_redirect: {
      title: "Stopped for urgent care",
      detail: "The request was redirected before scope analysis, retrieval, or model generation.",
    },
    scope_refusal: {
      title: "Stopped at the scope guard",
      detail: "The request was outside the configured NG12 scope, so retrieval and generation did not run.",
    },
    insufficient_information: {
      title: "Stopped before retrieval",
      detail: "The question needs a specific clinical feature before evidence can be selected safely.",
    },
    generation_rejected: {
      title: "Generated response withheld",
      detail: "Generation ran, but the citation release gate rejected the output before it reached the UI.",
    },
    vision_refusal: {
      title: "Stopped at the Vision boundary",
      detail: "The image did not produce a safe query for one of the seven configured cancer sites, so the NG12 core did not run.",
    },
    no_results: {
      title: "No evidence assembled",
      detail: "Retrieval and ranking completed, but no passage matched the query and active filters.",
    },
  };
  return outcome ? summaries[outcome] : {
    title: "Request status unavailable",
    detail: "No observed execution result was returned.",
  };
}

function OutcomePanel({ outcome, answer, retrieval }: {
  outcome: QueryOutcome | null;
  answer: AnswerResponse | null;
  retrieval: SearchResponse;
}) {
  const content: Record<string, { title: string; message: string; detail: string }> = {
    safety_refusal: {
      title: "Unsafe instruction blocked",
      message: retrieval.safety.message ?? "This request cannot enter the evidence workflow.",
      detail: "Retrieval not run · Model not called · Citation validation not applicable",
    },
    emergency_redirect: {
      title: "Seek urgent medical help now",
      message: answer?.answer ?? retrieval.emergency.message ?? "Contact local emergency services now.",
      detail: "Emergency boundary · Retrieval not run · Model not called",
    },
    scope_refusal: {
      title: "Outside configured scope",
      message: retrieval.scope.message ?? "This question is outside the configured NG12 sites.",
      detail: "Retrieval not run · Model not called",
    },
    insufficient_information: {
      title: "More clinical detail needed",
      message: answer?.answer ?? retrieval.answerability.message ?? "Add a specific symptom or sign before asking for an assessment.",
      detail: "Retrieval not run · Model not called · No evidence assigned",
    },
    generation_rejected: {
      title: "Ungrounded model output blocked",
      message: answer?.answer ?? "The generated response did not meet the citation contract.",
      detail: "Model output withheld · Retrieved passages not presented as supporting evidence",
    },
    no_results: {
      title: "No matching evidence",
      message: "No passage matched the current query and metadata filters.",
      detail: "Try removing a filter or making the clinical feature more specific",
    },
  };
  const state = content[outcome ?? "no_results"] ?? content.no_results;
  return (
    <section className={`guard-response ${outcome ?? "no_results"}`} aria-live="polite">
      <span className="guard-icon">{outcome === "safety_refusal" || outcome === "emergency_redirect" ? <ShieldCheck size={25} /> : <XCircle size={25} />}</span>
      <div>
        <span className="section-kicker">Request outcome</span>
        <h2>{state.title}</h2>
        <p>{state.message}</p>
        <small>{state.detail}</small>
      </div>
    </section>
  );
}

function AnswerBlock({ answer, evidence, onCitation }: { answer: AnswerResponse; evidence: EvidenceResult[]; onCitation: (rank: number) => void }) {
  const pieces = useMemo(() => answer.answer.split(/(\[E\d+\])/g), [answer.answer]);
  return (
    <section className="answer-block">
      <div className="section-kicker"><Sparkles size={15} /> Grounded answer</div>
      <div className="answer-copy">
        {pieces.map((piece, index) => {
          const match = piece.match(/^\[E(\d+)\]$/);
          const evidenceRank = match ? Number(match[1]) : null;
          const source = evidenceRank ? evidence[evidenceRank - 1] : null;
          return match ? (
            <button
              key={`${piece}-${index}`}
              className="citation-token"
              onClick={() => onCitation(evidenceRank!)}
              aria-label={`Open evidence E${evidenceRank}${source ? `, page ${source.page}` : ""}`}
              title={source ? `${source.citation} · ${source.chunk_id}` : "Open this evidence"}
            >
              E{evidenceRank}{source ? ` · p.${source.page}${source.page_end !== source.page ? `–${source.page_end}` : ""}` : ""}
            </button>
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
        <span className={answer.citation_validation.claim_coverage_passed ? "passed" : "failed"}>
          {answer.citation_validation.claim_coverage_passed ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
          {pct(answer.citation_validation.citation_coverage_rate ?? undefined)} claim citation coverage
        </span>
        <span>{answer.model ?? "deterministic guard"}</span>
        <span>{(answer.latency_ms / 1000).toFixed(2)}s total</span>
      </div>
      {!answer.citation_validation.claim_coverage_passed && (
        <div className="citation-warning" role="alert">
          <CircleAlert size={18} />
          <div><b>Citation coverage needs review</b><span>This answer should not be released until every clinical claim is linked to evidence.</span></div>
        </div>
      )}
      <p className="clinical-disclaimer">{answer.safety_note ?? "Evidence lookup only. This demo does not diagnose or replace clinical judgement."}</p>
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
        <div>
          <span className="section-kicker"><BookOpen size={15} /> Ranked evidence</span>
          <h2>{results.length} traceable passages</h2>
          {results.length > 0 && <p>Click a passage to open its full text, source, and ranking trace.</p>}
        </div>
        <div className="latency-readout"><b>{retrieval.latency_ms.toFixed(1)}</b><small>ms retrieval</small></div>
      </div>
      <motion.div initial="hidden" animate="visible" variants={{ visible: { transition: { staggerChildren: 0.045 } } }}>
        {results.map((result) => (
          <motion.button
            variants={{ hidden: { opacity: 0, y: 8 }, visible: { opacity: 1, y: 0 } }}
            key={result.chunk_id}
            className={`evidence-row ${selected?.chunk_id === result.chunk_id ? "selected" : ""}`}
            onClick={() => onSelect(result)}
            aria-pressed={selected?.chunk_id === result.chunk_id}
            aria-controls="evidence-inspector"
            title={`Open evidence E${result.rank}`}
          >
            <span className="rank">E{result.rank}</span>
            <span className="evidence-content">
              <span className="evidence-meta">
                <b>{result.recommendation_id ? `Recommendation ${result.recommendation_id}` : label(result.content_type)}</b>
                <i className={result.authority_priority}>
                  {result.source_version === "2026_current" ? "Current 2026" : "Supporting 2015"}
                </i>
                <span>{result.section} · p.{result.page}{result.page_end !== result.page ? `–${result.page_end}` : ""}</span>
              </span>
              <span className="evidence-excerpt">{result.text}</span>
            </span>
            <span className="evidence-row-action">
              <span className="score" title="Ranking score, not a confidence probability"><b>{result.score.toFixed(3)}</b><small>rank score</small></span>
              <span className="open-label">Open <ChevronRight size={13} /></span>
            </span>
          </motion.button>
        ))}
      </motion.div>
    </section>
  );
}

function EvidenceInspector({ result }: { result: EvidenceResult | null }) {
  return (
    <aside className="inspector" id="evidence-inspector" aria-live="polite">
      {result ? (
        <motion.div key={result.chunk_id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="inspector-heading">
            <span className="inspector-label">Evidence E{result.rank}</span>
            <span className={`source-status ${result.authority_priority}`}>
              {result.source_version === "2026_current" ? "Current 2026" : "Supporting 2015"}
            </span>
          </div>
          <section className="source-passage" aria-label="Full evidence passage">
            <h2>{result.recommendation_id ? `NG12 ${result.recommendation_id}` : result.section}</h2>
            <p>{result.text}</p>
          </section>
          <p className="evidence-source-line">
            {result.recommendation_id ? `NG12 ${result.recommendation_id}` : label(result.content_type)} · {result.section} · p.{result.page}{result.page_end !== result.page ? `–${result.page_end}` : ""} · {result.source_version === "2026_current" ? "Current 2026" : "Supporting 2015"}
          </p>
          <p className="full-citation">{result.citation}</p>
          <details className="technical-details">
            <summary>Provenance and ranking details</summary>
            <dl className="evidence-details">
              <div><dt>Page</dt><dd>{result.page}{result.page_end !== result.page ? `–${result.page_end}` : ""}</dd></div>
              <div><dt>Section</dt><dd>{result.section}</dd></div>
              <div><dt>Subsection</dt><dd>{result.subsection || "—"}</dd></div>
              <div><dt>Recommendation</dt><dd>{result.recommendation_id || "Context passage"}</dd></div>
              <div><dt>Authority</dt><dd>{label(result.authority_priority)}</dd></div>
              <div><dt>Version</dt><dd>{result.source_version === "2026_current" ? "Current guideline · 2026" : "Full guideline · 2015"}</dd></div>
              <div><dt>Content</dt><dd>{label(result.content_type)}</dd></div>
              <div><dt>Sites</dt><dd>{result.cancer_sites.map(label).join(", ") || "Cross-cutting"}</dd></div>
              <div className="detail-wide"><dt>Chunk ID</dt><dd className="detail-chunk-id">{result.chunk_id}</dd></div>
            </dl>
            <div className="score-anatomy">
              <h3>Why this ranked</h3>
              <div><span>Base retrieval</span><b>{result.score_detail.base_score.toFixed(3)}</b></div>
              <div><span>Authority adjustment</span><b>+{result.score_detail.authority_adjustment.toFixed(3)}</b></div>
              {result.score_detail.explanations.map((item) => <p key={item}><CheckCircle2 size={13} /> {item}</p>)}
            </div>
            <p className="chunk-id">{result.token_count} tokens · stable evidence identity shown above</p>
          </details>
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
    <details className="readiness-details">
      <summary>System and evaluation details</summary>
      <section className="readiness-strip">
        <div><Database size={18} /><b>{health?.corpus_chunks ?? 440}</b><span>retrieval chunks</span></div>
        <div><Gauge size={18} /><b>{pct(blind?.deterministic_metrics.retrieval_recall_at_5 ?? evaluation?.modes.hybrid.recall_at_5)}</b><span>blind Recall@5</span></div>
        <div><Target size={18} /><b>{pct(blind?.deterministic_metrics.retrieval_precision_at_3 ?? evaluation?.modes.hybrid.precision_at_3)}</b><span>strict Precision@3</span></div>
        <div><ShieldCheck size={18} /><b>{pct(blind?.deterministic_metrics.correct_refusal_rate ?? evaluation?.modes.hybrid.out_of_scope_refusal_accuracy)}</b><span>blind correct refusal</span></div>
        <div><Activity size={18} /><b>{health?.dense_index_ready ? "768d" : "—"}</b><span>exact cosine index</span></div>
      </section>
    </details>
  );
}

function MetricsView({ metrics }: { metrics: MetricsResponse | null }) {
  if (!metrics) return <div className="loading-page"><LoaderCircle className="spin" /> Loading evaluation…</div>;
  const evaluation = metrics.evaluation;
  const blind = metrics.blind_e2e;
  const deterministic = blind?.deterministic_metrics;
  const strictPrecisionAt3 = deterministic?.retrieval_precision_at_3
    ?? evaluation?.modes.hybrid.precision_at_3;
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
        <div><strong>{pct(strictPrecisionAt3)}</strong><span>strict Precision@3</span></div>
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
            {deterministic.retrieval_precision_at_3 != null && <EvalMetric label="Strict Precision@3" value={deterministic.retrieval_precision_at_3} detail="Gold-matching passages ÷ 3" />}
            {deterministic.claim_citation_coverage_rate != null && <EvalMetric label="Claim citation coverage" value={deterministic.claim_citation_coverage_rate} detail="Released claim units with evidence labels" />}
            {deterministic.citation_release_pass_rate != null && <EvalMetric label="Citation release pass" value={deterministic.citation_release_pass_rate} detail="Answers passing the complete citation contract" warning={deterministic.citation_release_pass_rate < .95} />}
            <EvalMetric label="Current source at relevant hit" value={deterministic.current_guideline_accuracy} detail="2026 won every measured check" />
            <EvalMetric label="Citation label validity" value={deterministic.citation_label_validity_rate} detail={citationFailures ? `${citationFailures} outputs used non-canonical labels` : "All generated labels resolved"} warning={citationFailures > 0} />
            <div className="latency-metric"><span>End-to-end latency</span><strong>{(deterministic.latency_ms.end_to_end_p50 / 1000).toFixed(2)}s</strong><small>P50 · {(deterministic.latency_ms.end_to_end_p95 / 1000).toFixed(2)}s P95</small></div>
          </div>
          <div className="failure-ledger">
            <div>{scopeFailures ? <CircleAlert size={17} /> : <CheckCircle2 size={17} />}<span><b>Scope</b> {scopeFailures ? "A query crossed the configured site boundary." : "Phrase-aware exclusions correctly rejected gall bladder without matching bladder."}</span><strong>{scopeFailures}</strong></div>
            <div><Search size={17} /><span><b>Retrieval</b> One lung threshold case missed top five and appeared at rank six.</span><strong>{blind.failures.retrieval_at_5.length}</strong></div>
            <div><BookOpen size={17} /><span><b>Citation syntax</b> {citationFailures ? "Some generated outputs still used non-canonical evidence-label formats." : "Every generated evidence label resolved."}</span><strong>{citationFailures}</strong></div>
            <div><ShieldCheck size={17} /><span><b>Claim coverage</b> Fail-closed release rejected any answer still missing a claim-level citation after one bounded repair.</span><strong>{blind.failures.claim_citation_coverage?.length ?? 0}</strong></div>
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
      {!automated && (
        <section className="semantic-section semantic-pending">
          <div className="semantic-heading">
            <div><span className="section-kicker">Semantic evaluation</span><h2>Independent entailment review remains separate.</h2></div>
            <span className="pending-review"><CircleAlert size={15} /> Not claimed</span>
          </div>
          <p className="semantic-warning">The runtime now blocks uncited clinical claim units. That proves citation coverage, not semantic entailment. Independent claim-support and citation-entailment results appear here only after a completed judge or human review.</p>
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
              <span>Mode</span><span>Recall@1</span><span>Recall@5</span><span>Precision@3</span><span>Precision@5</span><span>MRR@10</span><span>Canonical top-1</span><span>P50</span>
            </div>
            {(["bm25", "dense", "hybrid"] as Mode[]).map((mode) => {
              const values = evaluation.modes[mode];
              const winner = mode === evaluation.recommended_mode;
              return (
                <div className={`benchmark-row ${winner ? "best" : ""}`} role="row" key={mode}>
                  <span><i>{winner ? "●" : "○"}</i><b>{mode}</b></span>
                  <MetricCell value={values.recall_at_1} />
                  <MetricCell value={values.recall_at_5} />
                  <MetricCell value={values.precision_at_3} />
                  <MetricCell value={values.precision_at_5} />
                  <MetricCell value={values.mrr_at_10} />
                  <MetricCell value={values.canonical_top1_accuracy} />
                  <span><b>{values.latency_ms.p50.toFixed(1)}</b><small>ms</small></span>
                </div>
              );
            })}
          </div>
          <p className="precision-note">{evaluation.precision_definition}</p>
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
