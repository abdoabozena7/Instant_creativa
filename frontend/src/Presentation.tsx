import { motion } from "framer-motion";
import {
  ArrowDown,
  ArrowRight,
  Braces,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Database,
  FileCheck2,
  FileText,
  Fingerprint,
  Gauge,
  GitCompareArrows,
  Play,
  Search,
  ShieldCheck,
  Sparkles,
  Split,
  Target,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { HealthResponse, MetricsResponse } from "./types";


const chapters = [
  ["promise", "The promise"],
  ["sources", "Source truth"],
  ["pipeline", "Evidence pipeline"],
  ["retrieval", "Why hybrid"],
  ["safety", "Safety logic"],
  ["blind", "Blind baseline"],
  ["failures", "What failed"],
  ["evaluation", "Evaluation mode"],
  ["experiments", "Measured fixes"],
  ["demo", "Live evidence"],
] as const;

const pipeline = [
  { label: "Parse", detail: "Native text + table recovery", icon: FileText },
  { label: "Clean", detail: "Meaning preserved", icon: FileCheck2 },
  { label: "Structure", detail: "Recommendation-aware records", icon: Braces },
  { label: "Retrieve", detail: "BM25 + dense", icon: Search },
  { label: "Guard", detail: "Scope + authority", icon: ShieldCheck },
  { label: "Answer", detail: "Claim + evidence", icon: Sparkles },
];

function percent(value: number | undefined, digits = 1) {
  return value == null ? "—" : `${(value * 100).toFixed(value === 1 ? 0 : digits)}%`;
}

function displayLabel(value: string) {
  return value.replaceAll("_", " ");
}

export function Presentation({
  metrics,
  health,
  onOpenDemo,
}: {
  metrics: MetricsResponse | null;
  health: HealthResponse | null;
  onOpenDemo: () => void;
}) {
  const scroller = useRef<HTMLElement | null>(null);
  const [active, setActive] = useState(0);
  const blind = metrics?.blind_e2e;
  const deterministic = blind?.deterministic_metrics;
  const development = metrics?.evaluation;
  const corpus = metrics?.corpus;

  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const nodes = Array.from(root.querySelectorAll<HTMLElement>("[data-story-slide]"));
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(Number((visible.target as HTMLElement).dataset.storySlide));
      },
      { root, threshold: [0.45, 0.7] },
    );
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (["ArrowDown", "ArrowRight", "PageDown", " "].includes(event.key)) {
        event.preventDefault();
        go(Math.min(chapters.length - 1, active + 1));
      }
      if (["ArrowUp", "ArrowLeft", "PageUp"].includes(event.key)) {
        event.preventDefault();
        go(Math.max(0, active - 1));
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [active]);

  function go(index: number) {
    scroller.current
      ?.querySelector(`[data-story-slide="${index}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="story-shell">
      <aside className="story-rail" aria-label="Presentation chapters">
        <span className="story-progress">{String(active + 1).padStart(2, "0")}<i />{String(chapters.length).padStart(2, "0")}</span>
        <nav>
          {chapters.map(([id, title], index) => (
            <button key={id} className={active === index ? "active" : ""} onClick={() => go(index)}>
              <i /> <span>{title}</span>
            </button>
          ))}
        </nav>
        <div className="story-controls">
          <button onClick={() => go(Math.max(0, active - 1))} aria-label="Previous chapter"><ChevronLeft size={17} /></button>
          <button onClick={() => go(Math.min(chapters.length - 1, active + 1))} aria-label="Next chapter"><ChevronRight size={17} /></button>
        </div>
      </aside>

      <main className="story-scroller" ref={scroller}>
        <StorySlide index={0} id="promise" tone="ink">
          <div className="story-opening-copy">
            <motion.span className="story-overline" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>NICE NG12 · Evidence console</motion.span>
            <motion.h1 initial={{ y: 42, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: .65 }}>
              One question.<br />One traceable path<br />to the evidence.
            </motion.h1>
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: .35 }}>
              A narrow-scope medical RAG that shows its source, authority, limits, and failures—not just its answer.
            </motion.p>
          </div>
          <div className="opening-orbit" aria-hidden="true">
            <motion.div className="orbit-ring ring-one" animate={{ rotate: 360 }} transition={{ duration: 28, repeat: Infinity, ease: "linear" }}><span>2026</span></motion.div>
            <motion.div className="orbit-ring ring-two" animate={{ rotate: -360 }} transition={{ duration: 22, repeat: Infinity, ease: "linear" }}><span>NG12</span></motion.div>
            <div className="orbit-core"><ShieldCheck /><b>Evidence</b><small>before fluency</small></div>
          </div>
          <button className="story-next" onClick={() => go(1)}>Begin the evidence journey <ArrowDown size={16} /></button>
        </StorySlide>

        <StorySlide index={1} id="sources">
          <StoryHeading number="01" overline="Source truth" title="Two documents. One authority rule." copy="Current recommendations answer the question. Historical material explains why." />
          <div className="source-stage">
            <motion.div className="source-document current" initial={{ x: -80, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true, amount: .5 }}>
              <span>Primary</span><strong>2026</strong><b>101 pages</b><p>Current actions, ages, thresholds, investigations, and referral wording.</p>
            </motion.div>
            <div className="authority-spine"><motion.i initial={{ scaleY: 0 }} whileInView={{ scaleY: 1 }} viewport={{ once: true }} transition={{ duration: .8 }} /><ShieldCheck /><span>Always wins</span></div>
            <motion.div className="source-document history" initial={{ x: 80, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true, amount: .5 }}>
              <span>Supporting</span><strong>2015</strong><b>382 pages</b><p>Evidence tables, rationale, committee context, and historical wording.</p>
            </motion.div>
          </div>
          <div className="source-foot"><Fingerprint size={16} /> 483 physical pages preserved with page-level provenance · OCR not required</div>
        </StorySlide>

        <StorySlide index={2} id="pipeline" tone="soft">
          <StoryHeading number="02" overline="Evidence pipeline" title="Clinical structure survives every step." copy="The system never jumps from PDF to embeddings. Each stage produces an auditable artifact." />
          <div className="pipeline-stage">
            <motion.div className="pipeline-beam" initial={{ scaleX: 0 }} whileInView={{ scaleX: 1 }} viewport={{ once: true, amount: .6 }} transition={{ duration: 1.4 }} />
            {pipeline.map((step, index) => {
              const Icon = step.icon;
              return (
                <motion.div className="pipeline-node" key={step.label} initial={{ y: 28, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true }} transition={{ delay: index * .12 }}>
                  <span><Icon size={21} /></span><b>{step.label}</b><small>{step.detail}</small>
                </motion.div>
              );
            })}
          </div>
          <div className="corpus-result">
            <MetricBeat value={corpus?.records.retained_after_scope_filtering ?? 445} label="audited records" />
            <ArrowRight />
            <MetricBeat value={corpus?.records.retrieval_eligible ?? 404} label="retrieval eligible" />
            <ArrowRight />
            <MetricBeat value={corpus?.chunking.chunks_total ?? 440} label="traceable chunks" />
          </div>
        </StorySlide>

        <StorySlide index={3} id="retrieval" tone="ink">
          <StoryHeading number="03" overline="Measured retrieval" title="Hybrid won the experiment." copy="Dense was not assumed to be better. Three modes competed on the same development set." />
          <div className="retrieval-comparison">
            {(["bm25", "dense", "hybrid"] as const).map((mode, index) => {
              const result = development?.modes[mode];
              const value = result?.recall_at_1 ?? [0.964, 0.857, 1][index];
              return (
                <div className={`retrieval-bar ${mode === "hybrid" ? "winner" : ""}`} key={mode}>
                  <span>{mode}</span><motion.i initial={{ width: 0 }} whileInView={{ width: `${value * 100}%` }} viewport={{ once: true }} transition={{ duration: .8, delay: index * .16 }} /><strong>{percent(value)}</strong>
                </div>
              );
            })}
          </div>
          <div className="retrieval-verdict"><GitCompareArrows /><p><b>55% lexical</b> catches exact thresholds and recommendation IDs.<br /><b>45% dense</b> catches paraphrases and clinical intent.</p><span>Chosen by measurement</span></div>
        </StorySlide>

        <StorySlide index={4} id="safety">
          <StoryHeading number="04" overline="Safety logic" title="Before generation, the system decides whether it should answer." copy="Scope and source authority are deterministic gates—not suggestions to the language model." />
          <div className="safety-flow">
            <div className="question-signal"><span>Question</span><motion.i animate={{ x: [0, 25, 0] }} transition={{ repeat: Infinity, duration: 2 }} /></div>
            <Split size={32} />
            <motion.div className="safety-branch accepted" whileInView={{ x: 0, opacity: 1 }} initial={{ x: -30, opacity: 0 }}><Check /><b>In scope</b><span>Lung · colorectal · upper GI · bladder · renal</span></motion.div>
            <motion.div className="safety-branch refused" whileInView={{ x: 0, opacity: 1 }} initial={{ x: 30, opacity: 0 }}><ShieldCheck /><b>Out of scope</b><span>Refuse before retrieval and generation</span></motion.div>
          </div>
          <div className="authority-rule"><span>2026 action</span><motion.i initial={{ width: 0 }} whileInView={{ width: "100%" }} viewport={{ once: true }} /><span>2015 context</span><b>Never reversed</b></div>
        </StorySlide>

        <StorySlide index={5} id="blind" tone="soft">
          <StoryHeading number="05" overline="Blind end-to-end baseline" title="The easy 100% was not the final story." copy="Forty-four harder cases measured the complete journey and exposed real weaknesses." />
          <div className="blind-layout">
            <div className="case-matrix">
              {Object.entries(blind?.questions.by_scope_group ?? { lung: 11, colorectal: 11, upper_gi: 11, bladder_renal: 11 }).map(([group, count], groupIndex) => (
                <div key={group}><span>{displayLabel(group)}</span><i>{Array.from({ length: count }).map((_, index) => <motion.b key={index} initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }} transition={{ delay: (groupIndex * 11 + index) * .018 }} />)}</i><strong>{count}</strong></div>
              ))}
            </div>
            <div className="chance-stage">
              <RateRing value={deterministic?.retrieval_recall_at_1 ?? .7568} label="Top 1" detail="correct evidence first" />
              <ArrowRight size={28} />
              <RateRing value={deterministic?.retrieval_recall_at_5 ?? .973} label="Top 5" detail="correct evidence present" accent />
              <p>Measured hit rate on 37 retrieval-scored blind cases—not a clinical probability.</p>
            </div>
          </div>
          <div className="blind-metrics-line">
            <span><b>{percent(deterministic?.scope_classification_accuracy ?? .9773)}</b>scope</span>
            <span><b>{percent(deterministic?.correct_refusal_rate ?? .8)}</b>correct refusal</span>
            <span><b>{percent(deterministic?.false_refusal_rate ?? 0)}</b>false refusal</span>
            <span><b>{percent(deterministic?.current_guideline_accuracy ?? 1)}</b>2026 accuracy</span>
          </div>
        </StorySlide>

        <StorySlide index={6} id="failures" tone="ink">
          <StoryHeading number="06" overline="Failure analysis" title="Three failures. Three isolated fixes." copy="The blind run tells us what to change—and what not to touch." />
          <div className="failure-story">
            <FailureBeat marker="A" before="gall bladder" fault="substring: bladder" after="phrase-level scope" metric="Correct refusal 80% → 100%" />
            <FailureBeat marker="B" before="[ **E2** ]" fault="strict label parser" after="citation normalization" metric="Validity 80% → 100%" />
            <FailureBeat marker="C" before="Some stomach issues—is this serious?" fault="vague patient description" after="minimum-answerability gate" metric="No retrieval · no model call" />
          </div>
          <div className="do-not-touch"><ShieldCheck /><p><b>Retrieval architecture stays frozen.</b> Recall@5 is already above the 95% target.</p></div>
        </StorySlide>

        <StorySlide index={7} id="evaluation">
          <StoryHeading number="07" overline="Hackathon evaluation mode" title="Computer scores facts. Gemini judges meaning." copy="The sophisticated multi-judge runner remains available, but the presentation path is deliberately lightweight." />
          <div className="evaluation-split">
            <div className="eval-lane deterministic-lane"><span>Deterministic</span><strong>Exact</strong>{["Scope", "Recall@K", "MRR", "2026 priority", "Citation syntax", "Latency"].map((item) => <p key={item}><Check size={14} />{item}</p>)}</div>
            <motion.div className="evaluation-arrow" animate={{ opacity: [.3, 1, .3] }} transition={{ repeat: Infinity, duration: 1.8 }}><ArrowRight /></motion.div>
            <div className="eval-lane judge-lane"><span>Independent judge</span><strong>Semantic</strong>{["Claim support", "Citation entailment", "Overreach"].map((item) => <p key={item}><Sparkles size={14} />{item}</p>)}</div>
            <div className="conditional-judge"><CircleAlert size={17} /><p><b>Gemini · one pass</b>Primary decision</p><ArrowDown /><p><b>Second judge only if needed</b>Any primary failure · disagreement fails closed</p></div>
          </div>
          <div className="evaluation-policy">SUPPORTED = pass <i /> PARTIAL · UNSUPPORTED · CONTRADICTED · UNCERTAIN = fail</div>
        </StorySlide>

        <StorySlide index={8} id="experiments" tone="soft">
          <StoryHeading number="08" overline="Failure-driven development" title="One change. Same 44 cases. One decision." copy="Every experiment is accepted or rejected against a frozen baseline." />
          <div className="experiment-track">
            <Experiment marker="A" title="Scope matching" target="Correct refusal ≥95%" guardrail="False refusal 0–2%" />
            <Experiment marker="B" title="Citation normalization" target="Validity ≥98%" guardrail="No retrieval change" />
            <Experiment marker="C" title="Evidence-only generation" target="Claim support ↑" guardrail="Overreach ↓ · answers useful" />
          </div>
          <div className="decision-loop"><span>Baseline</span><ArrowRight /><span>Change one thing</span><ArrowRight /><span>Run same cases</span><ArrowRight /><span>Keep or revert</span></div>
        </StorySlide>

        <StorySlide index={9} id="demo" tone="ink">
          <div className="final-stage">
            <span className="story-overline">Now show, don’t tell</span>
            <h2>Ask a real question.<br />Open every piece of evidence.</h2>
            <p>The live console exposes the answer, recommendation ID, page, authority, retrieval score, and exact NICE wording.</p>
            <button onClick={onOpenDemo}><Play size={18} /> Open live retrieval</button>
            <div className="final-proof">
              <span><Database />{health?.corpus_chunks ?? 440}<small>chunks</small></span>
              <span><Target />{percent(deterministic?.retrieval_recall_at_5 ?? .973)}<small>blind Recall@5</small></span>
              <span><ShieldCheck />{percent(deterministic?.current_guideline_accuracy ?? 1)}<small>current source</small></span>
              <span><Gauge />{((deterministic?.latency_ms.end_to_end_p50 ?? 3080) / 1000).toFixed(2)}s<small>end-to-end P50</small></span>
            </div>
          </div>
        </StorySlide>
      </main>
    </div>
  );
}

function StorySlide({ index, id, tone = "paper", children }: { index: number; id: string; tone?: "paper" | "ink" | "soft"; children: ReactNode }) {
  return <section id={id} data-story-slide={index} className={`story-slide ${tone}`}>{children}</section>;
}

function StoryHeading({ number, overline, title, copy }: { number: string; overline: string; title: string; copy: string }) {
  return (
    <motion.header className="story-heading" initial={{ y: 35, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true, amount: .65 }}>
      <span>{number} / {overline}</span><h2>{title}</h2><p>{copy}</p>
    </motion.header>
  );
}

function MetricBeat({ value, label }: { value: number; label: string }) {
  return <motion.div className="metric-beat" initial={{ opacity: 0, scale: .9 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }}><strong>{value}</strong><span>{label}</span></motion.div>;
}

function RateRing({ value, label, detail, accent = false }: { value: number; label: string; detail: string; accent?: boolean }) {
  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  return (
    <div className={`rate-ring ${accent ? "accent" : ""}`}>
      <svg viewBox="0 0 170 170" aria-hidden="true"><circle cx="85" cy="85" r={radius} /><motion.circle cx="85" cy="85" r={radius} initial={{ strokeDashoffset: circumference }} whileInView={{ strokeDashoffset: circumference * (1 - value) }} viewport={{ once: true }} transition={{ duration: 1.2 }} style={{ strokeDasharray: circumference }} /></svg>
      <div><span>{label}</span><strong>{percent(value)}</strong><small>{detail}</small></div>
    </div>
  );
}

function FailureBeat({ marker, before, fault, after, metric }: { marker: string; before: string; fault: string; after: string; metric: string }) {
  return (
    <motion.div className="failure-beat" initial={{ x: 55, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={{ once: true, amount: .6 }}>
      <b>{marker}</b><span className="failure-before">{before}</span><ArrowRight /><span className="failure-fault">{fault}</span><ArrowRight /><span className="failure-after">{after}</span><strong>{metric}</strong>
    </motion.div>
  );
}

function Experiment({ marker, title, target, guardrail }: { marker: string; title: string; target: string; guardrail: string }) {
  return <motion.div className="experiment" initial={{ y: 30, opacity: 0 }} whileInView={{ y: 0, opacity: 1 }} viewport={{ once: true }}><b>{marker}</b><span>{title}</span><strong>{target}</strong><small>{guardrail}</small></motion.div>;
}
