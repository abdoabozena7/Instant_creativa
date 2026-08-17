export type Mode = "hybrid" | "bm25" | "dense";
export type QueryOutcome =
  | "retrieval_results"
  | "grounded_answer"
  | "safety_refusal"
  | "scope_refusal"
  | "insufficient_information"
  | "generation_rejected"
  | "no_results";

export interface QuerySafety {
  status: "allowed" | "blocked";
  reason_codes: string[];
  message: string | null;
}

export interface ScoreDetail {
  base_score: number;
  bm25_score: number;
  dense_score: number | null;
  authority_adjustment: number;
  explanations: string[];
}

export interface EvidenceResult {
  chunk_id: string;
  record_id: string;
  source_version: "2026_current" | "2015_full";
  source_type: string;
  authority_priority: "primary" | "supporting";
  page: number;
  page_end: number;
  section: string;
  subsection: string | null;
  cancer_sites: string[];
  recommendation_id: string | null;
  content_type: string;
  text: string;
  rank: number;
  score: number;
  score_detail: ScoreDetail;
  citation: string;
  token_count: number;
}

export interface SearchResponse {
  query: string;
  outcome: QueryOutcome;
  mode_requested: Mode;
  mode_used: string;
  safety: QuerySafety;
  scope: {
    status: "in_scope" | "out_of_scope" | "not_assessed";
    selected_sites: string[];
    excluded_sites: string[];
    message: string | null;
  };
  answerability: {
    status: "model_assessed" | "insufficient" | "not_assessed";
    clinical_features: string[];
    message?: string;
  };
  results: EvidenceResult[];
  latency_ms: number;
  warnings: string[];
}

export interface AnswerResponse {
  query: string;
  outcome: QueryOutcome;
  answer: string;
  model: string | null;
  latency_ms: number;
  retrieval: SearchResponse;
  citation_validation: {
    applicable: boolean;
    passed: boolean | null;
    cited_evidence_ranks: number[];
    invalid_evidence_ranks: number[];
    available_evidence_count: number;
  };
  warnings: string[];
  safety_note?: string;
  ollama_metrics?: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  corpus_chunks: number;
  dense_index_ready: boolean;
  embedding_model: string;
  chat_model: string;
  ollama: {
    available: boolean;
    embedding_model_ready?: boolean;
    chat_model_ready?: boolean;
  };
}

export interface MetricsResponse {
  corpus: {
    records: {
      retained_after_scope_filtering: number;
      retrieval_eligible: number;
      audit_only_historical_recommendations: number;
      by_content_type: Record<string, number>;
      by_cancer_site: Record<string, number>;
    };
    reconciliation: {
      duplicates_detected_count: number;
      conflicts_detected_count: number;
      unmatched_historical_recommendations: number;
    };
    chunking: {
      chunks_total: number;
      chunk_token_distribution: Record<string, number>;
      chunks_by_content_type: Record<string, number>;
      chunks_by_cancer_site: Record<string, number>;
      chunks_over_maximum: unknown[];
    };
  };
  evaluation: {
    evaluation_set: { queries: number; by_category: Record<string, number> };
    embedding_model: string;
    recommended_mode: Mode;
    selection_rule: string;
    modes: Record<Mode, {
      recall_at_1: number;
      recall_at_3: number;
      recall_at_5: number;
      mrr_at_10: number;
      canonical_top1_accuracy: number;
      out_of_scope_refusal_accuracy: number;
      latency_ms: { p50: number; p95: number };
      recall_at_5_by_category: Record<string, number>;
    }>; 
  } | null;
  blind_e2e: {
    evaluation_name: string;
    architecture_sha256: string;
    architecture_frozen_at: string;
    questions: {
      total: number;
      by_scope_group: Record<string, number>;
      by_category: Record<string, number>;
      by_expected_behavior: Record<string, number>;
    };
    deterministic_metrics: {
      scope_classification_accuracy: number;
      correct_refusal_rate: number;
      false_refusal_rate: number;
      retrieval_queries_scored: number;
      retrieval_recall_at_1: number;
      retrieval_recall_at_3: number;
      retrieval_recall_at_5: number;
      retrieval_mrr_at_6: number;
      current_guideline_accuracy: number;
      citation_label_validity_rate: number;
      latency_ms: {
        end_to_end_p50: number;
        end_to_end_p95: number;
        generation_p50: number;
        generation_p95: number;
      };
    };
    semantic_metrics: {
      status: string;
      citation_accuracy: number | null;
      claim_support_rate: number | null;
      unsupported_claim_rate: number | null;
      answer_behavior_accuracy: number | null;
      note: string;
      provisional_model_assisted?: {
        status: string;
        judge_model: string;
        independence_limitation: string;
        cases: number;
        claims: number;
        citation_accuracy_provisional: number | null;
        claim_support_rate_provisional: number | null;
        unsupported_claim_rate_provisional: number | null;
        answer_behavior_accuracy_provisional: number | null;
        current_guideline_accuracy_provisional: number | null;
        failure_type_counts: Record<string, number>;
        human_review_queue: Array<{ case_id: string }>;
      };
    };
    failures: Record<string, Array<{ case_id: string }>>;
    by_scope_group: Record<string, {
      cases: number;
      scope_accuracy: number;
      retrieval_recall_at_5: number | null;
    }>;
  } | null;
  multi_judge: {
    status: string;
    evaluation_name: string;
    architecture_sha256: string;
    disclaimer: string;
    judges: Record<string, { model: string; passes: number }>;
    counts: {
      llm_tasks: number;
      deterministic_missing_citation_tasks: number;
      model_calls: number;
      consensus_decisions: number;
    };
    metrics: {
      judge_agreement_rate: number;
      claim_support_rate: number;
      unsupported_claim_rate: number;
      citation_entailment_rate: number;
      citation_coverage_rate: number;
      overreach_free_rate: number;
      answer_completeness_rate: number;
      refusal_quality_rate: number | null;
    };
    failure_counts_by_criterion: Record<string, number>;
    disagreements: Array<{
      task_id: string;
      criterion: string;
      case_id: string;
      claim_id: string | null;
    }>;
  } | null;
  runtime: {
    searches: number;
    answers: number;
    scope_refusals: number;
    safety_refusals: number;
    citation_validation_pass_rate: number | null;
    search_latency: { count: number; p50_ms: number; p95_ms: number };
    answer_latency: { count: number; p50_ms: number; p95_ms: number };
  };
}
