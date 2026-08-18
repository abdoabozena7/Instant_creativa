"""Deterministic second-stage reranking for a small auditable candidate set."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .bm25 import tokenize


RERANK_RETRIEVAL_WEIGHT = 0.60
RERANK_QUERY_COVERAGE_WEIGHT = 0.40
CONTENT_INTENT_BOOST = 0.20
EXPLICIT_SITE_MATCH_BOOST = 0.08
EXPLICIT_SITE_MISMATCH_PENALTY = -0.04

# These words describe the request shape rather than the clinical feature. They
# remain available to first-stage BM25; only the late reranker ignores them.
RERANK_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "might", "must", "of", "on", "or", "our", "should", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your", "someone", "somebody",
    "person", "people", "patient", "patients", "year", "years", "old",
    "aged", "age", "over", "under", "now", "previously", "according",
    "guideline", "ng12", "recommend", "recommendation", "recommended",
    "action", "advise", "advised", "consider", "considered", "assess",
    "assessment", "suspected", "cancer", "pathway", "referral",
    "investigate", "investigated", "investigation", "offer", "offered",
    "current", "direct", "access", "evidence",
}
RATIONALE_INTENT_TERMS = {"why", "reason", "rationale", "committee"}
EVIDENCE_INTENT_TERMS = {
    "evidence", "study", "studies", "quality", "bias", "predictive", "ppv",
    "limitation", "limitations",
}


def preferred_content_types(query: str) -> set[str]:
    """Infer only explicit evidence/rationale intent, never a clinical label."""

    query_terms = set(tokenize(query))
    if query_terms & RATIONALE_INTENT_TERMS:
        return {"rationale"}
    if query_terms & EVIDENCE_INTENT_TERMS:
        return {"evidence", "evidence_table"}
    return set()


def salient_query_terms(query: str) -> set[str]:
    return {
        term
        for term in tokenize(query)
        if term not in RERANK_QUERY_STOPWORDS and not term.isdigit()
    }


def idf_query_coverage(
    query_terms: set[str],
    document_terms: Counter[str],
    inverse_document_frequency: Mapping[str, float],
) -> float:
    """Return IDF-weighted coverage of salient query terms in one candidate."""

    denominator = sum(inverse_document_frequency.get(term, 0.0) for term in query_terms)
    if denominator <= 0:
        return 0.0
    numerator = sum(
        inverse_document_frequency.get(term, 0.0)
        for term in query_terms
        if document_terms.get(term, 0) > 0
    )
    return numerator / denominator


def rerank_candidates(
    query: str,
    candidates: Sequence[tuple[float, int, dict[str, Any]]],
    *,
    chunks: Sequence[dict[str, Any]],
    term_frequencies: Sequence[Counter[str]],
    inverse_document_frequency: Mapping[str, float],
    preferred_sites: set[str] | None = None,
) -> list[tuple[float, int, dict[str, Any]]]:
    """Rerank first-stage candidates without accessing evaluation labels."""

    if not candidates:
        return []
    maximum_first_stage_score = max(score for score, _, _ in candidates) or 1.0
    query_terms = salient_query_terms(query)
    content_types = preferred_content_types(query)
    preferred_sites = preferred_sites or set()
    reranked: list[tuple[float, int, dict[str, Any]]] = []

    for first_stage_rank, (score, index, detail) in enumerate(candidates, start=1):
        retrieval_component = score / maximum_first_stage_score
        coverage = idf_query_coverage(
            query_terms,
            term_frequencies[index],
            inverse_document_frequency,
        )
        content_adjustment = (
            CONTENT_INTENT_BOOST
            if content_types and chunks[index]["content_type"] in content_types
            else 0.0
        )
        candidate_sites = set(chunks[index].get("cancer_sites", []))
        if preferred_sites and candidate_sites & preferred_sites:
            site_adjustment = EXPLICIT_SITE_MATCH_BOOST
        elif preferred_sites and candidate_sites:
            site_adjustment = EXPLICIT_SITE_MISMATCH_PENALTY
        else:
            site_adjustment = 0.0
        rerank_score = (
            RERANK_RETRIEVAL_WEIGHT * retrieval_component
            + RERANK_QUERY_COVERAGE_WEIGHT * coverage
            + content_adjustment
            + site_adjustment
        )
        rerank_detail = dict(detail)
        rerank_detail.update(
            {
                "first_stage_rank": first_stage_rank,
                "first_stage_score": round(score, 6),
                "salient_query_coverage": round(coverage, 6),
                "content_intent_adjustment": round(content_adjustment, 6),
                "explicit_site_adjustment": round(site_adjustment, 6),
                "rerank_score": round(rerank_score, 6),
            }
        )
        if content_adjustment:
            rerank_detail["explanations"] = [
                *rerank_detail.get("explanations", []),
                "explicit content-type intent",
            ]
        if site_adjustment > 0:
            rerank_detail["explanations"] = [
                *rerank_detail.get("explanations", []),
                "explicit cancer-site match in reranker",
            ]
        reranked.append((rerank_score, index, rerank_detail))

    reranked.sort(
        key=lambda item: (
            item[0],
            chunks[item[1]]["source_version"] == "2026_current",
            -chunks[item[1]]["page"],
        ),
        reverse=True,
    )
    return reranked
