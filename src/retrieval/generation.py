"""Grounded answer generation with citation validation."""

from __future__ import annotations

import re
import time
from typing import Any

from .ollama_client import OllamaClient


CITATION_PATTERN = re.compile(r"\[E(\d+)\]")
_CITATION_SPACE = r"[\s\u200b\u200c\u200d\u2060\ufeff]*"
_CITATION_LABEL = rf"\*{{0,2}}{_CITATION_SPACE}E{_CITATION_SPACE}\d+{_CITATION_SPACE}\*{{0,2}}"
_CITATION_GROUP_PATTERN = re.compile(
    rf"[\[【(]{_CITATION_SPACE}{_CITATION_LABEL}"
    rf"(?:{_CITATION_SPACE}[;,/]{_CITATION_SPACE}{_CITATION_LABEL})*"
    rf"{_CITATION_SPACE}[\]】)]",
    re.IGNORECASE,
)
_CITATION_NUMBER = re.compile(rf"E{_CITATION_SPACE}(\d+)", re.IGNORECASE)
SYSTEM_PROMPT = """You are the NG12 Evidence Console, a retrieval-grounded assistant.
Use only the supplied evidence. The 2026 current guideline is authoritative for clinical
actions, thresholds, urgency, and wording. The 2015 full guideline is supporting context
only and must never override a current recommendation.

Rules:
1. Answer the question directly and concisely, normally under 220 words.
2. Cite every clinical claim using one or more evidence labels such as [E1].
3. Clearly distinguish a current recommendation from supporting evidence or rationale.
4. If the evidence is insufficient, say so. Do not fill gaps from memory.
5. Do not diagnose, estimate an individual's risk, or replace professional assessment.
6. Do not cite an evidence label that was not supplied.
7. Preserve ages, thresholds, tests, and urgency wording exactly when mentioned.
8. Do not expand a named referral pathway, investigation, or clinical term with a synonym
   unless that synonym appears in the supplied evidence.
"""


async def generate_grounded_answer(
    query: str,
    results: list[dict[str, Any]],
    ollama: OllamaClient,
) -> dict[str, Any]:
    started = time.perf_counter()
    evidence_blocks = []
    for index, result in enumerate(results, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[E{index}] {result['citation']}",
                    f"Authority: {result['authority_priority']}; content type: {result['content_type']}",
                    result["text"],
                ]
            )
        )
    user_prompt = (
        f"Question: {query}\n\nEvidence:\n\n"
        + "\n\n".join(evidence_blocks)
        + "\n\nAnswer using only this evidence."
    )
    response = await ollama.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    answer = response.get("message", {}).get("content", "").strip()
    answer = normalize_citation_labels(answer)
    citation_validation, warnings = validate_citation_labels(answer, len(results))
    return {
        "answer": answer,
        "model": ollama.chat_model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "citation_validation": citation_validation,
        "warnings": warnings,
        "ollama_metrics": {
            key: response.get(key)
            for key in [
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            ]
            if response.get(key) is not None
        },
    }


def normalize_citation_labels(answer: str) -> str:
    """Canonicalize formatting variants without reconstructing citations from prose."""

    def replace_group(match: re.Match[str]) -> str:
        numbers = _CITATION_NUMBER.findall(match.group(0))
        return " ".join(f"[E{int(value)}]" for value in numbers)

    return _CITATION_GROUP_PATTERN.sub(replace_group, answer)


def validate_citation_labels(
    answer: str, available_evidence_count: int
) -> tuple[dict[str, Any], list[str]]:
    """Validate canonical evidence labels against the supplied evidence set."""

    raw_citations = [int(value) for value in CITATION_PATTERN.findall(answer)]
    valid_citations = sorted(
        {value for value in raw_citations if 1 <= value <= available_evidence_count}
    )
    invalid_citations = sorted(
        {
            value
            for value in raw_citations
            if value < 1 or value > available_evidence_count
        }
    )
    warnings: list[str] = []
    if not valid_citations:
        warnings.append("The generated answer did not include a valid evidence citation.")
    if invalid_citations:
        warnings.append(f"Invalid evidence labels were generated: {invalid_citations}")
    return {
        "passed": bool(valid_citations) and not invalid_citations,
        "cited_evidence_ranks": valid_citations,
        "invalid_evidence_ranks": invalid_citations,
        "available_evidence_count": available_evidence_count,
    }, warnings
