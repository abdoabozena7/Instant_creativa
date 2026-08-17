"""Grounded answer generation with citation validation."""

from __future__ import annotations

import re
import time
from typing import Any

from .ollama_client import OllamaClient
from .query_safety import assess_query_safety
from .scope_guard import assess_query_answerability


CITATION_PATTERN = re.compile(r"\[E(\d+)\]")
_CITATION_SPACE = r"[\s\u200b\u200c\u200d\u2060\ufeff]*"
_CITATION_LABEL = rf"\*{{0,2}}{_CITATION_SPACE}E{_CITATION_SPACE}\d+{_CITATION_SPACE}\*{{0,2}}"
_CITATION_GROUP_PATTERN = re.compile(
    rf"[\[【(]{_CITATION_SPACE}(?:Evidence{_CITATION_SPACE})?{_CITATION_LABEL}"
    rf"(?:{_CITATION_SPACE}[;,/]{_CITATION_SPACE}{_CITATION_LABEL})*"
    rf"{_CITATION_SPACE}[\]】)]",
    re.IGNORECASE,
)
_CITATION_NUMBER = re.compile(rf"E{_CITATION_SPACE}(\d+)", re.IGNORECASE)
_LINE_REFERENCE_CITATION = re.compile(
    rf"[【\[]{_CITATION_SPACE}E{_CITATION_SPACE}(\d+)"
    rf"{_CITATION_SPACE}†{_CITATION_SPACE}L\d+"
    rf"(?:{_CITATION_SPACE}-{_CITATION_SPACE}L?\d+)?"
    rf"{_CITATION_SPACE}[】\]]",
    re.IGNORECASE,
)
_STANDALONE_EVIDENCE_LABEL = re.compile(
    rf"(?<![\w\[])\*{{0,2}}E{_CITATION_SPACE}(\d+)\*{{0,2}}"
    rf"(?={_CITATION_SPACE}[–—:-])",
    re.IGNORECASE,
)
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
9. Treat every patient attribute not stated in the question—including age, symptoms,
   duration, smoking history, and test results—as unknown. Never copy an eligibility
   condition from evidence and present it as a fact about the patient.
10. For a patient-specific referral, investigation, or action question, do not begin with
   Yes or No unless the question supplies every qualifier needed by the cited criterion.
   If qualifiers are missing, state that the information is insufficient, list what is
   missing, and describe the guideline only conditionally.
11. Preserve recommendation modality exactly. In particular, do not strengthen "consider"
   into "must", "arrange", or an unconditional recommendation.
12. Do not add after-referral information or patient-support guidance unless it answers the
   question directly.
13. For a general or underspecified clinical-feature question, state only the applicable
   current criteria and the qualifiers needed to use them. Do not add rationale, quantitative
   historical evidence, or unrelated workflow guidance unless the question asks for it.
14. Preserve AND/OR structure exactly. When the evidence says "any of the following", one
   listed feature is sufficient; do not imply that additional listed features are required.
"""


async def generate_grounded_answer(
    query: str,
    results: list[dict[str, Any]],
    ollama: OllamaClient,
) -> dict[str, Any]:
    started = time.perf_counter()
    safety = assess_query_safety(query)
    if safety["status"] == "blocked":
        return {
            "answer": safety["message"],
            "model": None,
            "response_status": "safety_refusal",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety": safety,
            "answerability": {
                "status": "not_assessed",
                "clinical_features": [],
            },
            "citation_validation": {
                "applicable": False,
                "passed": None,
                "cited_evidence_ranks": [],
                "invalid_evidence_ranks": [],
                "available_evidence_count": 0,
            },
            "warnings": [
                "Generation skipped by the deterministic instruction-safety guard."
            ],
            "ollama_metrics": {},
        }
    answerability = assess_query_answerability(query)
    if answerability["status"] == "insufficient":
        return {
            "answer": answerability["message"],
            "model": None,
            "response_status": "insufficient_information",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety": safety,
            "answerability": answerability,
            "citation_validation": {
                "applicable": False,
                "passed": None,
                "cited_evidence_ranks": [],
                "invalid_evidence_ranks": [],
                "available_evidence_count": 0,
            },
            "warnings": [
                "Generation skipped because the patient assessment contained no concrete clinical feature."
            ],
            "ollama_metrics": {},
        }
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
        + (
            "\n\nAnswer using only this evidence. Patient attributes absent from the "
            "question are unknown; do not infer them from eligibility criteria in the evidence. "
            "Preserve the recommendation's exact modality: 'consider' must remain 'consider'."
        )
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
    response_status = (
        "grounded_answer" if citation_validation["passed"] else "generation_rejected"
    )
    if response_status == "generation_rejected":
        answer = (
            "The model response was withheld because it did not satisfy the "
            "evidence-citation contract. Please rephrase the clinical question."
        )
        warnings.append("Ungrounded model output was not returned to the user.")
    return {
        "answer": answer,
        "model": ollama.chat_model,
        "response_status": response_status,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "safety": safety,
        "answerability": answerability,
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

    answer = _LINE_REFERENCE_CITATION.sub(
        lambda match: f"[E{int(match.group(1))}]", answer
    )
    answer = _STANDALONE_EVIDENCE_LABEL.sub(
        lambda match: f"[E{int(match.group(1))}]", answer
    )
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
        "applicable": True,
        "passed": bool(valid_citations) and not invalid_citations,
        "cited_evidence_ranks": valid_citations,
        "invalid_evidence_ranks": invalid_citations,
        "available_evidence_count": available_evidence_count,
    }, warnings
