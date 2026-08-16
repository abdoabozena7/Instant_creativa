"""Section-aware chunking after two-source reconciliation."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import tiktoken

from .models import ChunkRecord, NormalizedRecord


ENCODING_NAME = "cl100k_base"
MAX_SUPPORTING_TOKENS = 700
TARGET_SUPPORTING_TOKENS = 550
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9•])")


def build_chunks(
    records: list[NormalizedRecord],
    *,
    max_supporting_tokens: int = MAX_SUPPORTING_TOKENS,
    target_supporting_tokens: int = TARGET_SUPPORTING_TOKENS,
) -> tuple[list[ChunkRecord], dict[str, Any]]:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    before_lengths = [len(encoding.encode(record.text)) for record in records if record.retrieval_eligible]
    chunks: list[ChunkRecord] = []
    oversize_current_units: list[dict[str, Any]] = []

    for record in records:
        if not record.retrieval_eligible:
            continue
        record_tokens = len(encoding.encode(record.text))

        # Current clinical units are already aligned to recommendations, table
        # rows, or definitions. Inspection showed a maximum of 239 tokens, so
        # they remain intact even if a future source revision grows.
        if record.source_type == "current_guideline":
            parts = [record.text]
            if record_tokens > max_supporting_tokens:
                oversize_current_units.append(
                    {"record_id": record.record_id, "token_count": record_tokens}
                )
        elif record_tokens <= max_supporting_tokens:
            parts = [record.text]
        else:
            units = _logical_units(record)
            parts = _pack_units(
                units,
                encoding,
                target_tokens=target_supporting_tokens,
                max_tokens=max_supporting_tokens,
            )

        total = len(parts)
        for index, text in enumerate(parts, start=1):
            token_count = len(encoding.encode(text))
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{record.record_id}_c{index:02d}",
                    record_id=record.record_id,
                    document=record.document,
                    source_file=record.source_file,
                    source_version=record.source_version,
                    source_type=record.source_type,
                    authority_priority=record.authority_priority,
                    page=record.page,
                    page_end=record.page_end or record.page,
                    section=record.section,
                    subsection=record.subsection,
                    cancer_sites=record.cancer_sites,
                    recommendation_id=record.recommendation_id,
                    content_type=record.content_type,
                    text=text,
                    canonical_recommendation=record.canonical_recommendation,
                    related_recommendation_ids=record.related_recommendation_ids,
                    supporting_record_ids=record.supporting_record_ids,
                    chunk_index=index,
                    chunk_count=total,
                    token_count=token_count,
                    token_encoding=ENCODING_NAME,
                )
            )

    after_lengths = [chunk.token_count for chunk in chunks]
    content_counts = Counter(chunk.content_type for chunk in chunks)
    site_counts = Counter(site for chunk in chunks for site in chunk.cancer_sites)
    source_counts = Counter(chunk.source_version for chunk in chunks)
    diagnostics = {
        "strategy": {
            "current_guideline": (
                "One chunk per recommendation, symptom-table row, definition, or shared guidance record."
            ),
            "full_guideline": (
                "Keep short structural records intact; split only records over 700 cl100k tokens. "
                "Evidence tables split on extracted row/line boundaries and narrative evidence/rationale "
                "split on sentence boundaries, packed toward 550 tokens without overlap."
            ),
            "historical_recommendations": "Excluded from chunks after reconciliation; 2026 canonical versions win.",
            "token_encoding": ENCODING_NAME,
            "target_supporting_tokens": target_supporting_tokens,
            "maximum_supporting_tokens": max_supporting_tokens,
        },
        "record_token_distribution_before_chunking": _distribution(before_lengths),
        "chunk_token_distribution": _distribution(after_lengths),
        "chunks_total": len(chunks),
        "chunks_by_source": dict(sorted(source_counts.items())),
        "chunks_by_content_type": dict(sorted(content_counts.items())),
        "chunks_by_cancer_site": dict(sorted(site_counts.items())),
        "oversize_current_logical_units": oversize_current_units,
        "chunks_over_maximum": [
            {"chunk_id": chunk.chunk_id, "token_count": chunk.token_count}
            for chunk in chunks
            if chunk.source_type == "full_guideline"
            and chunk.token_count > max_supporting_tokens
        ],
    }
    return chunks, diagnostics


def _logical_units(record: NormalizedRecord) -> list[str]:
    if record.content_type == "evidence_table":
        units = [line.strip() for line in record.text.splitlines() if line.strip()]
    else:
        paragraphs = [part.strip() for part in record.text.split("\n") if part.strip()]
        units = []
        for paragraph in paragraphs:
            if paragraph.startswith(("•", "-")):
                units.append(paragraph)
            else:
                units.extend(
                    sentence.strip()
                    for sentence in SENTENCE_BOUNDARY.split(paragraph)
                    if sentence.strip()
                )
    return units or [record.text]


def _pack_units(
    units: list[str],
    encoding: Any,
    *,
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    expanded: list[str] = []
    for unit in units:
        token_ids = encoding.encode(unit)
        if len(token_ids) <= max_tokens:
            expanded.append(unit)
            continue
        # This fallback applies only when a single source sentence/table line
        # is itself too large. It is deterministic and emits no overlap.
        for start in range(0, len(token_ids), target_tokens):
            expanded.append(encoding.decode(token_ids[start : start + target_tokens]))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in expanded:
        unit_tokens = len(encoding.encode(unit))
        separator_tokens = 1 if current else 0
        if current and current_tokens + separator_tokens + unit_tokens > max_tokens:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += separator_tokens + unit_tokens
        if current_tokens >= target_tokens:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0
    if current:
        chunks.append("\n".join(current))
    return chunks


def _distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def _percentile(values: list[int], proportion: float) -> int:
    index = min(len(values) - 1, math.ceil(proportion * len(values)) - 1)
    return values[index]
