"""Reporting for the reconciled two-source NG12 corpus."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .models import ChunkRecord, NormalizedRecord


def build_merge_report(
    *,
    current_pdf: Path,
    full_pdf: Path,
    current_pages: int,
    full_pages: int,
    current_parsing_report: dict[str, Any],
    full_inspection: dict[str, Any],
    records: list[NormalizedRecord],
    reconciliation: dict[str, Any],
    chunks: list[ChunkRecord],
    chunking: dict[str, Any],
) -> dict[str, Any]:
    records_by_source = Counter(record.source_version for record in records)
    records_by_content = Counter(record.content_type for record in records)
    records_by_site = Counter(site for record in records for site in record.cancer_sites)
    retrieval_records = [record for record in records if record.retrieval_eligible]

    return {
        "document": "NICE NG12",
        "scope": [
            "lung",
            "colorectal",
            "oesophageal",
            "stomach",
            "pancreatic",
            "bladder",
            "renal",
        ],
        "source_priority_rule": {
            "canonical": "2026_current",
            "supporting": "2015_full",
            "rule": (
                "Current 2026 wording is authoritative for clinical recommendations. "
                "The 2015 full guideline supplies supporting evidence and rationale only; "
                "historical recommendation copies are retained for audit but are not chunked."
            ),
        },
        "sources": {
            "2026_current": {
                "source_file": "ng12_current_2026.pdf",
                "source_type": "current_guideline",
                "authority_priority": "primary",
                "sha256": _sha256(current_pdf),
                "pages": current_pages,
                "parser": current_parsing_report["parser_selection"],
                "inspection": current_parsing_report["inspection"],
            },
            "2015_full": {
                "source_file": "ng12_full_2015.pdf",
                "source_type": "full_guideline",
                "authority_priority": "supporting",
                "sha256": _sha256(full_pdf),
                "pages": full_pages,
                "parser": {
                    "page_and_structure_parser": "PyMuPDF 1.26.4",
                    "table_observation_parser": "pdfplumber 0.11.9",
                    "ocr_used": False,
                    "why_selected": (
                        "381 of 382 pages have native text and 69 bookmarks provide reliable "
                        "chapter/site boundaries. Layout-aware PyMuPDF span coordinates remove "
                        "the update sidebar and identify old recommendation boxes. Supporting "
                        "evidence tables retain their original line/row text and table titles."
                    ),
                },
                "inspection": full_inspection,
            },
        },
        "records": {
            "from_each_pdf": dict(sorted(records_by_source.items())),
            "retained_after_scope_filtering": len(records),
            "retrieval_eligible": len(retrieval_records),
            "audit_only_historical_recommendations": sum(
                record.content_type == "historical_recommendation" for record in records
            ),
            "by_content_type": dict(sorted(records_by_content.items())),
            "by_cancer_site": dict(sorted(records_by_site.items())),
        },
        "reconciliation": reconciliation,
        "chunking": chunking,
        "chunk_examples": _chunk_examples(chunks),
        "current_to_supporting_examples": _support_examples(records),
        "parsing_and_merge_problems": [
            {
                "source": "2015_full",
                "issue": (
                    "The full guideline has no recommendation IDs and its Word-style evidence "
                    "tables do not flatten cleanly as plain prose."
                ),
                "handling": (
                    "Bookmarks define site ranges; font/position identifies unnumbered recommendation "
                    "boxes; table titles and extracted line order are retained in evidence_table records."
                ),
            },
            {
                "source": "2015_full",
                "issue": "One front-matter page has no text layer.",
                "handling": "The blank physical page is preserved in pages_full.jsonl; OCR is unnecessary.",
            },
            {
                "source": "both",
                "issue": (
                    "Some 2015 recommendations overlap current recommendations but differ after updates."
                ),
                "handling": (
                    "Historical copies are audit-only, linked where possible, and possible threshold/action "
                    "differences are reported without modifying either source text."
                ),
            },
            {
                "source": "2026_current",
                "issue": (
                    "The symptom index intentionally repeats some actions under multiple symptom categories."
                ),
                "handling": "Each source row remains a separate, cited symptom-table chunk.",
            },
        ],
        "complementarity_assessment": {
            "complement_cleanly": reconciliation["conflicts_detected_count"] == 0,
            "assessment": (
                "The sources complement each other structurally: the 2026 PDF supplies concise canonical "
                "actions, while the full guideline supplies clinical questions, PPV evidence, limitations, "
                "and committee reasoning. Version differences are isolated in the reconciliation report "
                "rather than blended into canonical text."
            ),
        },
    }


def readable_merge_summary(report: dict[str, Any]) -> str:
    records = report["records"]
    reconciliation = report["reconciliation"]
    chunking = report["chunking"]
    return "\n".join(
        [
            "NG12 two-source corpus complete",
            f"Pages: current={report['sources']['2026_current']['pages']}, full={report['sources']['2015_full']['pages']}",
            f"Scoped records: {records['retained_after_scope_filtering']} ({records['from_each_pdf']})",
            f"Retrieval-eligible records: {records['retrieval_eligible']}",
            f"Historical recommendation records (audit only): {records['audit_only_historical_recommendations']}",
            f"Duplicates detected: {reconciliation['duplicates_detected_count']}",
            f"Possible conflicts/version differences: {reconciliation['conflicts_detected_count']}",
            f"Chunks: {chunking['chunks_total']} ({chunking['chunks_by_source']})",
            f"Chunk token distribution: {chunking['chunk_token_distribution']}",
            f"Chunks by content type: {chunking['chunks_by_content_type']}",
            f"Chunks by cancer site: {chunking['chunks_by_cancer_site']}",
        ]
    )


def _chunk_examples(chunks: list[ChunkRecord]) -> list[dict[str, Any]]:
    selectors = [
        lambda chunk: chunk.recommendation_id == "1.1.2",
        lambda chunk: chunk.content_type == "symptom_table" and len(chunk.cancer_sites) >= 5,
        lambda chunk: chunk.source_version == "2015_full" and chunk.content_type == "evidence",
        lambda chunk: chunk.source_version == "2015_full" and chunk.content_type == "rationale",
        lambda chunk: chunk.source_version == "2015_full" and chunk.content_type == "evidence_table",
    ]
    examples: list[dict[str, Any]] = []
    for selector in selectors:
        chunk = next((item for item in chunks if selector(item)), None)
        if chunk is None:
            continue
        payload = chunk.to_dict()
        if len(payload["text"]) > 900:
            payload["text"] = payload["text"][:897] + "..."
        examples.append(payload)
    return examples


def _support_examples(records: list[NormalizedRecord]) -> list[dict[str, Any]]:
    by_id = {record.record_id: record for record in records}
    examples: list[dict[str, Any]] = []
    for recommendation_id in ["1.1.2", "1.2.5", "1.3.1", "1.6.6"]:
        current = next(
            (
                record
                for record in records
                if record.recommendation_id == recommendation_id
                and record.source_version == "2026_current"
            ),
            None,
        )
        if current is None:
            continue
        candidates = [
            by_id[record_id]
            for record_id in current.supporting_record_ids
            if record_id in by_id
            and by_id[record_id].content_type in {"evidence", "rationale", "clinical_question"}
        ]
        content_priority = {"evidence": 0, "rationale": 1, "clinical_question": 2}
        candidates.sort(
            key=lambda record: (content_priority[record.content_type], record.page, record.record_id)
        )
        supporting = candidates[0] if candidates else None
        examples.append(
            {
                "recommendation_id": recommendation_id,
                "canonical_record_id": current.record_id,
                "canonical_page": current.page,
                "supporting_record_id": supporting.record_id if supporting else None,
                "supporting_page": supporting.page if supporting else None,
                "supporting_content_type": supporting.content_type if supporting else None,
                "supporting_preview": (
                    " ".join(supporting.text.split())[:500] if supporting else None
                ),
            }
        )
    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
