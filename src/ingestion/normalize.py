"""Normalize current-guideline records into the two-source corpus schema."""

from __future__ import annotations

from .models import NormalizedRecord, ScopedRecord


def normalize_current_records(records: list[ScopedRecord]) -> list[NormalizedRecord]:
    normalized: list[NormalizedRecord] = []
    for record in records:
        content_type = _current_content_type(record)
        structured_metadata = {
            "section_type": record.section_type,
            "symptom": record.symptom,
            "specific_features": record.specific_features,
            "possible_cancers_raw": record.possible_cancers_raw,
            "action": record.action,
            "recommendation_refs": record.recommendation_refs,
            "symptoms": record.symptoms,
            "age_conditions": record.age_conditions,
            "investigation": record.investigation,
            "action_type": record.action_type,
            "urgency": record.urgency,
            "term": record.term,
            "source_table_index": record.source_table_index,
            "source_row_index": record.source_row_index,
        }
        normalized.append(
            NormalizedRecord(
                record_id=record.record_id,
                document=record.document,
                source_file="ng12_current_2026.pdf",
                source_version="2026_current",
                source_type="current_guideline",
                authority_priority="primary",
                page=record.page,
                page_end=record.page_end or record.page,
                section=record.section,
                subsection=record.subsection,
                cancer_sites=record.cancer_sites,
                recommendation_id=record.recommendation_id,
                content_type=content_type,
                text=record.source_text,
                source_text=record.source_text,
                canonical_recommendation=record.recommendation_id is not None,
                heading_path=[
                    item
                    for item in [record.section, record.subsection]
                    if item is not None
                ],
                related_recommendation_ids=record.recommendation_refs,
                retrieval_eligible=True,
                metadata={
                    key: value
                    for key, value in structured_metadata.items()
                    if value is not None and value != []
                },
            )
        )
    return normalized


def _current_content_type(record: ScopedRecord) -> str:
    if record.section_type == "symptom_table":
        return "symptom_table"
    if record.term:
        return "definition"
    if record.recommendation_id:
        if record.recommendation_id.startswith("1.15."):
            return "safety_netting"
        if record.recommendation_id.startswith("1.14."):
            return "patient_support"
        if record.recommendation_id.startswith("1.16."):
            return "diagnostic_process"
        return "recommendation"
    return "guidance"
