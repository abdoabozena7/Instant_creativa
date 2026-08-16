"""Small, dependency-free record models used by the parsing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PageRecord:
    document: str
    page: int
    text: str
    source_file: str
    guideline_version: str
    guideline_code: str = "NG12"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScopedRecord:
    record_id: str
    document: str
    page: int
    section_type: str
    section: str
    cancer_sites: list[str]
    source_text: str
    source_file: str
    guideline_version: str
    guideline_code: str = "NG12"
    page_end: int | None = None
    subsection: str | None = None
    recommendation_id: str | None = None
    recommendation_text: str | None = None
    symptom: str | None = None
    specific_features: str | None = None
    possible_cancers_raw: str | None = None
    action: str | None = None
    recommendation_refs: list[str] = field(default_factory=list)
    symptoms: list[str] | None = None
    age_conditions: list[str] | None = None
    investigation: str | None = None
    action_type: str | None = None
    urgency: str | None = None
    term: str | None = None
    source_table_index: int | None = None
    source_row_index: int | None = None
    in_scope: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NormalizedRecord:
    """Common clean-record schema shared by both NG12 source documents."""

    record_id: str
    document: str
    source_file: str
    source_version: str
    source_type: str
    authority_priority: str
    page: int
    section: str
    cancer_sites: list[str]
    content_type: str
    text: str
    canonical_recommendation: bool
    page_end: int | None = None
    subsection: str | None = None
    recommendation_id: str | None = None
    source_text: str | None = None
    heading_path: list[str] = field(default_factory=list)
    related_recommendation_ids: list[str] = field(default_factory=list)
    supporting_record_ids: list[str] = field(default_factory=list)
    retrieval_eligible: bool = True
    conflict_status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    record_id: str
    document: str
    source_file: str
    source_version: str
    source_type: str
    authority_priority: str
    page: int
    page_end: int
    section: str
    subsection: str | None
    cancer_sites: list[str]
    recommendation_id: str | None
    content_type: str
    text: str
    canonical_recommendation: bool
    related_recommendation_ids: list[str]
    supporting_record_ids: list[str]
    chunk_index: int
    chunk_count: int
    token_count: int
    token_encoding: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
