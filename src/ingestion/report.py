"""Build the machine-readable and console parsing reports."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from .models import PageRecord, ScopedRecord


def build_parsing_report(
    pdf_path: Path,
    pages: list[PageRecord],
    records: list[ScopedRecord],
    inspection: dict[str, Any],
    scope_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    site_counts = Counter(site for record in records for site in record.cancer_sites)
    section_type_counts = Counter(record.section_type for record in records)
    recommendation_ids = sorted(
        (
            record.recommendation_id
            for record in records
            if record.recommendation_id is not None
        ),
        key=_recommendation_sort_key,
    )
    bullet_lines = sum(
        line.lstrip().startswith(("•", "-"))
        for page in pages
        for line in page.text.splitlines()
    )
    table_info = scope_diagnostics["tables"]

    examples = _select_examples(records)
    return {
        "document": "NICE NG12",
        "guideline_version": "2026",
        "source_file": "ng12.pdf",
        "source_sha256": _sha256(pdf_path),
        "parser_selection": {
            "page_and_prose_parser": f"PyMuPDF {fitz.VersionBind}",
            "table_parser": f"pdfplumber {pdfplumber.__version__}",
            "ocr_used": False,
            "why_selected": (
                "All 101 pages have a usable native text layer and the PDF exposes "
                "bookmark boundaries for headings and recommendation IDs. PyMuPDF "
                "therefore provides the simplest reliable page/prose parser. The symptom "
                "index uses ruled three-column tables; pdfplumber recovered those rows "
                "cleanly and is used only for pages 39-84."
            ),
        },
        "inspection": {
            **inspection,
            "bullets": {
                "readable": bullet_lines > 0,
                "bullet_lines_after_cleaning": bullet_lines,
            },
            "text_order": {
                "prose_reading_order_usable": True,
                "table_reading_order_usable_with_pdfplumber": True,
            },
            "tables": {
                "native_plain_text_is_column-interleaved": True,
                "pdfplumber_tables_detected": table_info["tables_detected"],
                "pdfplumber_rows_seen": table_info["rows_seen"],
                "malformed_rows": table_info["malformed_rows"],
            },
        },
        "summary": {
            "pdf_pages_parsed": len(pages),
            "total_raw_pages": len(pages),
            "total_scoped_records": len(records),
            "records_by_cancer_site": dict(sorted(site_counts.items())),
            "records_by_section_type": dict(sorted(section_type_counts.items())),
            "recommendation_ids_extracted": len(recommendation_ids),
            "recommendation_id_values": recommendation_ids,
            "symptom_table_records_extracted": section_type_counts["symptom_table"],
            "cross_cutting_records_extracted": section_type_counts["cross_cutting"],
        },
        "scope_extraction": scope_diagnostics,
        "difficult_tables": [
            {
                "pages": "40-84",
                "issue": (
                    "Plain-text extraction interleaves the three columns and occasionally "
                    "joins adjacent font spans."
                ),
                "handling": (
                    "Rows and cells are recovered from ruled geometry with pdfplumber; "
                    "word coordinates restore cell reading order and known joined-span "
                    "artifacts are repaired without changing clinical wording."
                ),
            },
            {
                "pages": "53-56",
                "issue": (
                    "Some symptom rows are intentionally repeated under more than one "
                    "alphabetical category in the NICE source."
                ),
                "handling": "Both source occurrences are retained with distinct page provenance.",
            },
        ],
        "pages_requiring_special_handling": [
            {
                "pages": "39-84",
                "reason": "Symptom/findings index; table-aware row extraction required.",
            },
            {
                "pages": "85-88",
                "reason": "Definition boundaries are heading-based rather than recommendation IDs.",
            },
        ],
        "parsing_warnings": [
            (
                "Symptom rows whose Possible cancer cell mixes an in-scope site with an "
                "excluded site are retained, but cancer_sites contains only explicit "
                "in-scope sites; possible_cancers_raw and source_text preserve the full cell."
            ),
            (
                "Broad labels such as 'urogenital' or 'urological cancer' are not expanded "
                "into bladder/renal because that would add specificity not present in the PDF."
            ),
            (
                "Action type, urgency, investigation, and age conditions are deliberately "
                "left null in Phase 1 unless represented directly as a source field; their "
                "wording remains in recommendation_text/source_text for later cleaning."
            ),
            (
                "Typography such as bold and underlining is not represented in JSON, but "
                "headings, list markers, wording, IDs, update markers, and page provenance are retained."
            ),
        ],
        "real_parsed_examples": examples,
    }


def readable_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    parser = report["parser_selection"]
    lines = [
        "NG12 parsing complete",
        f"Pages parsed: {summary['pdf_pages_parsed']}",
        (
            "Parsers: "
            f"{parser['page_and_prose_parser']} (pages/prose); "
            f"{parser['table_parser']} (symptom tables)"
        ),
        f"OCR used: {parser['ocr_used']}",
        f"Scoped records: {summary['total_scoped_records']}",
        f"Records by section type: {summary['records_by_section_type']}",
        f"Records by cancer site: {summary['records_by_cancer_site']}",
        f"Recommendation IDs extracted: {summary['recommendation_ids_extracted']}",
        f"Symptom-table records: {summary['symptom_table_records_extracted']}",
        f"Cross-cutting records: {summary['cross_cutting_records_extracted']}",
        (
            "Excluded sections detected: "
            + ", ".join(
                report["scope_extraction"]["excluded_cancer_sections_detected"]
            )
        ),
        "Examples:",
    ]
    for example in report["real_parsed_examples"]:
        evidence = f"page {example['page']}"
        if example.get("recommendation_id"):
            evidence += f", recommendation {example['recommendation_id']}"
        preview = " ".join(example["source_text"].split())
        if len(preview) > 220:
            preview = preview[:217] + "..."
        lines.append(f"- {example['record_id']} ({evidence}): {preview}")
    return "\n".join(lines)


def _select_examples(records: list[ScopedRecord]) -> list[dict[str, Any]]:
    chosen: list[ScopedRecord] = []
    wanted_ids = [
        "ng12_1.1.2",
        "ng12_1.2.5",
        "ng12_1.3.1",
        "ng12_1.6.6",
        "ng12_1.15.2",
    ]
    by_id = {record.record_id: record for record in records}
    chosen.extend(by_id[item] for item in wanted_ids if item in by_id)

    multi_site = next(
        (
            record
            for record in records
            if record.section_type == "symptom_table" and len(record.cancer_sites) >= 5
        ),
        None,
    )
    if multi_site:
        chosen.append(multi_site)
    term = next((record for record in records if record.term == "Urgent"), None)
    if term:
        chosen.append(term)
    return [record.to_dict() for record in chosen]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recommendation_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))
