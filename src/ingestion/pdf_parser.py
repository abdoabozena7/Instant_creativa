"""Faithful page-level parsing and inspection for the NG12 PDF."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from .cleaner import clean_page_text
from .models import PageRecord


DOCUMENT_NAME = "NICE NG12"
GUIDELINE_VERSION = "2026"
PAGE_LABEL = re.compile(r"Page\s+(\d+)\s+of\s+101", re.IGNORECASE | re.DOTALL)
RECOMMENDATION_ID = re.compile(r"(?m)^\s*(1\.\d+(?:\.\d+)+)\b")


def parse_pdf_pages(pdf_path: Path) -> tuple[list[PageRecord], dict[str, Any], list[list[Any]]]:
    """Parse every page using the native text layer and retain physical pages."""

    document = fitz.open(pdf_path)
    pages: list[PageRecord] = []
    clean_counts: Counter[str] = Counter()
    raw_character_counts: list[int] = []
    printed_labels_found = 0
    printed_label_mismatches: list[dict[str, int]] = []
    raw_recommendation_ids: set[str] = set()

    for page_index, pdf_page in enumerate(document):
        page_number = page_index + 1
        raw_text = pdf_page.get_text("text", sort=True)
        raw_character_counts.append(len(raw_text))
        raw_recommendation_ids.update(RECOMMENDATION_ID.findall(raw_text))

        label_match = PAGE_LABEL.search(" ".join(raw_text.split()))
        if label_match:
            printed_labels_found += 1
            printed_page = int(label_match.group(1))
            if printed_page != page_number:
                printed_label_mismatches.append(
                    {"pdf_page": page_number, "printed_page": printed_page}
                )

        cleaned_text, counts = clean_page_text(raw_text, page_number)
        clean_counts.update(counts)
        pages.append(
            PageRecord(
                document=DOCUMENT_NAME,
                guideline_code="NG12",
                guideline_version=GUIDELINE_VERSION,
                page=page_number,
                text=cleaned_text,
                source_file="ng12.pdf",
            )
        )

    toc = document.get_toc()
    metadata = dict(document.metadata)
    page_count = document.page_count
    document.close()

    inspection = {
        "pdf_pages": page_count,
        "native_text_layer": {
            "usable": all(count > 0 for count in raw_character_counts),
            "pages_with_text": sum(count > 0 for count in raw_character_counts),
            "total_characters": sum(raw_character_counts),
            "minimum_characters_on_a_page": min(raw_character_counts),
            "maximum_characters_on_a_page": max(raw_character_counts),
        },
        "headings": {
            "bookmark_entries": len(toc),
            "usable_for_boundaries": len(toc) > 0,
        },
        "recommendation_ids": {
            "unique_native_ids": len(raw_recommendation_ids),
            "examples": sorted(raw_recommendation_ids, key=_recommendation_sort_key)[:12],
        },
        "page_numbering": {
            "physical_pages_preserved": True,
            "printed_labels_found": printed_labels_found,
            "printed_label_mismatches": printed_label_mismatches,
        },
        "cleaning": dict(clean_counts),
        "metadata": metadata,
    }
    return pages, inspection, toc


def _recommendation_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))
