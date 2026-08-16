"""Parser for the 2015 full NG12 guideline.

The full guideline has a native text layer but a different layout from the
current web guideline: recommendation boxes are unnumbered and evidence/rationale
content is arranged in Word-style tables. This module reuses the same page and
record models while applying layout-aware heading and font rules specific to
that supporting document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .cleaner import join_wrapped_lines
from .models import NormalizedRecord, PageRecord


DOCUMENT_NAME = "NICE NG12"
SOURCE_FILE = "ng12_full_2015.pdf"
SOURCE_VERSION = "2015_full"

TARGET_SITE_TITLES = {
    "7.1 Lung cancer": "lung",
    "8.1 Oesophageal cancer": "oesophageal",
    "8.2 Pancreatic cancer": "pancreatic",
    "8.3 Stomach cancer": "stomach",
    "9.1 Colorectal cancer": "colorectal",
    "12.2 Bladder cancer": "bladder",
    "12.3 Renal cancer": "renal",
}

SHARED_TITLES = {
    "Methodology": "methodology",
    "3 Research recommendations": "research_recommendation",
    "4 Patient information and support": "patient_support",
    "5 Safety netting": "safety_netting",
    "6 The diagnostic process": "diagnostic_process",
    "19 Non-site-specific symptoms": "clinical_context",
}

ACTION_START = re.compile(
    r"^(?:Refer|Offer|Consider|For people|Discuss|Explain|Give|The information|"
    r"Provide|Reassure|When referring|If the person|Ensure|Take part|Discussion|"
    r"Put in place|Include|Use local|Once the decision|Healthcare professionals|Advise)\b",
    re.IGNORECASE,
)
UPDATE_MARKER = re.compile(r"\[(?:(?:new|amended)\s+)?(?:19|20)\d{2}\]", re.IGNORECASE)
TABLE_TITLE = re.compile(r"^Table\s+\d+[:.)]", re.IGNORECASE)
NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)+\s+")

RATIONALE_MARKERS = (
    "Relative value placed on the",
    "Quality of the evidence",
    "Trade-off between clinical",
    "Trade-off between net health",
    "Economic considerations",
    "Other considerations",
)

SITE_PATTERNS = {
    "lung": re.compile(r"\blung\b", re.IGNORECASE),
    "colorectal": re.compile(r"\bcolorectal\b", re.IGNORECASE),
    "oesophageal": re.compile(r"\boesophageal\b|\bgastro-?oesophageal\b", re.IGNORECASE),
    "stomach": re.compile(r"\bstomach\b|\bgastric\b|\bgastro-?oesophageal\b", re.IGNORECASE),
    "pancreatic": re.compile(r"\bpancreatic\b", re.IGNORECASE),
    "bladder": re.compile(r"\bbladder\b", re.IGNORECASE),
    "renal": re.compile(r"\brenal\b", re.IGNORECASE),
}
SITE_ORDER = ["lung", "colorectal", "oesophageal", "stomach", "pancreatic", "bladder", "renal"]


@dataclass(slots=True)
class TextLine:
    page: int
    y: float
    x: float
    text: str
    font_names: tuple[str, ...]
    max_size: float

    @property
    def is_bold(self) -> bool:
        return any("Bold" in font for font in self.font_names)

    @property
    def is_italic(self) -> bool:
        return any("Italic" in font for font in self.font_names)

    @property
    def key(self) -> tuple[int, int, int, str]:
        return (self.page, round(self.y), round(self.x), self.text)


@dataclass(slots=True)
class ScopeSection:
    title: str
    display_title: str
    content_type: str
    page_start: int
    page_end: int
    cancer_sites: list[str]
    heading_path: list[str]


def parse_full_guideline(
    pdf_path: Path,
) -> tuple[list[PageRecord], list[NormalizedRecord], dict[str, Any], list[list[Any]]]:
    document = fitz.open(pdf_path)
    toc = document.get_toc()
    all_page_lines: dict[int, list[TextLine]] = {}
    pages: list[PageRecord] = []
    raw_character_counts: list[int] = []
    header_lines_removed = 0
    footer_lines_removed = 0
    update_markers_removed = 0

    for page_index, pdf_page in enumerate(document):
        page_number = page_index + 1
        raw_character_counts.append(len(pdf_page.get_text("text", sort=True)))
        lines, cleaning = _extract_page_lines(pdf_page, page_number)
        all_page_lines[page_number] = lines
        header_lines_removed += cleaning["header_lines_removed"]
        footer_lines_removed += cleaning["footer_lines_removed"]
        update_markers_removed += cleaning["update_markers_removed"]
        pages.append(
            PageRecord(
                document=DOCUMENT_NAME,
                guideline_code="NG12",
                guideline_version="2015",
                page=page_number,
                text="\n".join(line.text for line in lines),
                source_file=SOURCE_FILE,
            )
        )

    sections = _scope_sections(toc, document.page_count)
    records: list[NormalizedRecord] = []
    section_diagnostics: list[dict[str, Any]] = []
    for section in sections:
        lines = [
            line
            for page_number in range(section.page_start, section.page_end + 1)
            for line in all_page_lines[page_number]
        ]
        section_records, diagnostics = _extract_section_records(section, lines)
        records.extend(section_records)
        section_diagnostics.append(diagnostics)

    document_metadata = dict(document.metadata)
    page_count = document.page_count
    document.close()

    records.sort(key=lambda record: (record.page, record.record_id))
    inspection = {
        "pdf_pages": page_count,
        "native_text_layer": {
            "usable": sum(count > 0 for count in raw_character_counts) >= page_count - 1,
            "pages_with_text": sum(count > 0 for count in raw_character_counts),
            "blank_pages": [
                index + 1 for index, count in enumerate(raw_character_counts) if count == 0
            ],
            "total_characters": sum(raw_character_counts),
            "minimum_characters_on_a_nonblank_page": min(
                count for count in raw_character_counts if count > 0
            ),
            "maximum_characters_on_a_page": max(raw_character_counts),
        },
        "headings": {
            "bookmark_entries": len(toc),
            "usable_for_section_boundaries": len(toc) > 0,
            "recommendation_ids_present": False,
        },
        "cleaning": {
            "header_lines_removed": header_lines_removed,
            "footer_lines_removed": footer_lines_removed,
            "update_markers_removed": update_markers_removed,
        },
        "scoped_sections": section_diagnostics,
        "metadata": document_metadata,
    }
    return pages, records, inspection, toc


def _extract_page_lines(page: Any, page_number: int) -> tuple[list[TextLine], dict[str, int]]:
    extracted: list[TextLine] = []
    counts = {
        "header_lines_removed": 0,
        "footer_lines_removed": 0,
        "update_markers_removed": 0,
    }
    page_height = float(page.rect.height)
    for block in page.get_text("dict", sort=True).get("blocks", []):
        for line in block.get("lines", []):
            y = float(line["bbox"][1])
            x = float(line["bbox"][0])
            if page_number > 1 and y < 52:
                counts["header_lines_removed"] += 1
                continue
            if y > page_height - 52:
                counts["footer_lines_removed"] += 1
                continue

            kept_spans = []
            for span in line.get("spans", []):
                span_text = " ".join(str(span.get("text", "")).split())
                if not span_text:
                    continue
                if span_text in {"Update", "2015", "Update 2015"} and float(span["bbox"][0]) > 525:
                    counts["update_markers_removed"] += 1
                    continue
                kept_spans.append(span)
            if not kept_spans:
                continue

            text = "".join(str(span.get("text", "")) for span in kept_spans)
            text = " ".join(text.split())
            if not text or text == "© National Collaborating Centre for Cancer":
                counts["footer_lines_removed"] += 1
                continue
            if text.startswith("•"):
                text = "• " + text[1:].lstrip()
            extracted.append(
                TextLine(
                    page=page_number,
                    y=y,
                    x=x,
                    text=text,
                    font_names=tuple(str(span.get("font", "")) for span in kept_spans),
                    max_size=max(float(span.get("size", 0)) for span in kept_spans),
                )
            )

    extracted.sort(key=lambda line: (line.y, line.x))
    deduplicated: list[TextLine] = []
    seen: set[tuple[int, int, str]] = set()
    for line in extracted:
        key = (round(line.y), round(line.x), line.text)
        if key not in seen:
            seen.add(key)
            deduplicated.append(line)
    return deduplicated, counts


def _scope_sections(toc: list[list[Any]], page_count: int) -> list[ScopeSection]:
    entries = [(int(level), str(title), int(page)) for level, title, page, *_ in toc]
    parent_stack: dict[int, str] = {}
    sections: list[ScopeSection] = []
    wanted = {**TARGET_SITE_TITLES, **SHARED_TITLES}

    for index, (level, title, page) in enumerate(entries):
        for deeper in [key for key in parent_stack if key >= level]:
            parent_stack.pop(deeper, None)
        parents = [parent_stack[key] for key in sorted(parent_stack)]
        if title in wanted:
            end_page = page_count
            for next_level, _, next_page in entries[index + 1 :]:
                if next_level <= level:
                    end_page = next_page - 1
                    break
            if title in TARGET_SITE_TITLES:
                cancer_sites = [TARGET_SITE_TITLES[title]]
                content_type = "clinical_context"
            else:
                cancer_sites = []
                content_type = SHARED_TITLES[title]
            sections.append(
                ScopeSection(
                    title=title,
                    display_title=re.sub(r"^\d+(?:\.\d+)?\s+", "", title),
                    content_type=content_type,
                    page_start=page,
                    page_end=end_page,
                    cancer_sites=cancer_sites,
                    heading_path=parents + [title],
                )
            )
        parent_stack[level] = title
    return sections


def _extract_section_records(
    section: ScopeSection, lines: list[TextLine]
) -> tuple[list[NormalizedRecord], dict[str, Any]]:
    if section.content_type == "methodology":
        page_records = _methodology_page_records(section, lines)
        return page_records, {
            "section": section.display_title,
            "pages": [section.page_start, section.page_end],
            "records_extracted": len(page_records),
            "historical_recommendations": 0,
            "cancer_sites": [],
        }

    historical_records, historical_keys = _extract_historical_recommendations(section, lines)
    records: list[NormalizedRecord] = []
    mode = section.content_type
    subsection = section.display_title
    buffer: list[TextLine] = []
    ordinal = 0
    references_reached = False

    def finish_buffer() -> None:
        nonlocal buffer, ordinal
        if not buffer:
            return
        if mode == "evidence_table":
            text = "\n".join(line.text for line in buffer)
        else:
            text = join_wrapped_lines(line.text for line in buffer)
        text = text.strip()
        if len(text) < 40:
            buffer = []
            return
        sites = section.cancer_sites or _canonical_sites(text)
        if section.content_type == "research_recommendation" and not sites:
            buffer = []
            return
        ordinal += 1
        records.append(
            NormalizedRecord(
                record_id=(
                    f"ng12_full_{_slug(section.display_title)}_{mode}_"
                    f"p{buffer[0].page:03d}_{ordinal:03d}"
                ),
                document=DOCUMENT_NAME,
                source_file=SOURCE_FILE,
                source_version=SOURCE_VERSION,
                source_type="full_guideline",
                authority_priority="supporting",
                page=buffer[0].page,
                page_end=buffer[-1].page,
                section=section.display_title,
                subsection=subsection,
                cancer_sites=sites,
                content_type=mode,
                text=text,
                source_text=text,
                canonical_recommendation=False,
                heading_path=section.heading_path + ([subsection] if subsection else []),
                metadata={"source_heading": section.title},
            )
        )
        buffer = []

    for line in lines:
        if (
            buffer
            and line.page != buffer[-1].page
            and mode in {"evidence_table", "rationale"}
        ):
            finish_buffer()
        if line.key in historical_keys or line.text == "Recommendations":
            continue
        if references_reached:
            continue
        text = line.text

        if text == "References":
            finish_buffer()
            references_reached = True
            continue
        if text in {section.title, section.display_title} or text in section.heading_path:
            continue
        if text.startswith("Clinical question"):
            finish_buffer()
            mode = "clinical_question"
            subsection = "Clinical question"
        elif text == "Clinical evidence":
            finish_buffer()
            mode = "evidence"
            subsection = "Clinical evidence"
        elif text in {"Evidence statement", "Evidence statements"}:
            finish_buffer()
            mode = "evidence"
            subsection = text
        elif TABLE_TITLE.match(text):
            finish_buffer()
            mode = "evidence_table"
            subsection = text
        elif text == "Cost-effectiveness evidence":
            finish_buffer()
            mode = "evidence"
            subsection = text
        elif text.startswith(RATIONALE_MARKERS):
            finish_buffer()
            mode = "rationale"
            subsection = text
        elif section.content_type == "research_recommendation" and NUMBERED_HEADING.match(text):
            finish_buffer()
            mode = "research_recommendation"
            subsection = text
        elif mode in {"evidence", "rationale"} and _is_supporting_subheading(line):
            finish_buffer()
            subsection = text

        buffer.append(line)
    finish_buffer()
    records.extend(historical_records)
    return records, {
        "section": section.display_title,
        "pages": [section.page_start, section.page_end],
        "records_extracted": len(records),
        "historical_recommendations": len(historical_records),
        "cancer_sites": section.cancer_sites,
    }


def _methodology_page_records(
    section: ScopeSection, lines: list[TextLine]
) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    by_page: dict[int, list[TextLine]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)
    for page_number, page_lines in sorted(by_page.items()):
        text = join_wrapped_lines(
            line.text
            for line in page_lines
            if line.text not in {section.title, section.display_title}
        ).strip()
        if len(text) < 40:
            continue
        heading = next(
            (
                line.text
                for line in page_lines
                if line.is_bold and line.max_size >= 11 and len(line.text.split()) <= 14
            ),
            section.display_title,
        )
        records.append(
            NormalizedRecord(
                record_id=f"ng12_full_methodology_p{page_number:03d}",
                document=DOCUMENT_NAME,
                source_file=SOURCE_FILE,
                source_version=SOURCE_VERSION,
                source_type="full_guideline",
                authority_priority="supporting",
                page=page_number,
                page_end=page_number,
                section=section.display_title,
                subsection=heading,
                cancer_sites=[],
                content_type="methodology",
                text=text,
                source_text=text,
                canonical_recommendation=False,
                heading_path=section.heading_path + [heading],
                metadata={"source_heading": section.title},
            )
        )
    return records


def _extract_historical_recommendations(
    section: ScopeSection, lines: list[TextLine]
) -> tuple[list[NormalizedRecord], set[tuple[int, int, int, str]]]:
    records: list[NormalizedRecord] = []
    used_keys: set[tuple[int, int, int, str]] = set()
    current: list[TextLine] = []
    ordinal = 0

    for original_line in lines:
        candidate = (
            original_line.x >= 250
            and 9.0 <= original_line.max_size <= 10.6
            and original_line.is_bold
            and original_line.text not in {"Update", "2015", "Update 2015"}
        )
        if not candidate:
            if current and original_line.page > current[-1].page + 1:
                current = []
            continue

        used_keys.add(original_line.key)
        for line in _split_update_marked_line(original_line):
            if not current:
                if ACTION_START.match(line.text):
                    current = [line]
                    if UPDATE_MARKER.search(line.text):
                        ordinal += 1
                        records.append(_historical_record(section, current, ordinal))
                        current = []
                continue
            current.append(line)
            combined = " ".join(item.text for item in current)
            if UPDATE_MARKER.search(combined):
                ordinal += 1
                records.append(_historical_record(section, current, ordinal))
                current = []

    if current and len(" ".join(item.text for item in current)) >= 80:
        ordinal += 1
        records.append(_historical_record(section, current, ordinal))
    return records, used_keys


def _historical_record(
    section: ScopeSection, lines: list[TextLine], ordinal: int
) -> NormalizedRecord:
    prepared = [
        ("- " + line.text[2:].strip()) if line.text.startswith("o ") else line.text
        for line in lines
    ]
    text = join_wrapped_lines(prepared)
    sites = section.cancer_sites or _canonical_sites(text)
    return NormalizedRecord(
        record_id=(
            f"ng12_full_{_slug(section.display_title)}_historical_recommendation_"
            f"p{lines[0].page:03d}_{ordinal:02d}"
        ),
        document=DOCUMENT_NAME,
        source_file=SOURCE_FILE,
        source_version=SOURCE_VERSION,
        source_type="full_guideline",
        authority_priority="supporting",
        page=lines[0].page,
        page_end=lines[-1].page,
        section=section.display_title,
        subsection="2015 recommendation text",
        cancer_sites=sites,
        content_type="historical_recommendation",
        text=text,
        source_text=text,
        canonical_recommendation=False,
        heading_path=section.heading_path + ["Recommendations"],
        retrieval_eligible=False,
        metadata={"historical_source_year": 2015, "historical_ordinal": ordinal},
    )


def _split_update_marked_line(line: TextLine) -> list[TextLine]:
    """Split two recommendation sentences when Word emitted them as one line."""

    match = UPDATE_MARKER.search(line.text)
    if not match or not line.text[match.end() :].strip():
        return [line]
    first = line.text[: match.end()].strip()
    remainder = line.text[match.end() :].strip()
    return [
        TextLine(
            page=line.page,
            y=line.y,
            x=line.x,
            text=first,
            font_names=line.font_names,
            max_size=line.max_size,
        ),
        TextLine(
            page=line.page,
            y=line.y + 0.01,
            x=line.x,
            text=remainder,
            font_names=line.font_names,
            max_size=line.max_size,
        ),
    ]


def _is_supporting_subheading(line: TextLine) -> bool:
    text = line.text
    if line.is_italic and 2 <= len(text.split()) <= 9:
        return True
    return text in {
        "Signs and symptoms",
        "Investigations in primary care",
        "Risk of bias in the included studies",
        "Diagnostic accuracy",
    }


def _canonical_sites(text: str) -> list[str]:
    direct = text.replace("Gall bladder", "").replace("gall bladder", "")
    found = {site for site, pattern in SITE_PATTERNS.items() if pattern.search(direct)}
    return [site for site in SITE_ORDER if site in found]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
