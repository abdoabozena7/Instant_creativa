"""Heading-aware narrow-scope extraction for NG12.

Site recommendations are selected from bookmark/recommendation boundaries. The
symptom index is selected row-by-row from its Possible cancer column.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber

from .cleaner import join_wrapped_lines
from .models import PageRecord, ScopedRecord
from .pdf_parser import DOCUMENT_NAME, GUIDELINE_VERSION


RECOMMENDATION_START = re.compile(r"^(1\.\d+(?:\.\d+)+)\s+(.*)$")
RECOMMENDATION_REF = re.compile(r"\[(1\.\d+(?:\.\d+)+)\]")

SITE_HEADINGS = {
    "Lung cancer": "lung",
    "Oesophageal cancer": "oesophageal",
    "Pancreatic cancer": "pancreatic",
    "Stomach cancer": "stomach",
    "Colorectal cancer": "colorectal",
    "Bladder cancer": "bladder",
    "Renal cancer": "renal",
}

SITE_ORDER = [
    "lung",
    "colorectal",
    "oesophageal",
    "stomach",
    "pancreatic",
    "bladder",
    "renal",
]

NON_SITE_SPECIFIC_IDS = {"1.13.2", "1.13.3", "1.13.4"}
SHARED_PREFIXES = {
    "1.14": "Patient information and support",
    "1.15": "Safety netting",
    "1.16": "The diagnostic process",
}

ALL_TERM_HEADINGS = [
    "Children",
    "Children and young people",
    "Consistent with",
    "Direct access",
    "Hormone replacement therapy (HRT)",
    "Immediate",
    "Non-urgent",
    "Non-specific symptoms pathway",
    "Persistent",
    "Raises the suspicion of",
    "Safety netting",
    "Suspected cancer pathway referral",
    "Unexplained",
    "Unexplained post-menopausal bleeding",
    "Unscheduled vaginal bleeding on HRT",
    "Urgent",
    "Very urgent",
    "Young people",
]

SELECTED_TERMS = {
    "Consistent with",
    "Direct access",
    "Immediate",
    "Non-urgent",
    "Non-specific symptoms pathway",
    "Persistent",
    "Raises the suspicion of",
    "Safety netting",
    "Suspected cancer pathway referral",
    "Unexplained",
    "Urgent",
    "Very urgent",
}

EXCLUDED_SECTION_LABELS = [
    "Mesothelioma",
    "Gall bladder cancer",
    "Liver cancer",
    "Anal cancer",
    "Breast cancer",
    "Gynaecological cancers",
    "Prostate cancer",
    "Testicular cancer",
    "Penile cancer",
    "Skin cancers",
    "Haematological cancers",
]

TABLE_GLUE_REPAIRS = {
    "Rectalmass": "Rectal mass",
    "massconsistent": "mass consistent",
    "signsconsistent": "signs consistent",
}


def extract_scoped_records(
    pdf_path: Path, pages: list[PageRecord], toc: list[list[Any]]
) -> tuple[list[ScopedRecord], dict[str, Any]]:
    recommendation_records, recommendation_diagnostics = _extract_recommendations(
        pages, toc
    )
    referral_guidance = _extract_referral_guidance(pages)
    symptom_records, table_diagnostics = _extract_symptom_tables(pdf_path, toc)
    term_records = _extract_terms(pages)

    records = recommendation_records + symptom_records + term_records
    if referral_guidance is not None:
        records.append(referral_guidance)
    records.sort(key=lambda record: (record.page, record.record_id))

    excluded_detected = [
        label
        for label in EXCLUDED_SECTION_LABELS
        if any(label.casefold() in str(entry[1]).casefold() for entry in toc)
    ]
    diagnostics = {
        "recommendations": recommendation_diagnostics,
        "tables": table_diagnostics,
        "term_definitions_extracted": len(term_records),
        "referral_guidance_extracted": referral_guidance is not None,
        "excluded_cancer_sections_detected": excluded_detected,
    }
    return records, diagnostics


def _extract_recommendations(
    pages: list[PageRecord], toc: list[list[Any]]
) -> tuple[list[ScopedRecord], dict[str, Any]]:
    contexts = _recommendation_contexts(toc)
    heading_titles = {
        _normalise_heading(str(entry[1]))
        for entry in toc
        if 9 <= int(entry[2]) <= 38 and not _is_recommendation_id(str(entry[1]))
    }

    extracted: dict[str, dict[str, Any]] = {}
    current_id: str | None = None
    current_lines: list[str] = []
    current_pages: list[int] = []

    def finish_current() -> None:
        nonlocal current_id, current_lines, current_pages
        if current_id is not None:
            extracted[current_id] = {
                "text": join_wrapped_lines(current_lines),
                "pages": sorted(set(current_pages)),
            }
        current_id = None
        current_lines = []
        current_pages = []

    for page in pages:
        if not 9 <= page.page <= 38:
            continue
        for line in page.text.splitlines():
            match = RECOMMENDATION_START.match(line)
            if match:
                finish_current()
                current_id = match.group(1)
                current_lines = [match.group(2)]
                current_pages = [page.page]
                continue

            if _normalise_heading(line) in heading_titles:
                finish_current()
                continue

            if current_id is not None:
                current_lines.append(line)
                current_pages.append(page.page)
    finish_current()

    records: list[ScopedRecord] = []
    selected_ids: set[str] = set()
    expected_ids: set[str] = set()

    for recommendation_id, context in contexts.items():
        parents: list[str] = context["parents"]
        site_heading = next(
            (heading for heading in reversed(parents) if heading in SITE_HEADINGS), None
        )
        shared_section = next(
            (
                section
                for prefix, section in SHARED_PREFIXES.items()
                if recommendation_id.startswith(prefix + ".")
            ),
            None,
        )
        is_non_site_specific = recommendation_id in NON_SITE_SPECIFIC_IDS

        if not (site_heading or shared_section or is_non_site_specific):
            continue
        expected_ids.add(recommendation_id)
        parsed = extracted.get(recommendation_id)
        if not parsed or not parsed["text"]:
            continue

        text = _repair_table_glue(parsed["text"])
        source_text = f"{recommendation_id} {text}"
        page_numbers = parsed["pages"] or [int(context["page"])]

        if site_heading:
            cancer_sites = [SITE_HEADINGS[site_heading]]
            section_type = "site_recommendation"
            section = site_heading
            subsection = _nearest_numbered_parent(parents)
        elif is_non_site_specific:
            cancer_sites = _canonical_sites(text)
            section_type = "cross_cutting"
            section = "Non-site-specific symptoms"
            subsection = "Symptoms of concern in adults"
        else:
            cancer_sites = []
            section_type = "cross_cutting"
            section = str(shared_section)
            subsection = "Recommendations on patient support, safety netting and the diagnostic process"

        records.append(
            ScopedRecord(
                record_id=f"ng12_{recommendation_id}",
                document=DOCUMENT_NAME,
                guideline_code="NG12",
                guideline_version=GUIDELINE_VERSION,
                page=page_numbers[0],
                page_end=page_numbers[-1],
                section_type=section_type,
                section=section,
                subsection=subsection,
                cancer_sites=cancer_sites,
                recommendation_id=recommendation_id,
                recommendation_text=text,
                source_text=source_text,
                source_file="ng12.pdf",
            )
        )
        selected_ids.add(recommendation_id)

    return records, {
        "expected_scoped_ids": sorted(expected_ids, key=_recommendation_sort_key),
        "extracted_scoped_ids": sorted(selected_ids, key=_recommendation_sort_key),
        "missing_scoped_ids": sorted(expected_ids - selected_ids, key=_recommendation_sort_key),
    }


def _recommendation_contexts(toc: list[list[Any]]) -> dict[str, dict[str, Any]]:
    stack: dict[int, str] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for level, title, page, *_ in toc:
        level = int(level)
        title = str(title)
        for deeper in [key for key in stack if key >= level]:
            stack.pop(deeper, None)
        if _is_recommendation_id(title):
            contexts[title] = {
                "page": int(page),
                "parents": [stack[key] for key in sorted(stack)],
            }
        stack[level] = title
    return contexts


def _extract_referral_guidance(pages: list[PageRecord]) -> ScopedRecord | None:
    page = next((item for item in pages if item.page == 35), None)
    if page is None:
        return None
    lines = page.text.splitlines()
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("Use this guideline to guide referrals.")
        )
        end = lines.index("1.14 Patient information and support")
    except (StopIteration, ValueError):
        return None
    source_text = join_wrapped_lines(lines[start:end])
    return ScopedRecord(
        record_id="ng12_cross_cutting_referral_guidance",
        document=DOCUMENT_NAME,
        guideline_code="NG12",
        guideline_version=GUIDELINE_VERSION,
        page=35,
        page_end=35,
        section_type="cross_cutting",
        section="Referral guidance",
        subsection="Recommendations on patient support, safety netting and the diagnostic process",
        cancer_sites=[],
        source_text=source_text,
        source_file="ng12.pdf",
    )


def _extract_terms(pages: list[PageRecord]) -> list[ScopedRecord]:
    all_headings = set(ALL_TERM_HEADINGS)
    records: list[ScopedRecord] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    current_pages: list[int] = []

    def finish_current() -> None:
        nonlocal current_heading, current_lines, current_pages
        if current_heading in SELECTED_TERMS:
            definition = join_wrapped_lines(current_lines)
            if definition:
                records.append(
                    ScopedRecord(
                        record_id=f"ng12_term_{_slug(current_heading)}",
                        document=DOCUMENT_NAME,
                        guideline_code="NG12",
                        guideline_version=GUIDELINE_VERSION,
                        page=current_pages[0],
                        page_end=current_pages[-1],
                        section_type="cross_cutting",
                        section="Terms used in this guideline",
                        subsection=current_heading,
                        cancer_sites=[],
                        term=current_heading,
                        source_text=f"{current_heading}\n{definition}",
                        source_file="ng12.pdf",
                    )
                )
        current_heading = None
        current_lines = []
        current_pages = []

    for page in pages:
        if not 85 <= page.page <= 88:
            continue
        for line in page.text.splitlines():
            if line in all_headings:
                finish_current()
                current_heading = line
                current_pages = [page.page]
            elif current_heading is not None:
                current_lines.append(line)
                current_pages.append(page.page)
    finish_current()
    return records


def _extract_symptom_tables(
    pdf_path: Path, toc: list[list[Any]]
) -> tuple[list[ScopedRecord], dict[str, Any]]:
    section_titles = {
        _normalise_heading(str(title)): str(title).replace("‑", "-")
        for level, title, page, *_ in toc
        if int(level) == 3 and 39 <= int(page) <= 84
    }
    current_section = "Abdominal symptoms"
    records: list[ScopedRecord] = []
    tables_detected = 0
    rows_seen = 0
    malformed_rows: list[dict[str, int]] = []
    pages_with_scoped_rows: set[int] = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page_number in range(39, 85):
            page = pdf.pages[page_number - 1]
            tables = page.find_tables()
            tables_detected += len(tables)
            markers = _section_markers(page, section_titles)

            for table_index, table in enumerate(tables):
                marker = next(
                    (
                        title
                        for top, title in reversed(markers)
                        if top < float(table.bbox[1])
                    ),
                    None,
                )
                if marker:
                    current_section = marker

                extracted_rows = table.extract()
                table_rows = table.rows
                if not extracted_rows or len(extracted_rows) != len(table_rows):
                    malformed_rows.append(
                        {"page": page_number, "table": table_index, "row": -1}
                    )
                    continue

                for row_index, row in enumerate(table_rows[1:], start=1):
                    rows_seen += 1
                    if len(row.cells) != 3 or any(cell is None for cell in row.cells):
                        malformed_rows.append(
                            {
                                "page": page_number,
                                "table": table_index,
                                "row": row_index,
                            }
                        )
                        continue

                    feature_lines = _cell_lines(page, row.cells[0])
                    cancer_lines = _cell_lines(page, row.cells[1])
                    action_lines = _cell_lines(page, row.cells[2])
                    specific_features = _repair_table_glue(" ".join(feature_lines))
                    possible_cancers = _repair_table_glue(" ".join(cancer_lines))
                    action = _repair_table_glue(join_wrapped_lines(action_lines))
                    cancer_sites = _canonical_sites(possible_cancers)
                    if not cancer_sites:
                        continue

                    symptom = _repair_table_glue(
                        _bold_text_in_cell(page, row.cells[0])
                    )
                    if not symptom:
                        symptom = None
                    source_text = (
                        f"Symptom and specific features: {specific_features}\n"
                        f"Possible cancer: {possible_cancers}\n"
                        f"Actions: {action}"
                    )
                    recommendation_refs = list(
                        dict.fromkeys(RECOMMENDATION_REF.findall(action))
                    )
                    records.append(
                        ScopedRecord(
                            record_id=(
                                f"ng12_symptom_p{page_number:03d}_"
                                f"t{table_index + 1:02d}_r{row_index:02d}"
                            ),
                            document=DOCUMENT_NAME,
                            guideline_code="NG12",
                            guideline_version=GUIDELINE_VERSION,
                            page=page_number,
                            page_end=page_number,
                            section_type="symptom_table",
                            section=current_section,
                            subsection=symptom,
                            cancer_sites=cancer_sites,
                            symptom=symptom,
                            specific_features=specific_features,
                            possible_cancers_raw=possible_cancers,
                            action=action,
                            recommendation_refs=recommendation_refs,
                            symptoms=[symptom] if symptom else None,
                            source_text=source_text,
                            source_file="ng12.pdf",
                            source_table_index=table_index + 1,
                            source_row_index=row_index,
                        )
                    )
                    pages_with_scoped_rows.add(page_number)

            # A new section can start after the final table on a page and then
            # continue on the next page without repeating its title.
            if markers:
                current_section = markers[-1][1]

    site_counts: Counter[str] = Counter(
        site for record in records for site in record.cancer_sites
    )
    return records, {
        "pages_scanned": [39, 84],
        "tables_detected": tables_detected,
        "rows_seen": rows_seen,
        "scoped_rows": len(records),
        "pages_with_scoped_rows": sorted(pages_with_scoped_rows),
        "scoped_rows_by_cancer_site": dict(sorted(site_counts.items())),
        "malformed_rows": malformed_rows,
    }


def _section_markers(
    page: Any, section_titles: dict[str, str]
) -> list[tuple[float, str]]:
    words = page.extract_words(extra_attrs=["fontname", "size"])
    lines = _group_words_by_line(words)
    markers: list[tuple[float, str]] = []
    for top, line_words in lines:
        text = " ".join(str(word["text"]) for word in line_words)
        normalised = _normalise_heading(text)
        if normalised in section_titles and any(
            float(word.get("size", 0)) >= 20 for word in line_words
        ):
            markers.append((top, section_titles[normalised]))
    return markers


def _cell_lines(page: Any, bbox: Any) -> list[str]:
    words = page.within_bbox(bbox).extract_words(extra_attrs=["fontname", "size"])
    return [
        " ".join(str(word["text"]) for word in line_words)
        for _, line_words in _group_words_by_line(words)
    ]


def _bold_text_in_cell(page: Any, bbox: Any) -> str:
    words = page.within_bbox(bbox).extract_words(extra_attrs=["fontname", "size"])
    return " ".join(
        str(word["text"])
        for word in words
        if "SemiBold" in str(word.get("fontname", ""))
    )


def _group_words_by_line(words: list[dict[str, Any]]) -> list[tuple[float, list[dict[str, Any]]]]:
    ordered = sorted(words, key=lambda word: (float(word["top"]), float(word["x0"])))
    groups: list[tuple[float, list[dict[str, Any]]]] = []
    for word in ordered:
        top = float(word["top"])
        if groups and abs(groups[-1][0] - top) <= 2.0:
            groups[-1][1].append(word)
            groups[-1][1].sort(key=lambda item: float(item["x0"]))
        else:
            groups.append((top, [word]))
    return groups


def _canonical_sites(text: str) -> list[str]:
    normalised = _normalise_heading(text).lower()
    # "Gall bladder" is explicitly excluded and must not trigger the broader
    # word-level bladder match.
    direct = normalised.replace("gall bladder", "")
    found: set[str] = set()
    if re.search(r"\blung\b", direct):
        found.add("lung")
    if re.search(r"\bcolorectal\b", direct):
        found.add("colorectal")
    if re.search(r"\boesophageal\b|\bgastro-?oesophageal\b", direct):
        found.add("oesophageal")
    if re.search(r"\bstomach\b|\bgastric\b|\bgastro-?oesophageal\b", direct):
        found.add("stomach")
    if re.search(r"\bpancreatic\b", direct):
        found.add("pancreatic")
    if re.search(r"\bbladder\b", direct):
        found.add("bladder")
    if re.search(r"\brenal\b", direct):
        found.add("renal")
    return [site for site in SITE_ORDER if site in found]


def _repair_table_glue(text: str) -> str:
    repaired = text
    for bad, good in TABLE_GLUE_REPAIRS.items():
        repaired = repaired.replace(bad, good)
    return re.sub(r"[ \t]+", " ", repaired).strip()


def _nearest_numbered_parent(parents: list[str]) -> str | None:
    return next(
        (parent for parent in reversed(parents) if re.match(r"^1\.\d+\s", parent)),
        None,
    )


def _is_recommendation_id(value: str) -> bool:
    return bool(re.fullmatch(r"1\.\d+(?:\.\d+)+", value))


def _normalise_heading(value: str) -> str:
    return " ".join(value.replace("‑", "-").replace("–", "-").split())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _recommendation_sort_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))
