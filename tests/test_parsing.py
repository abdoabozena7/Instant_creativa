from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"


def _load_jsonl(name: str) -> list[dict]:
    path = PARSED_DIR / name
    assert path.is_file(), f"Run scripts/parse_ng12.py first; missing {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_all_physical_pages_are_present_and_ordered() -> None:
    pages = _load_jsonl("pages.jsonl")
    assert len(pages) == 101
    assert [page["page"] for page in pages] == list(range(1, 102))
    assert all(page["document"] == "NICE NG12" for page in pages)
    assert all(page["guideline_version"] == "2026" for page in pages)
    assert "1.1.1" in pages[8]["text"]
    assert "ISBN:" in pages[100]["text"]


def test_repeated_headers_and_footers_are_removed() -> None:
    pages = _load_jsonl("pages.jsonl")
    pollution = re.compile(r"© NICE 2026|Page\s+\d+\s+of\s+101|notice-of-rights")
    assert not any(pollution.search(page["text"]) for page in pages)
    repeated_header = "Suspected cancer: recognition and referral (NG12)"
    assert all(repeated_header not in page["text"] for page in pages[1:])


def test_site_recommendations_match_the_narrow_scope() -> None:
    records = _load_jsonl("scoped_records.jsonl")
    site_records = [
        record for record in records if record["section_type"] == "site_recommendation"
    ]
    expected = {
        "lung": {"1.1.1", "1.1.2", "1.1.3"},
        "oesophageal": {"1.2.1", "1.2.2", "1.2.3"},
        "pancreatic": {"1.2.4", "1.2.5"},
        "stomach": {"1.2.6", "1.2.7", "1.2.8", "1.2.9"},
        "colorectal": {"1.3.1", "1.3.2", "1.3.3", "1.3.4", "1.3.5"},
        "bladder": {"1.6.4", "1.6.5"},
        "renal": {"1.6.6"},
    }
    for site, ids in expected.items():
        actual = {
            record["recommendation_id"]
            for record in site_records
            if site in record["cancer_sites"]
        }
        assert actual == ids


def test_excluded_site_recommendations_are_absent() -> None:
    records = _load_jsonl("scoped_records.jsonl")
    excluded_ids = {
        "1.1.4",
        "1.1.5",
        "1.1.6",
        "1.2.10",
        "1.2.11",
        "1.3.6",
        "1.6.1",
        "1.6.2",
        "1.6.3",
        "1.6.7",
        "1.6.8",
        "1.6.9",
        "1.6.10",
    }
    ids = {record["recommendation_id"] for record in records}
    assert ids.isdisjoint(excluded_ids)
    allowed_sites = {
        "lung",
        "oesophageal",
        "stomach",
        "pancreatic",
        "colorectal",
        "bladder",
        "renal",
    }
    assert all(set(record["cancer_sites"]) <= allowed_sites for record in records)


def test_recommendation_boundaries_pages_and_lists_are_intact() -> None:
    records = _load_jsonl("scoped_records.jsonl")
    by_id = {record["recommendation_id"]: record for record in records}
    lung = by_id["1.1.2"]
    assert lung["page"] == 9
    assert lung["page_end"] == 9
    assert "aged 40 and over" in lung["recommendation_text"]
    assert lung["recommendation_text"].count("•") >= 2
    assert "[2015]" in lung["recommendation_text"]
    assert "Mesothelioma" not in lung["recommendation_text"]
    colorectal = by_id["1.3.1"]
    assert colorectal["page"] == 14
    assert colorectal["page_end"] == 15
    assert "faecal immunochemical testing" in colorectal["recommendation_text"]


def test_symptom_tables_retain_multi_site_relationships() -> None:
    records = _load_jsonl("scoped_records.jsonl")
    symptom_records = [
        record for record in records if record["section_type"] == "symptom_table"
    ]
    assert len(symptom_records) >= 60
    appetite = next(
        record
        for record in symptom_records
        if record["specific_features"] == "Appetite loss (unexplained)"
    )
    assert set(appetite["cancer_sites"]) == {
        "lung",
        "oesophageal",
        "stomach",
        "pancreatic",
        "colorectal",
        "bladder",
        "renal",
    }
    haematuria = next(
        record
        for record in symptom_records
        if record["page"] == 61 and set(record["cancer_sites"]) == {"bladder", "renal"}
    )
    assert haematuria["recommendation_refs"] == ["1.6.4", "1.6.6"]
    assert haematuria["section"] == "Urological symptoms"
    assert all("massconsistent" not in record["source_text"] for record in symptom_records)
    assert all("Rectalmass" not in record["source_text"] for record in symptom_records)


def test_cross_cutting_safety_netting_and_terms_exist() -> None:
    records = _load_jsonl("scoped_records.jsonl")
    by_id = {record["recommendation_id"]: record for record in records}
    assert "1.15.1" in by_id and "1.15.2" in by_id
    assert by_id["1.15.2"]["section_type"] == "cross_cutting"
    assert by_id["1.15.2"]["cancer_sites"] == []
    terms = {record["term"] for record in records if record["term"]}
    assert {
        "Direct access",
        "Persistent",
        "Safety netting",
        "Suspected cancer pathway referral",
        "Unexplained",
        "Urgent",
    } <= terms


def test_report_matches_outputs_and_has_no_parser_failures() -> None:
    pages = _load_jsonl("pages.jsonl")
    records = _load_jsonl("scoped_records.jsonl")
    report = json.loads((PARSED_DIR / "parsing_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["total_raw_pages"] == len(pages)
    assert report["summary"]["total_scoped_records"] == len(records)
    assert report["parser_selection"]["ocr_used"] is False
    assert report["inspection"]["native_text_layer"]["usable"] is True
    assert report["scope_extraction"]["recommendations"]["missing_scoped_ids"] == []
    assert report["scope_extraction"]["tables"]["malformed_rows"] == []
