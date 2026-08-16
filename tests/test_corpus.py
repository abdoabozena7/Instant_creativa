from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
ALLOWED_SITES = {
    "lung",
    "colorectal",
    "oesophageal",
    "stomach",
    "pancreatic",
    "bladder",
    "renal",
}


def _load_jsonl(name: str) -> list[dict]:
    path = PARSED_DIR / name
    assert path.is_file(), f"Run scripts/build_corpus.py first; missing {path}"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_report() -> dict:
    path = PARSED_DIR / "merge_report.json"
    assert path.is_file(), f"Run scripts/build_corpus.py first; missing {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_source_page_sets_are_complete_and_provenanced() -> None:
    current = _load_jsonl("pages_current.jsonl")
    full = _load_jsonl("pages_full.jsonl")
    assert len(current) == 101
    assert len(full) == 382
    assert [row["page"] for row in current] == list(range(1, 102))
    assert [row["page"] for row in full] == list(range(1, 383))
    assert all(row["source_version"] == "2026_current" for row in current)
    assert all(row["authority_priority"] == "primary" for row in current)
    assert all(row["source_version"] == "2015_full" for row in full)
    assert all(row["authority_priority"] == "supporting" for row in full)
    # Page 2 is blank in the native layer; pages 7 and 104 contain only
    # presentation/update furniture that the layout cleaner removes.
    assert {row["page"] for row in full if not row["text"].strip()} == {2, 7, 104}
    assert _load_report()["sources"]["2015_full"]["inspection"]["native_text_layer"][
        "blank_pages"
    ] == [2]


def test_clean_pages_do_not_contain_repeated_footer_noise() -> None:
    current = _load_jsonl("pages_current.jsonl")
    full = _load_jsonl("pages_full.jsonl")
    current_noise = re.compile(r"Page\s+\d+\s+of\s+101|notice-of-rights", re.I)
    full_noise = re.compile(r"©\s*(?:NICE|National Collaborating Centre)", re.I)
    assert not any(current_noise.search(row["text"]) for row in current)
    assert not any(full_noise.search(row["text"]) for row in full)


def test_clean_record_identity_scope_and_report_counts() -> None:
    records = _load_jsonl("records_clean.jsonl")
    chunks = _load_jsonl("chunks.jsonl")
    report = _load_report()
    assert len(records) == report["records"]["retained_after_scope_filtering"]
    assert len(chunks) == report["chunking"]["chunks_total"]
    assert len({row["record_id"] for row in records}) == len(records)
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert all(set(row["cancer_sites"]) <= ALLOWED_SITES for row in records)
    assert Counter(row["source_version"] for row in records) == {
        "2026_current": 122,
        "2015_full": 323,
    }


def test_current_recommendations_are_primary_canonical_units() -> None:
    records = _load_jsonl("records_clean.jsonl")
    current_recommendations = [
        row
        for row in records
        if row["source_version"] == "2026_current"
        and row["content_type"] == "recommendation"
    ]
    assert len(current_recommendations) == 23
    assert all(row["canonical_recommendation"] for row in current_recommendations)
    assert all(row["authority_priority"] == "primary" for row in current_recommendations)
    assert all(row["retrieval_eligible"] for row in current_recommendations)
    assert {row["recommendation_id"] for row in current_recommendations} >= {
        "1.1.1",
        "1.1.2",
        "1.2.1",
        "1.2.5",
        "1.2.9",
        "1.3.1",
        "1.6.4",
        "1.6.6",
    }


def test_historical_recommendations_are_audit_only_and_never_chunked() -> None:
    records = _load_jsonl("records_clean.jsonl")
    chunks = _load_jsonl("chunks.jsonl")
    historical = [row for row in records if row["content_type"] == "historical_recommendation"]
    chunk_record_ids = {row["record_id"] for row in chunks}
    assert len(historical) == 41
    assert all(row["source_version"] == "2015_full" for row in historical)
    assert all(row["authority_priority"] == "supporting" for row in historical)
    assert all(not row["canonical_recommendation"] for row in historical)
    assert all(not row["retrieval_eligible"] for row in historical)
    assert chunk_record_ids.isdisjoint(row["record_id"] for row in historical)


def test_current_logical_units_each_form_one_unchanged_chunk() -> None:
    records = _load_jsonl("records_clean.jsonl")
    chunks = _load_jsonl("chunks.jsonl")
    current_records = {
        row["record_id"]: row for row in records if row["source_version"] == "2026_current"
    }
    current_chunks = [row for row in chunks if row["source_version"] == "2026_current"]
    assert len(current_records) == 122
    assert len(current_chunks) == 122
    by_record = Counter(row["record_id"] for row in current_chunks)
    assert set(by_record.values()) == {1}
    for chunk in current_chunks:
        assert chunk["chunk_count"] == 1
        assert chunk["text"] == current_records[chunk["record_id"]]["text"]


def test_supporting_chunk_limits_and_structural_content() -> None:
    chunks = _load_jsonl("chunks.jsonl")
    report = _load_report()
    full_chunks = [row for row in chunks if row["source_version"] == "2015_full"]
    assert full_chunks
    assert max(row["token_count"] for row in full_chunks) <= 700
    assert report["chunking"]["chunks_over_maximum"] == []
    assert report["chunking"]["oversize_current_logical_units"] == []
    content_types = {row["content_type"] for row in full_chunks}
    assert {"clinical_question", "evidence", "evidence_table", "rationale"} <= content_types
    for site in ALLOWED_SITES:
        site_types = {
            row["content_type"] for row in full_chunks if site in row["cancer_sites"]
        }
        assert {"evidence", "rationale"} <= site_types


def test_multi_site_symptom_rows_and_shared_guidance_survive_merge() -> None:
    records = _load_jsonl("records_clean.jsonl")
    appetite = next(
        row
        for row in records
        if row["content_type"] == "symptom_table"
        and row["metadata"].get("specific_features") == "Appetite loss (unexplained)"
    )
    assert set(appetite["cancer_sites"]) == ALLOWED_SITES
    assert appetite["related_recommendation_ids"]
    assert any(row["content_type"] == "safety_netting" for row in records)
    assert any(row["content_type"] == "diagnostic_process" for row in records)
    assert any(row["content_type"] == "definition" for row in records)


def test_reconciliation_is_explicit_and_links_are_reciprocal() -> None:
    records = _load_jsonl("records_clean.jsonl")
    report = _load_report()
    by_id = {row["record_id"]: row for row in records}
    reconciliation = report["reconciliation"]
    assert reconciliation["duplicates_detected_count"] == 34
    assert reconciliation["conflicts_detected_count"] == 7
    assert all(
        item["resolution"].startswith("Not auto-resolved; 2026 wording remains canonical")
        for item in reconciliation["possible_conflicts"]
    )
    current = by_id["ng12_1.1.2"]
    assert current["supporting_record_ids"]
    for supporting_id in current["supporting_record_ids"]:
        supporting = by_id[supporting_id]
        assert "1.1.2" in supporting["related_recommendation_ids"]
    assert all(example["supporting_record_id"] for example in report["current_to_supporting_examples"])


def test_required_outputs_are_present_without_downstream_rag_artifacts() -> None:
    required = {
        "pages_current.jsonl",
        "pages_full.jsonl",
        "records_clean.jsonl",
        "chunks.jsonl",
        "merge_report.json",
    }
    assert required <= {path.name for path in PARSED_DIR.iterdir()}
    forbidden_suffixes = {".faiss", ".index", ".sqlite", ".db"}
    assert not any(path.suffix.lower() in forbidden_suffixes for path in PROJECT_ROOT.rglob("*"))
