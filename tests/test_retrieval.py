from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.retrieval.engine import RetrievalEngine
from src.retrieval.generation import (
    CITATION_PATTERN,
    generate_grounded_answer,
    normalize_citation_labels,
    validate_citation_labels,
)
from src.retrieval.scope_guard import assess_scope


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dense_index_matches_the_chunk_corpus() -> None:
    chunks = (PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    embeddings = np.load(PROJECT_ROOT / "data" / "index" / "chunk_embeddings.npy")
    manifest = json.loads(
        (PROJECT_ROOT / "data" / "index" / "index_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert embeddings.shape == (len(chunks), 768)
    assert manifest["rows"] == len(chunks)
    assert manifest["normalized"] is True
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)


def test_retrieval_fails_loudly_when_dense_index_is_stale(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "chunk_embeddings.npy"
    manifest = tmp_path / "index_manifest.json"
    shutil.copy2(PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl", chunks)
    shutil.copy2(PROJECT_ROOT / "data" / "index" / "chunk_embeddings.npy", embeddings)
    shutil.copy2(PROJECT_ROOT / "data" / "index" / "index_manifest.json", manifest)
    chunks.write_text(
        chunks.read_text(encoding="utf-8").replace("NICE NG12", "STALE NG12", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the chunk corpus"):
        RetrievalEngine(chunks_path=chunks, embeddings_path=embeddings)


def test_scope_guard_refuses_excluded_only_queries() -> None:
    prostate = assess_scope("What PSA result triggers prostate cancer referral?")
    assert prostate["status"] == "out_of_scope"
    assert prostate["excluded_sites"] == ["prostate cancer"]
    mixed = assess_scope("Compare lung cancer and mesothelioma referral")
    assert mixed["status"] == "in_scope"
    assert mixed["selected_sites"] == ["lung"]
    assert mixed["excluded_sites"] == ["mesothelioma"]

    gall_bladder = assess_scope("When should gall bladder cancer be referred?")
    assert gall_bladder["status"] == "out_of_scope"
    assert gall_bladder["selected_sites"] == []
    assert gall_bladder["excluded_sites"] == ["gall bladder cancer"]

    bladder_mixed = assess_scope("Compare bladder cancer with gall bladder cancer")
    assert bladder_mixed["status"] == "in_scope"
    assert bladder_mixed["selected_sites"] == ["bladder"]
    assert bladder_mixed["excluded_sites"] == ["gall bladder cancer"]


def test_bm25_returns_the_canonical_current_recommendation_first() -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search(
            "FIT result at least 10 micrograms haemoglobin per gram colorectal referral",
            mode="bm25",
            top_k=5,
        )
    )
    top = response["results"][0]
    assert top["recommendation_id"] == "1.3.2"
    assert top["source_version"] == "2026_current"
    assert top["authority_priority"] == "primary"
    assert "canonical current recommendation" in top["score_detail"]["explanations"]


def test_out_of_scope_query_never_reaches_retrieval_results() -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search("When should breast cancer be referred?", mode="hybrid")
    )
    assert response["mode_used"] == "scope_guard"
    assert response["results"] == []


def test_citation_pattern_accepts_canonical_evidence_labels() -> None:
    assert CITATION_PATTERN.findall("Referral is recommended [E1] and supported [E3].") == [
        "1",
        "3",
    ]


def test_citation_normalization_handles_observed_model_formatting_variants() -> None:
    answer = (
        "One[\u200bE1], two[**E2**], three[\u202fE3\u202f], "
        "four[\u200bE4】, and grouped (E5;\u202fE6)."
    )
    normalized = normalize_citation_labels(answer)
    assert normalized == (
        "One[E1], two[E2], three[E3], four[E4], and grouped [E5] [E6]."
    )
    validation, warnings = validate_citation_labels(normalized, 6)
    assert validation["passed"] is True
    assert validation["cited_evidence_ranks"] == [1, 2, 3, 4, 5, 6]
    assert warnings == []


def test_citation_validation_rejects_labels_outside_supplied_evidence() -> None:
    validation, warnings = validate_citation_labels("Supported [E1], invented [E7].", 6)
    assert validation["passed"] is False
    assert validation["invalid_evidence_ranks"] == [7]
    assert warnings == ["Invalid evidence labels were generated: [7]"]


def test_generation_normalizes_and_validates_model_citation_brackets() -> None:
    class FakeOllama:
        chat_model = "fake-model"

        async def chat(self, messages: list[dict[str, str]]) -> dict:
            assert "[E1]" in messages[1]["content"]
            return {"message": {"content": "Use the current recommendation【E1】."}}

    result = asyncio.run(
        generate_grounded_answer(
            "test question",
            [
                {
                    "citation": "NICE NG12 · Recommendation 1.1.1 · Page 9",
                    "authority_priority": "primary",
                    "content_type": "recommendation",
                    "text": "source text",
                }
            ],
            FakeOllama(),  # type: ignore[arg-type]
        )
    )
    assert result["answer"] == "Use the current recommendation[E1]."
    assert result["citation_validation"]["passed"] is True


def test_fastapi_exposes_health_search_metrics_and_frontend() -> None:
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["corpus_chunks"] == 440
        assert health.json()["dense_index_ready"] is True

        search = client.post(
            "/api/search",
            json={
                "query": "renal cancer visible haematuria age 45",
                "mode": "bm25",
                "top_k": 5,
            },
        )
        assert search.status_code == 200
        assert search.json()["results"][0]["recommendation_id"] == "1.6.6"

        metrics = client.get("/api/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["evaluation"]["recommended_mode"] == "hybrid"
        assert metrics.json()["blind_e2e"]["questions"]["total"] == 44
        assert (
            metrics.json()["blind_e2e"]["semantic_metrics"]["status"]
            == "pending_human_adjudication"
        )
        assert "multi_judge" in metrics.json()

        refusal = client.post(
            "/api/answer",
            json={"query": "When should mesothelioma be referred?", "mode": "hybrid"},
        )
        assert refusal.status_code == 200
        assert refusal.json()["model"] is None
        assert refusal.json()["retrieval"]["results"] == []

        frontend = client.get("/")
        assert frontend.status_code == 200
        assert "NG12 Evidence Console" in frontend.text
