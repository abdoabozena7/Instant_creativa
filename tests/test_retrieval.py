from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.retrieval.bm25 import BM25Index
from src.retrieval.engine import RetrievalEngine
from src.retrieval.generation import (
    CITATION_PATTERN,
    generate_grounded_answer,
    normalize_citation_labels,
    validate_citation_labels,
)
from src.retrieval.scope_guard import assess_query_answerability, assess_scope


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


def test_bm25_query_scaffolding_cannot_outrank_the_clinical_feature() -> None:
    index = BM25Index(
        [
            "what about the patient information and what the patient should expect",
            "pancreatic new-onset diabetes with weight loss",
        ]
    )
    scores = index.scores("What about the diabetes?")
    assert scores[1] > 0
    assert scores[0] == 0
    assert scores[1] > scores[0]


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_diabetes_query_retrieves_diabetes_evidence_before_patient_support(
    mode: str,
) -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search("What about the diabetes?", mode=mode, top_k=5)
    )
    top = response["results"][0]
    assert "diabetes" in top["text"].lower()
    assert top["chunk_id"] != "ng12_1.14.3_c01"


@pytest.mark.parametrize(
    "query",
    [
        "Should this patient be referred for suspected cancer?",
        "Does referral apply to this person?",
        "Should they enter an urgent cancer pathway?",
        "Can the referral decision be determined for this patient?",
        "What does NG12 recommend for this patient?",
        "Would this patient qualify for referral?",
        "Does this person require an urgent investigation?",
        "What should we do for this patient?",
        "Please tell me whether this person is eligible for a cancer pathway.",
    ],
)
def test_context_free_clinical_decisions_are_insufficient(query: str) -> None:
    answerability = assess_query_answerability(query)
    assert answerability["status"] == "insufficient"
    assert answerability["clinical_features"] == []


@pytest.mark.parametrize(
    "query",
    [
        "A patient has some stomach issues, is this serious?",
        "My stomach has problems. Should I worry?",
        "This person feels unwell around the stomach. Could it be cancer?",
        "A patient has gastric symptoms. Is that concerning?",
        "The patient has bowel problems. Is it dangerous?",
        "They have issues with their pancreas. What does this mean?",
        "Someone says something is wrong with their stomach. Is it urgent?",
        "A 70-year-old patient has persistent stomach issues. Is this serious?",
        "Patient reports digestive problems and asks whether they are serious.",
        "My abdominal symptoms are concerning. What do they mean?",
        "A patient has breathing issues. Could this be dangerous?",
        "This person has urinary problems. Should they worry?",
    ],
)
def test_patient_specific_vague_assessments_are_insufficient(query: str) -> None:
    answerability = assess_query_answerability(query)
    assert answerability["status"] == "insufficient"
    assert answerability["clinical_features"] == []


@pytest.mark.parametrize(
    "query, expected_feature",
    [
        ("Should someone with dysphagia be referred?", "dysphagia"),
        ("Does unexplained visible haematuria need referral?", "haematuria"),
        ("Should a patient with new-onset diabetes and weight loss get a scan?", "diabetes"),
        ("When is a cough investigated for lung cancer?", "cough"),
        (
            "A 62-year-old has weight loss and new-onset diabetes. What does NG12 recommend?",
            "diabetes",
        ),
        ("A patient has persistent vomiting. Is this serious?", "vomiting"),
        ("My stomach pain is accompanied by weight loss. Should I worry?", "pain"),
        ("This patient has dysphagia. Could it be cancer?", "dysphagia"),
        ("A person has haematemesis. Is it dangerous?", "haematemesis"),
        ("They found an abdominal mass. What does this mean?", "mass"),
        ("A patient has difficulty breathing. Is this serious?", "difficulty"),
        ("This person noticed blood in their urine. Should they worry?", "blood"),
    ],
)
def test_clinical_features_prevent_the_low_information_guard(
    query: str, expected_feature: str
) -> None:
    answerability = assess_query_answerability(query)
    assert answerability["status"] == "model_assessed"
    assert expected_feature in answerability["clinical_features"]


def test_out_of_scope_query_never_reaches_retrieval_results() -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search("When should breast cancer be referred?", mode="hybrid")
    )
    assert response["mode_used"] == "scope_guard"
    assert response["results"] == []


def test_context_free_decision_stops_before_retrieval_scoring() -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search(
            "Should this patient be referred for suspected cancer?",
            mode="hybrid",
        )
    )
    assert response["mode_used"] == "answerability_guard"
    assert response["answerability"]["status"] == "insufficient"
    assert response["results"] == []
    assert "inferred_anchor_sites" not in response["scope"]


def test_vague_patient_assessment_stops_before_retrieval_scoring() -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search(
            "A patient has some stomach issues, is this serious?",
            mode="hybrid",
        )
    )
    assert response["mode_used"] == "answerability_guard"
    assert response["answerability"]["status"] == "insufficient"
    assert response["results"] == []
    assert response["scope"]["selected_sites"] == ["stomach"]
    assert "inferred_anchor_sites" not in response["scope"]


def test_citation_pattern_accepts_canonical_evidence_labels() -> None:
    assert CITATION_PATTERN.findall("Referral is recommended [E1] and supported [E3].") == [
        "1",
        "3",
    ]


def test_citation_normalization_handles_observed_model_formatting_variants() -> None:
    answer = (
        "One[\u200bE1], two[**E2**], three[\u202fE3\u202f], "
        "four[\u200bE4】, grouped (E5;\u202fE6), line ref 【E2†L1-L5】, "
        "bullet **E3** – support, and named [Evidence\u202fE2; E4]."
    )
    normalized = normalize_citation_labels(answer)
    assert normalized == (
        "One[E1], two[E2], three[E3], four[E4], grouped [E5] [E6], line ref [E2], "
        "bullet [E3] – support, and named [E2] [E4]."
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


@pytest.mark.parametrize(
    "query",
    [
        "Should this person be referred for suspected cancer?",
        "A patient has some stomach issues, is this serious?",
    ],
)
def test_low_information_assessment_skips_the_generation_model(query: str) -> None:
    class ModelMustNotRun:
        chat_model = "must-not-run"

        async def chat(self, messages: list[dict[str, str]]) -> dict:
            raise AssertionError("The model must not receive a context-free decision query")

    result = asyncio.run(
        generate_grounded_answer(
            query,
            [
                {
                    "citation": "NICE NG12 · Patient support · Page 35",
                    "authority_priority": "primary",
                    "content_type": "patient_support",
                    "text": "Information for people after referral.",
                }
            ],
            ModelMustNotRun(),  # type: ignore[arg-type]
        )
    )
    assert result["model"] is None
    assert result["answerability"]["status"] == "insufficient"
    assert result["citation_validation"]["passed"] is True
    assert result["citation_validation"]["cited_evidence_ranks"] == []
    assert result["answer"].startswith("Insufficient information")


def test_specific_decision_reaches_model_with_missing_fact_contract() -> None:
    class InspectingModel:
        chat_model = "fake-model"

        async def chat(self, messages: list[dict[str, str]]) -> dict:
            system = messages[0]["content"]
            user = messages[1]["content"]
            assert "patient attribute not stated" in system
            assert "Preserve AND/OR structure exactly" in system
            assert "do not infer them from eligibility criteria" in user
            return {
                "message": {
                    "content": "The criterion depends on age, which was not supplied [E1]."
                }
            }

    result = asyncio.run(
        generate_grounded_answer(
            "Should someone with unexplained visible haematuria be referred?",
            [
                {
                    "citation": "NICE NG12 · Renal cancer · Recommendation 1.6.6 · Page 23",
                    "authority_priority": "primary",
                    "content_type": "recommendation",
                    "text": "Refer people aged 45 and over with unexplained visible haematuria.",
                }
            ],
            InspectingModel(),  # type: ignore[arg-type]
        )
    )
    assert result["model"] == "fake-model"
    assert result["answerability"]["status"] == "model_assessed"
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
        assert metrics.json()["blind_e2e"]["evaluation_name"] == "blind_end_to_end_v6"
        assert (
            metrics.json()["blind_e2e"]["semantic_metrics"]["status"] == "not_run"
        )
        assert "multi_judge" in metrics.json()

        refusal = client.post(
            "/api/answer",
            json={"query": "When should mesothelioma be referred?", "mode": "hybrid"},
        )
        assert refusal.status_code == 200
        assert refusal.json()["model"] is None
        assert refusal.json()["retrieval"]["results"] == []

        insufficient = client.post(
            "/api/answer",
            json={
                "query": "Should this patient be referred for suspected cancer?",
                "mode": "hybrid",
            },
        )
        assert insufficient.status_code == 200
        assert insufficient.json()["model"] is None
        assert insufficient.json()["answerability"]["status"] == "insufficient"
        assert insufficient.json()["retrieval"]["mode_used"] == "answerability_guard"
        assert insufficient.json()["retrieval"]["results"] == []

        vague = client.post(
            "/api/answer",
            json={
                "query": "A patient has some stomach issues, is this serious?",
                "mode": "hybrid",
            },
        )
        assert vague.status_code == 200
        assert vague.json()["model"] is None
        assert vague.json()["answerability"]["status"] == "insufficient"
        assert vague.json()["retrieval"]["mode_used"] == "answerability_guard"
        assert vague.json()["retrieval"]["results"] == []

        frontend = client.get("/")
        assert frontend.status_code == 200
        assert "NG12 Evidence Console" in frontend.text
