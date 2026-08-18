from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scripts.freeze_evaluation_architecture import ARCHITECTURE_FILES
from src.retrieval.engine import RetrievalEngine
from src.retrieval.reranker import preferred_content_types


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_content_intent_distinguishes_evidence_from_rationale() -> None:
    assert preferred_content_types("What evidence discusses study bias?") == {
        "evidence",
        "evidence_table",
    }
    assert preferred_content_types(
        "Why did the guideline committee focus on predictive value?"
    ) == {"rationale"}


def test_top_20_reranker_recovers_salient_lung_terms_from_generic_scaffolding() -> None:
    engine = RetrievalEngine()
    response = asyncio.run(
        engine.search(
            "A 55-year-old who has previously smoked now has an unexplained cough. "
            "What investigation is recommended?",
            mode="bm25",
            top_k=3,
            rerank=True,
        )
    )
    assert response["results"][0]["recommendation_id"] == "1.1.2"
    assert response["reranking"] == {
        "enabled": True,
        "method": "idf_query_coverage_and_content_intent",
        "candidate_k": 20,
        "candidates_considered": 20,
    }
    assert response["results"][0]["score_detail"]["first_stage_rank"] > 1


def test_reranker_is_in_future_architecture_freezes() -> None:
    assert "src/retrieval/reranker.py" in ARCHITECTURE_FILES


def test_blind_retrieval_experiment_preserves_recall_and_improves_precision() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "data"
            / "eval"
            / "blind_retrieval_rerank_experiment_v13.json"
        ).read_text(encoding="utf-8")
    )
    assert report["strict_precision_at_3_ceiling"] == 0.4054
    assert report["baseline"]["precision_at_3"] == 0.3423
    assert report["hybrid_reranked"]["precision_at_3"] == 0.3784
    assert report["baseline"]["recall_at_3"] == 0.973
    assert report["hybrid_reranked"]["recall_at_3"] == 0.973
    assert report["hybrid_reranked"]["recall_at_1"] == 0.8649
