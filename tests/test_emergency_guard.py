from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.retrieval.emergency_guard import assess_emergency
from src.retrieval.engine import RetrievalEngine
from src.retrieval.generation import generate_grounded_answer


@pytest.mark.parametrize(
    "query, reason",
    [
        (
            "I am vomiting blood heavily right now and feel faint. What should I do?",
            "major_bleeding",
        ),
        (
            "I have very heavy bleeding, I feel faint, and I'm getting dizzy. What should I do?",
            "major_bleeding",
        ),
        ("I can't breathe right now.", "breathing_emergency"),
        ("The patient is unconscious at the moment.", "loss_of_consciousness"),
        ("I am having sudden severe chest pain right now.", "acute_chest_pain"),
    ],
)
def test_explicit_current_emergencies_redirect(query: str, reason: str) -> None:
    result = assess_emergency(query)
    assert result["status"] == "redirect"
    assert reason in result["reason_codes"]
    assert "local emergency services" in str(result["message"])


@pytest.mark.parametrize(
    "query",
    [
        "What does NG12 recommend for haematemesis?",
        "What does heavy bleeding mean in general?",
        "When is coughing blood referred for suspected lung cancer?",
        "A person has a history of fainting. What evidence applies?",
        "Explain the guideline for chest pain in general.",
        "A patient has persistent vomiting. Is this serious?",
        "Should someone with dysphagia be referred?",
    ],
)
def test_general_guideline_questions_are_not_emergency_redirects(query: str) -> None:
    assert assess_emergency(query)["status"] == "clear"


def test_emergency_stops_before_embedding_or_retrieval() -> None:
    class ModelMustNotRun:
        async def embed(self, texts: list[str]):
            raise AssertionError("Emergency queries must not be embedded")

    engine = RetrievalEngine()
    engine.ollama = ModelMustNotRun()  # type: ignore[assignment]
    result = asyncio.run(
        engine.search("I can't breathe right now.", mode="hybrid")
    )
    assert result["outcome"] == "emergency_redirect"
    assert result["mode_used"] == "emergency_guard"
    assert result["results"] == []
    assert result["scope"]["status"] == "not_assessed"


def test_emergency_stops_direct_generation_call() -> None:
    class ModelMustNotRun:
        chat_model = "must-not-run"

        async def chat(self, messages: list[dict[str, str]]):
            raise AssertionError("Emergency queries must not reach generation")

    result = asyncio.run(
        generate_grounded_answer(
            "I am vomiting blood heavily right now and feel faint.",
            [],
            ModelMustNotRun(),  # type: ignore[arg-type]
        )
    )
    assert result["response_status"] == "emergency_redirect"
    assert result["model"] is None
    assert result["citation_validation"]["applicable"] is False


def test_emergency_api_response_has_no_evidence_or_model() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/answer",
            json={
                "query": (
                    "I have very heavy bleeding, I feel faint, and I'm getting dizzy. "
                    "What should I do?"
                ),
                "mode": "hybrid",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "emergency_redirect"
    assert payload["model"] is None
    assert payload["retrieval"]["results"] == []
    assert payload["retrieval"]["mode_used"] == "emergency_guard"
