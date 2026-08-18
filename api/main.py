"""FastAPI service for NG12 retrieval, grounded answers, and evaluation metrics."""

from __future__ import annotations

import base64
import binascii
import json
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.retrieval.engine import PROJECT_ROOT, RetrievalEngine
from src.retrieval.generation import generate_grounded_answer
from src.vision import GeminiVisionClient, VisionExtraction


ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BASE64_LENGTH = 11_200_000


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=20)
    cancer_sites: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    rerank: bool = True


class AnswerRequest(SearchRequest):
    evidence_k: int = Field(default=6, ge=1, le=10)


class VisionAnswerRequest(BaseModel):
    image_base64: str = Field(min_length=4, max_length=MAX_IMAGE_BASE64_LENGTH)
    mime_type: str = Field(min_length=3, max_length=100)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    cancer_sites: list[str] = Field(default_factory=list)
    case_context: str = Field(min_length=3, max_length=1000)


class RuntimeTelemetry:
    def __init__(self) -> None:
        self.search_latencies: deque[float] = deque(maxlen=500)
        self.answer_latencies: deque[float] = deque(maxlen=100)
        self.search_count = 0
        self.answer_count = 0
        self.scope_refusal_count = 0
        self.safety_refusal_count = 0
        self.emergency_redirect_count = 0
        self.citation_check_count = 0
        self.citation_pass_count = 0

    @staticmethod
    def _summary(values: deque[float]) -> dict[str, float | int]:
        ordered = sorted(values)
        if not ordered:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
        p95_index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
        return {
            "count": len(ordered),
            "p50_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(ordered[p95_index], 2),
        }

    def snapshot(self) -> dict:
        return {
            "searches": self.search_count,
            "answers": self.answer_count,
            "scope_refusals": self.scope_refusal_count,
            "safety_refusals": self.safety_refusal_count,
            "emergency_redirects": self.emergency_redirect_count,
            "citation_validation_pass_rate": (
                round(self.citation_pass_count / self.citation_check_count, 4)
                if self.citation_check_count
                else None
            ),
            "search_latency": self._summary(self.search_latencies),
            "answer_latency": self._summary(self.answer_latencies),
        }


app = FastAPI(
    title="NG12 Evidence Console API",
    version="0.4.0",
    description="Authority-aware retrieval and grounded answering over a narrow NG12 corpus.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RetrievalEngine()
telemetry = RuntimeTelemetry()
vision_client = GeminiVisionClient()


def _decode_and_validate_image(image_base64: str, mime_type: str) -> bytes:
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported.")
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Image data is not valid base64.") from error
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MB limit.")
    signatures = {
        "image/jpeg": image_bytes.startswith(b"\xff\xd8\xff"),
        "image/png": image_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP",
    }
    if not signatures[mime_type]:
        raise HTTPException(
            status_code=400,
            detail="Image bytes do not match the declared media type.",
        )
    return image_bytes


def _vision_public_result(extraction: VisionExtraction, model: str) -> dict:
    return {
        "status": "ready" if extraction.can_handoff else "refused",
        "model": model,
        **extraction.model_dump(),
        "limitations": (
            "Radiology images are converted to a neutral, unverified description; "
            "this feature does not diagnose or replace a radiologist."
        ),
    }


@app.get("/api/health")
async def health() -> dict:
    ollama = await engine.ollama.health()
    return {
        "status": "ready" if engine.chunks else "degraded",
        "corpus_chunks": len(engine.chunks),
        "dense_index_ready": engine.dense_available,
        "embedding_model": engine.ollama.embedding_model,
        "chat_model": engine.ollama.chat_model,
        "ollama": ollama,
        "vision": {
            "available": vision_client.configured,
            "model": vision_client.model,
            "accepted_mime_types": sorted(ALLOWED_IMAGE_MIME_TYPES),
            "max_image_bytes": MAX_IMAGE_BYTES,
        },
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "modes": ["hybrid", "bm25", "dense"],
        "reranker": {
            "default_enabled": True,
            "candidate_k": 20,
            "method": "idf_query_coverage_and_content_intent",
        },
        "cancer_sites": [
            "lung",
            "colorectal",
            "oesophageal",
            "stomach",
            "pancreatic",
            "bladder",
            "renal",
        ],
        "content_types": sorted({chunk["content_type"] for chunk in engine.chunks}),
        "source_priority": {
            "primary": "2026 current guideline",
            "supporting": "2015 full guideline evidence and rationale",
        },
    }


@app.post("/api/search")
async def search(request: SearchRequest) -> dict:
    started = time.perf_counter()
    try:
        response = await engine.search(
            request.query,
            mode=request.mode,
            top_k=request.top_k,
            cancer_sites=request.cancer_sites,
            content_types=request.content_types,
            rerank=request.rerank,
        )
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Retrieval dependency unavailable: {error}") from error
    latency = (time.perf_counter() - started) * 1000
    telemetry.search_count += 1
    telemetry.search_latencies.append(latency)
    if response["outcome"] == "safety_refusal":
        telemetry.safety_refusal_count += 1
    if response["outcome"] == "emergency_redirect":
        telemetry.emergency_redirect_count += 1
    if response["scope"]["status"] == "out_of_scope":
        telemetry.scope_refusal_count += 1
    return response


@app.post("/api/answer")
async def answer(request: AnswerRequest) -> dict:
    started = time.perf_counter()
    try:
        retrieval = await engine.search(
            request.query,
            mode=request.mode,
            top_k=request.evidence_k,
            cancer_sites=request.cancer_sites,
            content_types=request.content_types,
            rerank=request.rerank,
        )
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Retrieval dependency unavailable: {error}") from error

    telemetry.search_count += 1
    telemetry.search_latencies.append(retrieval["latency_ms"])
    if retrieval["outcome"] == "safety_refusal":
        telemetry.safety_refusal_count += 1
        return {
            "query": request.query,
            "outcome": "safety_refusal",
            "answer": retrieval["safety"]["message"],
            "model": None,
            "retrieval": retrieval,
            "safety": retrieval["safety"],
            "citation_validation": {
                "applicable": False,
                "passed": None,
                "cited_evidence_ranks": [],
                "invalid_evidence_ranks": [],
                "available_evidence_count": 0,
            },
            "warnings": [
                "Retrieval and generation skipped by the deterministic instruction-safety guard."
            ],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety_note": (
                "Evidence lookup only. This demo does not diagnose or replace clinical judgement."
            ),
        }
    if retrieval["outcome"] == "emergency_redirect":
        telemetry.emergency_redirect_count += 1
        return {
            "query": request.query,
            "outcome": "emergency_redirect",
            "answer": retrieval["emergency"]["message"],
            "model": None,
            "retrieval": retrieval,
            "emergency": retrieval["emergency"],
            "citation_validation": {
                "applicable": False,
                "passed": None,
                "label_validation_passed": None,
                "claim_coverage_passed": None,
                "claim_units_checked": 0,
                "cited_claim_units": 0,
                "citation_coverage_rate": None,
                "uncited_claim_units": [],
                "cited_evidence_ranks": [],
                "invalid_evidence_ranks": [],
                "available_evidence_count": 0,
            },
            "warnings": [
                "Retrieval and generation skipped by the deterministic emergency guard."
            ],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety_note": (
                "Emergency redirect only. This demo does not diagnose or provide urgent care."
            ),
        }
    if retrieval["scope"]["status"] == "out_of_scope":
        telemetry.scope_refusal_count += 1
        return {
            "query": request.query,
            "outcome": "scope_refusal",
            "answer": retrieval["scope"]["message"],
            "model": None,
            "retrieval": retrieval,
            "citation_validation": {
                "applicable": False,
                "passed": None,
                "cited_evidence_ranks": [],
                "invalid_evidence_ranks": [],
                "available_evidence_count": 0,
            },
            "warnings": ["Generation skipped by the deterministic scope guard."],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety_note": (
                "Evidence lookup only. This demo does not diagnose or replace clinical judgement."
            ),
        }
    if retrieval.get("answerability", {}).get("status") == "insufficient":
        generation = await generate_grounded_answer(
            request.query, [], engine.ollama
        )
        telemetry.answer_count += 1
        telemetry.answer_latencies.append(generation["latency_ms"])
        return {
            "query": request.query,
            "outcome": "insufficient_information",
            **generation,
            "retrieval": retrieval,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety_note": (
                "Evidence lookup only. This demo does not diagnose or replace clinical judgement."
            ),
        }
    if not retrieval["results"]:
        raise HTTPException(status_code=404, detail="No evidence matched the query and filters.")

    try:
        generation = await generate_grounded_answer(
            request.query, retrieval["results"], engine.ollama
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Ollama generation unavailable: {error}") from error
    telemetry.answer_count += 1
    telemetry.answer_latencies.append(generation["latency_ms"])
    telemetry.citation_check_count += 1
    if generation["citation_validation"]["passed"]:
        telemetry.citation_pass_count += 1
    outcome = generation["response_status"]
    public_retrieval = retrieval
    if outcome == "generation_rejected":
        public_retrieval = {**retrieval, "results": []}
    return {
        "query": request.query,
        "outcome": outcome,
        **generation,
        "retrieval": public_retrieval,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "safety_note": (
            "Evidence lookup only. This demo does not diagnose or replace clinical judgement."
        ),
    }


@app.post("/api/vision/answer")
async def vision_answer(request: VisionAnswerRequest) -> dict:
    """Add attachment context to a required question, then reuse the text answer path."""

    started = time.perf_counter()
    _decode_and_validate_image(request.image_base64, request.mime_type)
    if not vision_client.configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "Image analysis is not configured. Set GEMINI_API_KEY; the text-only "
                "NG12 core is still available."
            ),
        )
    try:
        extraction = await vision_client.analyze(
            image_base64=request.image_base64,
            mime_type=request.mime_type,
            case_context=request.case_context,
        )
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Vision response was invalid: {error}") from error
    except httpx.TimeoutException as error:
        raise HTTPException(status_code=503, detail="Vision service timed out.") from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Vision service rejected the request with status {error.response.status_code}.",
        ) from error
    except (RuntimeError, httpx.HTTPError) as error:
        raise HTTPException(status_code=503, detail=f"Vision service unavailable: {error}") from error

    vision = _vision_public_result(extraction, vision_client.model)
    if not extraction.can_handoff:
        return {
            "query": extraction.extracted_query,
            "outcome": "vision_refusal",
            "answer": (
                "The image could not be mapped safely to the configured NG12 scope. "
                "Upload a de-identified report or image related to lung, colorectal, "
                "oesophageal, stomach, pancreatic, bladder, or renal cancer."
            ),
            "model": None,
            "vision": vision,
            "warnings": ["The existing retrieval and generation pipeline was not called."],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "safety_note": "Image understanding is not a diagnosis or radiology report.",
        }

    core_response = await answer(
        AnswerRequest(
            query=extraction.extracted_query,
            mode=request.mode,
            evidence_k=6,
            cancer_sites=request.cancer_sites or extraction.cancer_sites,
        )
    )
    return {
        **core_response,
        "vision": vision,
        "input_method": "vision_adapter",
        "case_context": request.case_context,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


@app.get("/api/metrics")
def metrics() -> dict:
    merge_path = PROJECT_ROOT / "data" / "parsed" / "merge_report.json"
    evaluation_path = PROJECT_ROOT / "data" / "eval" / "retrieval_metrics.json"
    blind_evaluation_path = (
        PROJECT_ROOT / "data" / "eval" / "blind_e2e_report_v12.json"
    )
    multi_judge_path = (
        PROJECT_ROOT / "data" / "eval" / "multi_judge_report_v1.json"
    )
    merge_report = json.loads(merge_path.read_text(encoding="utf-8"))
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.is_file()
        else None
    )
    blind_e2e = (
        json.loads(blind_evaluation_path.read_text(encoding="utf-8"))
        if blind_evaluation_path.is_file()
        else None
    )
    multi_judge = (
        json.loads(multi_judge_path.read_text(encoding="utf-8"))
        if multi_judge_path.is_file()
        else None
    )
    return {
        "corpus": {
            "sources": merge_report["sources"],
            "records": merge_report["records"],
            "reconciliation": {
                "duplicates_detected_count": merge_report["reconciliation"]["duplicates_detected_count"],
                "conflicts_detected_count": merge_report["reconciliation"]["conflicts_detected_count"],
                "unmatched_historical_recommendations": len(
                    merge_report["reconciliation"]["unmatched_historical_recommendations"]
                ),
            },
            "chunking": merge_report["chunking"],
        },
        "evaluation": evaluation,
        "blind_e2e": blind_e2e,
        "multi_judge": multi_judge,
        "runtime": telemetry.snapshot(),
    }


@app.get("/api/chunks/{chunk_id}")
def chunk_by_id(chunk_id: str) -> dict:
    chunk = next((item for item in engine.chunks if item["chunk_id"] == chunk_id), None)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk


frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
