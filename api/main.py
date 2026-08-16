"""FastAPI service for NG12 retrieval, grounded answers, and evaluation metrics."""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.retrieval.engine import PROJECT_ROOT, RetrievalEngine
from src.retrieval.generation import generate_grounded_answer


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(default=8, ge=1, le=20)
    cancer_sites: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)


class AnswerRequest(SearchRequest):
    evidence_k: int = Field(default=6, ge=1, le=10)


class RuntimeTelemetry:
    def __init__(self) -> None:
        self.search_latencies: deque[float] = deque(maxlen=500)
        self.answer_latencies: deque[float] = deque(maxlen=100)
        self.search_count = 0
        self.answer_count = 0
        self.scope_refusal_count = 0
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
            "citation_validation_pass_rate": (
                round(self.citation_pass_count / self.answer_count, 4)
                if self.answer_count
                else None
            ),
            "search_latency": self._summary(self.search_latencies),
            "answer_latency": self._summary(self.answer_latencies),
        }


app = FastAPI(
    title="NG12 Evidence Console API",
    version="0.3.0",
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
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "modes": ["hybrid", "bm25", "dense"],
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
        )
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Retrieval dependency unavailable: {error}") from error
    latency = (time.perf_counter() - started) * 1000
    telemetry.search_count += 1
    telemetry.search_latencies.append(latency)
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
        )
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Retrieval dependency unavailable: {error}") from error

    telemetry.search_count += 1
    telemetry.search_latencies.append(retrieval["latency_ms"])
    if retrieval["scope"]["status"] == "out_of_scope":
        telemetry.scope_refusal_count += 1
        return {
            "query": request.query,
            "answer": retrieval["scope"]["message"],
            "model": None,
            "retrieval": retrieval,
            "citation_validation": {
                "passed": True,
                "cited_evidence_ranks": [],
                "invalid_evidence_ranks": [],
                "available_evidence_count": 0,
            },
            "warnings": ["Generation skipped by the deterministic scope guard."],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
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
    if generation["citation_validation"]["passed"]:
        telemetry.citation_pass_count += 1
    return {
        "query": request.query,
        **generation,
        "retrieval": retrieval,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "safety_note": (
            "Evidence lookup only. This demo does not diagnose or replace clinical judgement."
        ),
    }


@app.get("/api/metrics")
def metrics() -> dict:
    merge_path = PROJECT_ROOT / "data" / "parsed" / "merge_report.json"
    evaluation_path = PROJECT_ROOT / "data" / "eval" / "retrieval_metrics.json"
    blind_evaluation_path = (
        PROJECT_ROOT / "data" / "eval" / "blind_e2e_report_v1.json"
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
