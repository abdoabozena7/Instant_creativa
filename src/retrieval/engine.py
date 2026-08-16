"""Transparent BM25, dense, and hybrid retrieval for the NG12 corpus."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .bm25 import BM25Index
from .ollama_client import OllamaClient
from .scope_guard import assess_scope


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "index" / "chunk_embeddings.npy"
BM25_WEIGHT = 0.55
DENSE_WEIGHT = 0.45
EXACT_RECOMMENDATION_BOOST = 0.30
EVIDENCE_INTENT_BOOST = 0.06
CANONICAL_RECOMMENDATION_BOOST = 0.22
CURRENT_SOURCE_BOOST = 0.03
METHODOLOGY_PENALTY = -0.04
EXPLICIT_SITE_BOOST = 0.03
ANCHOR_MINIMUM_SCORE = 0.75
ANCHOR_SITE_BOOST = 0.10
ANCHOR_OTHER_SITE_PENALTY = -0.06
EVIDENCE_TERMS = re.compile(
    r"\b(?:evidence|rationale|reason|why|study|studies|ppv|predictive value|risk of bias|committee)\b",
    re.IGNORECASE,
)
METHODOLOGY_TERMS = re.compile(r"\b(?:methodology|guideline development|search strategy)\b", re.I)
RECOMMENDATION_ID = re.compile(r"\b1\.\d{1,2}\.\d{1,2}\b")


class RetrievalEngine:
    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS,
        embeddings_path: Path = DEFAULT_EMBEDDINGS,
        *,
        ollama: OllamaClient | None = None,
    ) -> None:
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Missing chunk corpus: {chunks_path}")
        self.chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.bm25 = BM25Index(
            [self._retrieval_text(chunk) for chunk in self.chunks]
        )
        self.ollama = ollama or OllamaClient()
        self.embeddings_path = embeddings_path
        self.embeddings: np.ndarray | None = None
        if embeddings_path.is_file():
            self.embeddings = self._load_embeddings(chunks_path, embeddings_path)

    @property
    def dense_available(self) -> bool:
        return self.embeddings is not None

    async def search(
        self,
        query: str,
        *,
        mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
        top_k: int = 8,
        cancer_sites: list[str] | None = None,
        content_types: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        query = " ".join(query.split()).strip()
        if len(query) < 3:
            raise ValueError("Query must contain at least 3 characters")
        top_k = max(1, min(top_k, 20))
        scope = assess_scope(query)
        if scope["status"] == "out_of_scope":
            return {
                "query": query,
                "mode_requested": mode,
                "mode_used": "scope_guard",
                "scope": scope,
                "results": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": [],
            }

        allowed = np.asarray(
            [
                (not cancer_sites or bool(set(cancer_sites) & set(chunk["cancer_sites"])))
                and (not content_types or chunk["content_type"] in content_types)
                for chunk in self.chunks
            ],
            dtype=bool,
        )
        if not allowed.any():
            return {
                "query": query,
                "mode_requested": mode,
                "mode_used": mode,
                "scope": scope,
                "results": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "warnings": ["No chunks match the selected metadata filters."],
            }

        warnings: list[str] = []
        bm25_scores = self.bm25.scores(query)
        dense_scores = np.zeros(len(self.chunks), dtype=np.float32)
        mode_used = mode
        if mode in {"dense", "hybrid"}:
            if self.embeddings is None:
                warnings.append("Dense index unavailable; fell back to BM25.")
                mode_used = "bm25"
            else:
                query_vector = (await self.ollama.embed([f"search_query: {query}"]))[0]
                dense_scores = self.embeddings @ query_vector

        candidate_indices = np.flatnonzero(allowed)
        bm25_normalized = self._normalize(bm25_scores, allowed)
        dense_normalized = np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0)

        evidence_intent = bool(EVIDENCE_TERMS.search(query))
        methodology_intent = bool(METHODOLOGY_TERMS.search(query))
        explicit_ids = set(RECOMMENDATION_ID.findall(query))
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index in candidate_indices.tolist():
            chunk = self.chunks[index]
            if mode_used == "bm25":
                base_score = float(bm25_normalized[index])
            elif mode_used == "dense":
                base_score = float(dense_normalized[index])
            else:
                # The weights remain visible in score_detail and are evaluated
                # against the single-mode baselines rather than assumed.
                base_score = float(
                    BM25_WEIGHT * bm25_normalized[index]
                    + DENSE_WEIGHT * dense_normalized[index]
                )

            authority_adjustment = 0.0
            explanations: list[str] = []
            if chunk["recommendation_id"] in explicit_ids:
                authority_adjustment += EXACT_RECOMMENDATION_BOOST
                explanations.append("exact recommendation ID")
            if evidence_intent:
                if chunk["content_type"] in {"evidence", "evidence_table", "rationale"}:
                    authority_adjustment += EVIDENCE_INTENT_BOOST
                    explanations.append("evidence-intent match")
            elif chunk["canonical_recommendation"]:
                authority_adjustment += CANONICAL_RECOMMENDATION_BOOST
                explanations.append("canonical current recommendation")
            elif chunk["source_version"] == "2026_current":
                authority_adjustment += CURRENT_SOURCE_BOOST
                explanations.append("current authoritative source")
            if not methodology_intent and chunk["content_type"] == "methodology":
                authority_adjustment += METHODOLOGY_PENALTY
            if scope["selected_sites"] and set(scope["selected_sites"]) & set(chunk["cancer_sites"]):
                authority_adjustment += EXPLICIT_SITE_BOOST
                explanations.append("explicit cancer-site match")

            final_score = max(0.0, base_score + authority_adjustment)
            scored.append(
                (
                    final_score,
                    index,
                    {
                        "base_score": round(base_score, 6),
                        "bm25_score": round(float(bm25_scores[index]), 6),
                        "dense_score": round(float(dense_scores[index]), 6) if mode_used != "bm25" else None,
                        "authority_adjustment": round(authority_adjustment, 6),
                        "explanations": explanations,
                    },
                )
            )

        # When the query names no site but a current recommendation is a
        # strong match, use that recommendation's site as a transparent
        # coherence signal for the remaining evidence. This improves the
        # context set without hard-filtering useful cross-cutting records.
        if not evidence_intent and not scope["selected_sites"]:
            anchor_candidates = [
                item
                for item in scored
                if self.chunks[item[1]]["canonical_recommendation"]
                and self.chunks[item[1]]["cancer_sites"]
            ]
            if anchor_candidates:
                anchor = max(anchor_candidates, key=lambda item: item[0])
                anchor_sites = set(self.chunks[anchor[1]]["cancer_sites"])
                if anchor[0] >= ANCHOR_MINIMUM_SCORE:
                    scope["inferred_anchor_sites"] = sorted(anchor_sites)
                    coherent: list[tuple[float, int, dict[str, Any]]] = []
                    for score, index, detail in scored:
                        chunk_sites = set(self.chunks[index]["cancer_sites"])
                        if chunk_sites & anchor_sites:
                            adjustment = ANCHOR_SITE_BOOST
                            detail["explanations"].append(
                                "site coherence from top canonical match"
                            )
                        elif chunk_sites:
                            adjustment = ANCHOR_OTHER_SITE_PENALTY
                        else:
                            adjustment = 0.0
                        detail["authority_adjustment"] = round(
                            detail["authority_adjustment"] + adjustment, 6
                        )
                        coherent.append((max(0.0, score + adjustment), index, detail))
                    scored = coherent
        scored.sort(
            key=lambda item: (
                item[0],
                self.chunks[item[1]]["source_version"] == "2026_current",
                -self.chunks[item[1]]["page"],
            ),
            reverse=True,
        )

        results = []
        for rank, (score, index, score_detail) in enumerate(scored[:top_k], start=1):
            chunk = dict(self.chunks[index])
            chunk.update(
                {
                    "rank": rank,
                    "score": round(score, 6),
                    "score_detail": score_detail,
                    "citation": self._citation(chunk),
                }
            )
            results.append(chunk)
        return {
            "query": query,
            "mode_requested": mode,
            "mode_used": mode_used,
            "scope": scope,
            "filters": {"cancer_sites": cancer_sites or [], "content_types": content_types or []},
            "results": results,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "warnings": warnings,
        }

    @staticmethod
    def _retrieval_text(chunk: dict[str, Any]) -> str:
        metadata = " ".join(
            filter(
                None,
                [
                    chunk.get("section"),
                    chunk.get("subsection"),
                    chunk.get("recommendation_id"),
                    " ".join(chunk.get("cancer_sites", [])),
                    chunk.get("content_type"),
                ],
            )
        )
        return f"{metadata}\n{chunk['text']}"

    def _load_embeddings(
        self, chunks_path: Path, embeddings_path: Path
    ) -> np.ndarray:
        manifest_path = embeddings_path.with_name("index_manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing dense-index manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_digest = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        if manifest.get("chunks_sha256") != chunks_digest:
            raise ValueError(
                "Dense index does not match the chunk corpus; rebuild the retrieval index."
            )
        if manifest.get("embedding_model") != self.ollama.embedding_model:
            raise ValueError(
                "Dense index embedding model does not match the configured query model."
            )

        matrix = np.load(embeddings_path, allow_pickle=False)
        expected_shape = (len(self.chunks), int(manifest.get("dimensions", -1)))
        if matrix.shape != expected_shape or manifest.get("rows") != len(self.chunks):
            raise ValueError(
                f"Embedding shape {matrix.shape} does not match manifest/corpus {expected_shape}."
            )
        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(matrix).all() or not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Dense index contains invalid or non-normalized vectors.")
        return matrix.astype(np.float32, copy=False)

    @staticmethod
    def _normalize(scores: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        output = np.zeros_like(scores, dtype=np.float32)
        selected = scores[allowed]
        maximum = float(selected.max()) if selected.size else 0.0
        if maximum > 0:
            output[allowed] = selected / maximum
        return output

    @staticmethod
    def _citation(chunk: dict[str, Any]) -> str:
        parts = ["NICE NG12", chunk["section"]]
        if chunk.get("recommendation_id"):
            parts.append(f"Recommendation {chunk['recommendation_id']}")
        page = str(chunk["page"])
        if chunk.get("page_end") and chunk["page_end"] != chunk["page"]:
            page = f"{page}–{chunk['page_end']}"
        parts.append(f"Page {page}")
        parts.append("current 2026" if chunk["source_version"] == "2026_current" else "full guideline 2015")
        return " · ".join(parts)
