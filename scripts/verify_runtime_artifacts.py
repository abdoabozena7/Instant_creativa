"""Verify that a fresh clone contains a self-consistent runnable NG12 snapshot."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.engine import RetrievalEngine  # noqa: E402


REQUIRED_FILES = [
    PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl",
    PROJECT_ROOT / "data" / "parsed" / "merge_report.json",
    PROJECT_ROOT / "data" / "index" / "chunk_embeddings.npy",
    PROJECT_ROOT / "data" / "index" / "index_manifest.json",
]


async def verify() -> dict[str, object]:
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required runtime artifacts: "
            + ", ".join(missing)
            + ". Run scripts/bootstrap.ps1 with both source PDF arguments to rebuild the corpus."
        )

    engine = RetrievalEngine()
    if len(engine.chunks) != 440:
        raise ValueError(f"Expected the evaluated 440-chunk snapshot, found {len(engine.chunks)}")
    result = await engine.search(
        "renal cancer visible haematuria age 45",
        mode="bm25",
        top_k=1,
    )
    top = result["results"][0] if result["results"] else None
    if not top or top.get("recommendation_id") != "1.6.6":
        raise ValueError("Runtime smoke query did not return canonical recommendation 1.6.6")

    return {
        "status": "ready",
        "chunks": len(engine.chunks),
        "dense_index_ready": engine.dense_available,
        "embedding_model": engine.ollama.embedding_model,
        "chat_model": engine.ollama.chat_model,
        "smoke_query_top_chunk": top["chunk_id"],
        "smoke_query_source": top["source_version"],
    }


def main() -> int:
    print(json.dumps(asyncio.run(verify()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
