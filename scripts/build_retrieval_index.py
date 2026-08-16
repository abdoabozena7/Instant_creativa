"""Build a normalized Ollama embedding matrix for exact-cosine NG12 retrieval."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.engine import RetrievalEngine  # noqa: E402
from src.retrieval.ollama_client import OllamaClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed" / "chunks.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "index"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


async def build(chunks_path: Path, output_dir: Path, batch_size: int) -> None:
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    client = OllamaClient()
    documents = [
        "search_document: " + RetrievalEngine._retrieval_text(chunk) for chunk in chunks
    ]
    batches: list[np.ndarray] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        batches.append(await client.embed(batch, timeout=300.0))
        print(f"Embedded {min(start + len(batch), len(documents))}/{len(documents)}")
    matrix = np.vstack(batches).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "chunk_embeddings.npy", matrix)
    digest = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    manifest = {
        "document": "NICE NG12",
        "embedding_model": client.embedding_model,
        "chunks_file": chunks_path.name,
        "chunks_sha256": digest,
        "rows": int(matrix.shape[0]),
        "dimensions": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "normalized": True,
        "index_type": "exact cosine over NumPy matrix",
    }
    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def main() -> int:
    args = parse_args()
    asyncio.run(build(args.chunks.resolve(), args.output_dir.resolve(), args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
