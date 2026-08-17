"""Fingerprint the frozen NG12 retrieval/generation system before blind evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_FILES = [
    "src/retrieval/bm25.py",
    "src/retrieval/engine.py",
    "src/retrieval/generation.py",
    "src/retrieval/ollama_client.py",
    "src/retrieval/scope_guard.py",
    "data/parsed/chunks.jsonl",
    "data/index/index_manifest.json",
    "data/index/chunk_embeddings.npy",
]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "evaluation_freeze.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    files = {}
    aggregate = hashlib.sha256()
    for relative in ARCHITECTURE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        files[relative] = {"sha256": digest, "bytes": path.stat().st_size}
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(digest.encode("ascii"))

    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "architecture_sha256": aggregate.hexdigest(),
        "policy": (
            "No retriever, index, scope-guard, prompt, embedding model, or chat-model changes "
            "are permitted while this blind run is being scored. Any later change creates a new freeze."
        ),
        "retrieval_mode": "hybrid",
        "embedding_model": "nomic-embed-text:latest",
        "chat_model": "gpt-oss:120b-cloud",
        "files": files,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
