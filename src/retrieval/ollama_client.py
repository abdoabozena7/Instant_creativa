"""Minimal async Ollama client for embeddings and grounded generation."""

from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        embedding_model: str | None = None,
        chat_model: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.embedding_model = embedding_model or os.getenv(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"
        )
        self.chat_model = chat_model or os.getenv(
            "OLLAMA_CHAT_MODEL", "gpt-oss:120b-cloud"
        )

    async def embed(self, texts: list[str], *, timeout: float = 120.0) -> np.ndarray:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embedding_model, "input": texts},
            )
            response.raise_for_status()
        matrix = np.asarray(response.json()["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
        return response.json()

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
            models = [item.get("name") or item.get("model") for item in response.json().get("models", [])]
            return {
                "available": True,
                "models": models,
                "embedding_model_ready": self.embedding_model in models,
                "chat_model_ready": self.chat_model in models,
            }
        except (httpx.HTTPError, ValueError) as error:
            return {"available": False, "error": str(error), "models": []}
