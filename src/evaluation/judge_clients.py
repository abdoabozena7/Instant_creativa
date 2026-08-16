"""Secret-safe structured clients for independent semantic judges."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx


JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = JSON_FENCE.sub("", text.strip())
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    return value


class SemanticJudge(ABC):
    name: str
    model: str

    @abstractmethod
    async def judge(
        self, prompt: str, schema: dict[str, Any], *, timeout: float = 240.0
    ) -> dict[str, Any]:
        raise NotImplementedError


class GeminiJudge(SemanticJudge):
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv(
            "GEMINI_API_KEY"
        )
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Configure it in the environment; "
                "never place API keys in source files or command arguments."
            )
        self.model = model or os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.6-flash")
        self.base_url = base_url.rstrip("/")

    async def judge(
        self, prompt: str, schema: dict[str, Any], *, timeout: float = 240.0
    ) -> dict[str, Any]:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "candidateCount": 1,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
            response.raise_for_status()
        body = response.json()
        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Gemini returned no structured candidate text") from error
        return parse_json_text(text)


class OllamaJudge(SemanticJudge):
    name = "gpt_oss"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_JUDGE_MODEL", "gpt-oss:120b-cloud")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")

    async def judge(
        self, prompt: str, schema: dict[str, Any], *, timeout: float = 240.0
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
        try:
            text = response.json()["message"]["content"]
        except (KeyError, TypeError) as error:
            raise ValueError("Ollama returned no structured message content") from error
        return parse_json_text(text)
