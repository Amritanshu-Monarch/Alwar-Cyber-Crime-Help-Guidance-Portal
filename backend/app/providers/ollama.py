"""Local Ollama provider for offline / private inference.

As per architectural design:
Small local models (e.g. llama3.2, qwen2.5, mistral) must NOT rely on tool calling.
All routing and RAG context injection is executed deterministically in code,
passing only verified retrieved chunks and a concise, strict prompt to Ollama.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings
from app.providers.base import LLMProvider, LLMResponse

log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name: str = "ollama"

    def __init__(self, settings: Settings):
        self._s = settings
        self._base_url = (settings.ollama_base_url or "http://localhost:11434").rstrip("/")
        self._model = settings.ollama_model or "llama3.2"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        # Strict deterministic context: ignore tools for small Ollama models to prevent hallucinatory JSON
        clean_messages = []
        for m in messages:
            clean_messages.append({
                "role": m.get("role", "user"),
                "content": str(m.get("content", "")),
            })

        payload = {
            "model": self._model,
            "messages": clean_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            if resp.status_code >= 400:
                log.warning("Ollama API error (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content.strip(),
            tool_calls=[],
            finish_reason="stop",
            provider=self.name,
            model=self._model,
        )

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                resp = await client.get(f"{self._base_url}/api/version")
                return resp.status_code == 200
        except Exception:
            return False
