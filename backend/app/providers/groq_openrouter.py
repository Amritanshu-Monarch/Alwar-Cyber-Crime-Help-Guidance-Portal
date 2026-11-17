"""Groq and OpenRouter LLM providers (OpenAI-compatible APIs with free tier options)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings
from app.providers.base import LLMProvider, LLMResponse, LLMToolCall

log = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    name: str = "groq"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.groq_api_key
        self._model = settings.groq_model or "llama-3.3-70b-versatile"
        self._base_url = "https://api.groq.com/openai/v1"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        if not self._key:
            raise ValueError("GROQ_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{self._base_url}/chat/completions", headers=headers, json=payload)
            if resp.status_code >= 400:
                log.warning("Groq API error (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(content="", provider=self.name, model=self._model)

        msg = choices[0].get("message", {})
        raw_tools = msg.get("tool_calls") or []
        tool_calls = [
            LLMToolCall(
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments", {}),
                call_id=tc.get("id", ""),
            )
            for tc in raw_tools
        ]

        return LLMResponse(
            content=msg.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason", "stop"),
            provider=self.name,
            model=self._model,
        )

    async def is_healthy(self) -> bool:
        return bool(self._key)


class OpenRouterProvider(LLMProvider):
    name: str = "openrouter"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.openrouter_api_key
        self._model = settings.openrouter_model or "meta-llama/llama-3.3-70b-instruct:free"
        self._base_url = "https://openrouter.ai/api/v1"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        if not self._key:
            raise ValueError("OPENROUTER_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://alwar.police.rajasthan.gov.in",
            "X-Title": "Alwar Cyber Crime Guidance Portal",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{self._base_url}/chat/completions", headers=headers, json=payload)
            if resp.status_code >= 400:
                log.warning("OpenRouter API error (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        return LLMResponse(
            content=content.strip(),
            tool_calls=[],
            finish_reason="stop",
            provider=self.name,
            model=self._model,
        )

    async def is_healthy(self) -> bool:
        return bool(self._key)
