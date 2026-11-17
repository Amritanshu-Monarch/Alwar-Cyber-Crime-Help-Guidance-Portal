"""Provider Registry and Orchestrator.
Manages LLM fallback order (429/timeout/5xx) and STT/TTS provider selection.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderHealth,
    STTProvider,
    TTSProvider,
)
from app.providers.gemini import (
    GeminiLLMProvider,
    GeminiSTTProvider,
    GeminiTTSProvider,
)
from app.providers.groq_openrouter import GroqProvider, OpenRouterProvider
from app.providers.ollama import OllamaProvider
from app.providers.sarvam import (
    SarvamLLMProvider,
    SarvamSTTProvider,
    SarvamTTSProvider,
)

log = logging.getLogger(__name__)


class ProviderRegistry:
    def __init__(self, settings: Settings):
        self._s = settings
        self._llm_providers: dict[str, LLMProvider] = {
            "gemini": GeminiLLMProvider(settings),
            "sarvam": SarvamLLMProvider(settings),
            "groq": GroqProvider(settings),
            "openrouter": OpenRouterProvider(settings),
            "ollama": OllamaProvider(settings),
        }
        self._stt_providers: dict[str, STTProvider] = {
            "gemini": GeminiSTTProvider(settings),
            "sarvam": SarvamSTTProvider(settings),
        }
        self._tts_providers: dict[str, TTSProvider] = {
            "gemini": GeminiTTSProvider(settings),
            "sarvam": SarvamTTSProvider(settings),
        }

    def get_llm_order(self) -> list[tuple[str, LLMProvider]]:
        order = []
        for name in self._s.llm_provider_order:
            provider = self._llm_providers.get(name.lower())
            if provider:
                order.append((name.lower(), provider))
        if not order:
            order.append(("gemini", self._llm_providers["gemini"]))
        return order

    async def generate_with_fallback(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        order = self.get_llm_order()
        last_error = None

        for name, provider in order:
            try:
                # For small local models like Ollama, do not send tools
                active_tools = None if name == "ollama" else tools
                return await provider.generate(
                    messages=messages,
                    tools=active_tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                log.warning("Provider '%s' failed: %s; falling back to next provider", name, e)
                last_error = e

        if last_error:
            raise last_error
        raise RuntimeError("No LLM providers available")

    async def transcribe(self, audio: bytes, mime: str = "audio/wav", language: str = "en") -> str:
        preferred = self._s.stt_provider.lower()
        order = [preferred] + [p for p in ("gemini", "sarvam") if p != preferred]

        for p_name in order:
            provider = self._stt_providers.get(p_name)
            if provider and await provider.is_healthy():
                try:
                    return await provider.transcribe(audio, mime=mime, language=language)
                except Exception as e:
                    log.warning("STT provider '%s' failed: %s", p_name, e)

        # Final attempt on preferred provider directly
        provider = self._stt_providers.get(preferred) or self._stt_providers["gemini"]
        return await provider.transcribe(audio, mime=mime, language=language)

    async def synthesize(self, text: str, language: str = "en", voice: str = "Kore") -> bytes:
        preferred = self._s.tts_provider.lower()
        order = [preferred] + [p for p in ("gemini", "sarvam") if p != preferred]

        for p_name in order:
            provider = self._tts_providers.get(p_name)
            if provider and await provider.is_healthy():
                try:
                    return await provider.synthesize(text, language=language, voice=voice)
                except Exception as e:
                    log.warning("TTS provider '%s' failed: %s", p_name, e)

        provider = self._tts_providers.get(preferred) or self._tts_providers["gemini"]
        return await provider.synthesize(text, language=language, voice=voice)

    async def check_health(self) -> dict[str, Any]:
        results: list[ProviderHealth] = []

        # Check LLMs
        for name, provider in self._llm_providers.items():
            healthy = await provider.is_healthy()
            results.append(ProviderHealth(provider=name, service_type="llm", available=healthy))

        # Check STTs
        for name, provider in self._stt_providers.items():
            healthy = await provider.is_healthy()
            results.append(ProviderHealth(provider=name, service_type="stt", available=healthy))

        # Check TTSs
        for name, provider in self._tts_providers.items():
            healthy = await provider.is_healthy()
            results.append(ProviderHealth(provider=name, service_type="tts", available=healthy))

        return {
            "ok": any(r.available for r in results if r.service_type == "llm"),
            "providers": [
                {
                    "provider": r.provider,
                    "type": r.service_type,
                    "available": r.available,
                }
                for r in results
            ],
            "llm_order": [name for name, _ in self.get_llm_order()],
            "stt_active": self._s.stt_provider,
            "tts_active": self._s.tts_provider,
        }
