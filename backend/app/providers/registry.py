"""Provider Registry and Orchestrator.
Manages LLM fallback order (429/timeout/5xx), STT/TTS provider selection,
and low-latency streaming pipelines.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, AsyncIterator

from app.agent.abstention import sanitize_high_risk_claims
from app.config import Settings
from app.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderHealth,
    STTProvider,
    TTSProvider,
    split_into_sentences,
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
        self._last_errors: dict[str, dict[str, Any]] = {}
        self._llm_providers: dict[str, LLMProvider] = {
            "gemini": GeminiLLMProvider(settings),
            "sarvam": SarvamLLMProvider(settings),
            "groq": GroqProvider(settings),
            "openrouter": OpenRouterProvider(settings),
            "ollama": OllamaProvider(settings),
        }
        self._stt_providers: dict[str, STTProvider] = {
            "sarvam": SarvamSTTProvider(settings),
            "gemini": GeminiSTTProvider(settings),
        }
        self._tts_providers: dict[str, TTSProvider] = {
            "sarvam": SarvamTTSProvider(settings),
            "gemini": GeminiTTSProvider(settings),
        }

    def record_error(
        self,
        provider: str,
        service_type: str,
        error: Exception | str,
        status_code: int | None = None,
    ) -> None:
        sc = status_code
        if sc is None:
            sc = getattr(error, "status_code", getattr(error, "code", None))
        reason = str(error)
        # Redact secrets from error messages before storing/logging
        reason = re.sub(r"key=[a-zA-Z0-9_\-]{8,}", "key=[REDACTED]", reason)
        reason = re.sub(
            r"(bearer|api-subscription-key|apikey|token)[:\s=]+[a-zA-Z0-9_\-]{8,}",
            r"\1: [REDACTED]",
            reason,
            flags=re.IGNORECASE,
        )
        reason = re.sub(r"\b[a-zA-Z0-9_\-]{35,}\b", "[REDACTED_TOKEN]", reason)
        entry = {
            "time": time.time(),
            "service": service_type,
            "status_code": sc,
            "reason": reason[:300],
        }
        self._last_errors[provider] = entry
        log.warning(
            "Provider '%s' (%s) failed (status=%s): %s; falling back",
            provider,
            service_type,
            sc,
            entry["reason"],
        )

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
        max_tokens: int = 1000,
    ) -> LLMResponse:
        order = self.get_llm_order()
        last_error = None

        for name, provider in order:
            try:
                # For small local models like Ollama, do not send tools
                active_tools = None if name in ("ollama", "sarvam") else tools
                return await provider.generate(
                    messages=messages,
                    tools=active_tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                self.record_error(name, "llm", e)
                last_error = e

        if last_error:
            raise last_error
        raise RuntimeError("No LLM providers available")

    async def stream_generate_with_fallback(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """Stream response tokens from the highest-priority reachable provider."""
        order = self.get_llm_order()
        last_error = None

        for name, provider in order:
            try:
                tokens_yielded = False
                async for token in provider.stream_generate(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    tokens_yielded = True
                    yield token
                if tokens_yielded:
                    return
            except Exception as e:
                self.record_error(name, "llm_stream", e)
                last_error = e

        if last_error:
            raise last_error

    async def transcribe(self, audio: bytes, mime: str = "audio/wav", language: str = "en") -> str:
        preferred = self._s.stt_provider.lower()
        order = [preferred] + [p for p in ("sarvam", "gemini") if p != preferred]

        for p_name in order:
            provider = self._stt_providers.get(p_name)
            if provider and await provider.is_healthy():
                try:
                    return await provider.transcribe(audio, mime=mime, language=language)
                except Exception as e:
                    self.record_error(p_name, "stt", e)

        # Final attempt on preferred provider directly
        provider = self._stt_providers.get(preferred) or self._stt_providers["gemini"]
        return await provider.transcribe(audio, mime=mime, language=language)

    async def synthesize(self, text: str, language: str = "en", voice: str = "") -> bytes:
        preferred = self._s.tts_provider.lower()
        order = [preferred] + [p for p in ("sarvam", "gemini") if p != preferred]

        for p_name in order:
            provider = self._tts_providers.get(p_name)
            if provider and await provider.is_healthy():
                try:
                    return await provider.synthesize(text, language=language, voice=voice)
                except Exception as e:
                    self.record_error(p_name, "tts", e)

        provider = self._tts_providers.get(preferred) or self._tts_providers["gemini"]
        return await provider.synthesize(text, language=language, voice=voice)

    async def stream_synthesize_sentences(
        self,
        text: str,
        language: str = "en",
        voice: str = "",
    ) -> AsyncIterator[tuple[str, bytes]]:
        """Sentence-by-sentence streaming TTS for live audio playback.
        Synthesizes the first sentence immediately so audio begins playing in ~250ms.
        """
        preferred = self._s.tts_provider.lower()
        sarvam_tts = self._tts_providers.get("sarvam")
        if preferred == "sarvam" and isinstance(sarvam_tts, SarvamTTSProvider):
            if await sarvam_tts.is_healthy():
                try:
                    chunks_yielded = False
                    async for sentence, audio_chunk in sarvam_tts.stream_synthesize_sentences(
                        text, language=language, voice=voice
                    ):
                        chunks_yielded = True
                        yield (sentence, audio_chunk)
                    if chunks_yielded:
                        return
                except Exception as e:
                    self.record_error("sarvam", "tts_stream", e)

        # Fallback: synthesize each sentence with Gemini TTS
        gemini_tts = self._tts_providers.get("gemini")
        if gemini_tts:
            sentences = split_into_sentences(text)
            for s in sentences:
                sanitized_s, _ = sanitize_high_risk_claims(s, lang=language)
                try:
                    audio = await gemini_tts.synthesize(sanitized_s, language=language, voice=voice or "Kore")
                    if audio:
                        yield (sanitized_s, audio)
                except Exception as e:
                    self.record_error("gemini", "tts_sentence", e)

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
            "last_errors": self._last_errors,
        }
