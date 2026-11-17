"""Sarvam AI provider: Indian multilingual STT (Saaras), TTS (Bulbul), and LLM (Sarvam-2B).
Requires SARVAM_API_KEY from https://www.sarvam.ai.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from app.config import Settings
from app.providers.base import (
    LLMProvider,
    LLMResponse,
    STTProvider,
    TTSProvider,
)
from app.services.speech import clean_for_speech, clean_transcript

log = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai"


class SarvamLLMProvider(LLMProvider):
    name: str = "sarvam"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.sarvam_api_key
        self._model = settings.sarvam_model or "sarvam-2b"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        if not self._key:
            raise ValueError("SARVAM_API_KEY not configured")

        headers = {
            "api-subscription-key": self._key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{SARVAM_BASE_URL}/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code >= 400:
                log.warning("Sarvam chat failed with status %s: %s", resp.status_code, resp.text)
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


class SarvamSTTProvider(STTProvider):
    name: str = "sarvam"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.sarvam_api_key

    async def transcribe(self, audio: bytes, mime: str = "audio/wav", language: str = "en") -> str:
        if not self._key:
            raise ValueError("SARVAM_API_KEY not configured")

        headers = {
            "api-subscription-key": self._key,
        }
        # Sarvam Saaras model accepts language code (e.g. 'hi-IN', 'en-IN', or 'unknown')
        lang_code = "hi-IN" if language == "hi" else "en-IN"
        files = {
            "file": ("audio.wav", audio, mime),
        }
        data = {
            "model": "saaras:v1",
            "language_code": lang_code,
            "with_diarization": "false",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{SARVAM_BASE_URL}/speech-to-text", headers=headers, files=files, data=data)
            if resp.status_code >= 400:
                log.warning("Sarvam STT failed (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            res_data = resp.json()

        transcript = res_data.get("transcript", "")
        return clean_transcript(transcript)

    async def is_healthy(self) -> bool:
        return bool(self._key)


class SarvamTTSProvider(TTSProvider):
    name: str = "sarvam"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.sarvam_api_key

    async def synthesize(self, text: str, language: str = "en", voice: str = "Kore") -> bytes:
        if not self._key:
            raise ValueError("SARVAM_API_KEY not configured")

        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return b""

        # Cap length to Sarvam TTS input limits (~500 chars per chunk)
        spoken_text = spoken_text[:500]

        headers = {
            "api-subscription-key": self._key,
            "Content-Type": "application/json",
        }
        lang_code = "hi-IN" if language in ("hi", "hinglish") else "en-IN"
        payload = {
            "inputs": [spoken_text],
            "target_language_code": lang_code,
            "speaker": "meera",
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 24000,
            "enable_preprocessing": True,
            "model": "bulbul:v1",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{SARVAM_BASE_URL}/text-to-speech", headers=headers, json=payload)
            if resp.status_code >= 400:
                log.warning("Sarvam TTS failed (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            res_data = resp.json()

        audios = res_data.get("audios", [])
        if not audios or not audios[0]:
            raise RuntimeError("Sarvam TTS returned empty audio list")

        return base64.b64decode(audios[0])

    async def is_healthy(self) -> bool:
        return bool(self._key)
