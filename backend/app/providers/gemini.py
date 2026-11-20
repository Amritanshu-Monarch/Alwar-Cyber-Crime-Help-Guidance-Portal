"""Gemini implementation for LLM, STT, and TTS using Google GenAI SDK."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

from google import genai
from google.genai import types

from app.config import Settings
from app.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    STTProvider,
    TTSProvider,
)
from app.services.speech import (
    STT_PROMPT,
    TTS_SAMPLE_RATE,
    clean_for_speech,
    clean_transcript,
    pcm_to_wav,
)

log = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    name: str = "gemini"

    def __init__(self, settings: Settings, model: str | None = None):
        self._s = settings
        self._model = model or settings.agent_model
        self._client = genai.Client(api_key=settings.google_api_key or None)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        system_instruction = ""
        contents = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_instruction = content
            elif role in ("user", "human"):
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=str(content))]))
            elif role in ("assistant", "model"):
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=str(content))]))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction or None,
        )

        def _call():
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

        resp = await asyncio.to_thread(_call)
        text = resp.text or ""
        return LLMResponse(
            content=text.strip(),
            tool_calls=[],
            finish_reason="stop",
            provider=self.name,
            model=self._model,
        )

    async def is_healthy(self) -> bool:
        return bool(self._s.google_api_key)


class GeminiSTTProvider(STTProvider):
    name: str = "gemini"

    def __init__(self, settings: Settings):
        self._s = settings
        self._client = genai.Client(api_key=settings.google_api_key or None)

    async def transcribe(self, audio: bytes, mime: str = "audio/wav", language: str = "en") -> str:
        def _call():
            return self._client.models.generate_content(
                model=self._s.stt_model,
                contents=[STT_PROMPT, types.Part.from_bytes(data=audio, mime_type=mime)],
                config=types.GenerateContentConfig(temperature=0.0),
            )

        resp = await asyncio.to_thread(_call)
        return clean_transcript(resp.text or "")

    async def is_healthy(self) -> bool:
        return bool(self._s.google_api_key)


class GeminiTTSProvider(TTSProvider):
    name: str = "gemini"

    def __init__(self, settings: Settings):
        self._s = settings
        self._client = genai.Client(api_key=settings.google_api_key or None)

    async def synthesize(self, text: str, language: str = "en", voice: str = "Kore") -> bytes:
        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return b""

        def _call():
            return self._client.models.generate_content(
                model=self._s.tts_model,
                contents=spoken_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice or self._s.tts_voice)
                        )
                    ),
                ),
            )

        resp = await asyncio.to_thread(_call)
        for part in getattr(resp, "parts", []) or []:
            if part.inline_data and part.inline_data.data:
                return pcm_to_wav(part.inline_data.data, rate=TTS_SAMPLE_RATE)
        for cand in resp.candidates or []:
            for part in (cand.content and cand.content.parts) or []:
                if part.inline_data and part.inline_data.data:
                    return pcm_to_wav(part.inline_data.data, rate=TTS_SAMPLE_RATE)

        raise RuntimeError("Gemini TTS returned no audio data")

    async def is_healthy(self) -> bool:
        return bool(self._s.google_api_key)
