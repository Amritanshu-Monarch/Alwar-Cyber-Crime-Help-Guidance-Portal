"""Sarvam AI provider: Indian multilingual STT (Saaras), TTS (Bulbul), and LLM (sarvam-105b-conversations).
Requires SARVAM_API_KEY from https://www.sarvam.ai.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, AsyncIterator

import httpx

from app.agent.abstention import sanitize_high_risk_claims
from app.config import Settings
from app.providers.base import (
    LLMProvider,
    LLMResponse,
    STTProvider,
    TTSProvider,
    split_into_sentences,
)
from app.services.speech import clean_for_speech, clean_transcript

log = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai"
SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


class SarvamLLMProvider(LLMProvider):
    name: str = "sarvam"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.sarvam_api_key
        # Use sarvam-105b-conversations as documented (deprecated models like Sarvam M / 30B are excluded)
        self._model = getattr(settings, "sarvam_model", "sarvam-105b-conversations") or "sarvam-105b-conversations"

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
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
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
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

    async def stream_generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> AsyncIterator[str]:
        """Stream LLM response tokens as Server-Sent Events (SSE)."""
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
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            async with client.stream(
                "POST", f"{SARVAM_BASE_URL}/v1/chat/completions", headers=headers, json=payload
            ) as resp:
                if resp.status_code >= 400:
                    err_body = await resp.aread()
                    log.warning("Sarvam streaming chat failed (%s): %s", resp.status_code, err_body.decode(errors="replace"))
                    resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue

    async def is_healthy(self) -> bool:
        return bool(self._key)


class SarvamSTTProvider(STTProvider):
    name: str = "sarvam"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.sarvam_api_key
        # Use saaras:v3 or saaras:v4 (saaras:v1 and saarika are deprecated)
        self._model = getattr(settings, "sarvam_stt_model", "saaras:v3") or "saaras:v3"

    async def transcribe(self, audio: bytes, mime: str = "audio/wav", language: str = "en") -> str:
        """Batch / REST transcription via Saaras endpoint."""
        if not self._key:
            raise ValueError("SARVAM_API_KEY not configured")

        headers = {
            "api-subscription-key": self._key,
        }
        # Saaras model language code: 'hi-IN', 'en-IN', or 'unknown' for auto-detection
        lang_code = "hi-IN" if language in ("hi", "hinglish") else "en-IN"
        files = {
            "file": ("audio.wav", audio, mime),
        }
        data = {
            "model": self._model,
            "language_code": lang_code,
            "with_diarization": "false",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{SARVAM_BASE_URL}/speech-to-text", headers=headers, files=files, data=data)
            if resp.status_code >= 400:
                log.warning("Sarvam STT failed (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            res_data = resp.json()

        transcript = res_data.get("transcript", "")
        return clean_transcript(transcript)

    def create_realtime_session(self, language_code: str = "hi-IN") -> SarvamRealtimeSTT:
        """Create a real-time streaming STT session using Saaras WebSocket."""
        return SarvamRealtimeSTT(self._s, language_code=language_code)

    async def is_healthy(self) -> bool:
        return bool(self._key)


class SarvamRealtimeSTT:
    """Manages real-time streaming speech-to-text over WebSocket with Sarvam Saaras.
    Endpoint: wss://api.sarvam.ai/speech-to-text-realtime/ws
    Model: saaras:v3-realtime (or saaras:v4)
    Audio input: 16000Hz mono PCM encoded in base64.
    """

    def __init__(self, settings: Settings, language_code: str = "hi-IN"):
        self._s = settings
        self._key = settings.sarvam_api_key
        self._model = getattr(settings, "sarvam_realtime_stt_model", "saaras:v3-realtime") or "saaras:v3-realtime"
        self._language_code = language_code
        self._ws: Any = None
        self._closed = False

    async def connect(self) -> bool:
        """Connect to Sarvam real-time STT WebSocket."""
        if not self._key:
            log.warning("SARVAM_API_KEY not set; skipping real-time WebSocket STT")
            return False
        try:
            import websockets

            url = (
                f"{SARVAM_WS_URL}?model={self._model}"
                f"&language_code={self._language_code}&sample_rate=16000"
            )
            # Authenticate via api-subscription-key header
            self._ws = await websockets.connect(
                url,
                additional_headers={"api-subscription-key": self._key},
                open_timeout=5.0,
                close_timeout=3.0,
            )
            self._closed = False
            log.info("Connected to Sarvam real-time STT WebSocket (%s)", self._model)
            return True
        except Exception as e:
            log.warning("Could not connect to Sarvam real-time STT (%s): %s", self._model, e)
            self._ws = None
            return False

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send 16kHz PCM audio chunk to Sarvam STT."""
        if not self._ws or self._closed:
            return
        try:
            b64_audio = base64.b64encode(pcm_data).decode("ascii")
            msg = json.dumps({"event": "audio_input", "audio": b64_audio})
            await self._ws.send(msg)
        except Exception as e:
            log.debug("Error sending audio to Sarvam STT ws: %s", e)

    async def send_eof(self) -> None:
        """Send end of audio stream event."""
        if not self._ws or self._closed:
            return
        try:
            await self._ws.send(json.dumps({"event": "eof"}))
        except Exception:
            pass

    async def receive_events(self) -> AsyncIterator[dict[str, Any]]:
        """Listen for partial and final transcription events from Sarvam."""
        if not self._ws or self._closed:
            return
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    payload = json.loads(raw)
                    yield payload
                except Exception:
                    continue
        except Exception as e:
            log.debug("Sarvam STT websocket receive ended: %s", e)

    async def close(self) -> None:
        """Cleanly close connection."""
        self._closed = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


class SarvamTTSProvider(TTSProvider):
    name: str = "sarvam"

    def __init__(self, settings: Settings):
        self._s = settings
        self._key = settings.sarvam_api_key
        # Use bulbul:v4-flash for low-latency streaming (sub-250ms), bulbul:v3 as fallback
        self._model = getattr(settings, "sarvam_tts_model", "bulbul:v4-flash") or "bulbul:v4-flash"
        self._speaker = getattr(settings, "sarvam_speaker", "kavya_hi_conversational") or "kavya_hi_conversational"
        if self._speaker == "meera":
            self._speaker = "kavya_hi_conversational"

    async def synthesize(self, text: str, language: str = "en", voice: str = "") -> bytes:
        """Synthesize text into complete audio bytes."""
        if not self._key:
            raise ValueError("SARVAM_API_KEY not configured")

        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return b""

        # Format digits so phone numbers are pronounced cleanly (1930, 8764874306)
        spoken_text = re.sub(r"\b([6-9]\d{9})\b", r"\1", spoken_text)

        headers = {
            "api-subscription-key": self._key,
            "Content-Type": "application/json",
        }
        lang_code = "hi-IN" if language in ("hi", "hinglish") else "en-IN"
        stream_payload = {
            "text": spoken_text[:1500],
            "target_language_code": lang_code,
            "speaker": voice or self._speaker,
            "model": self._model,
            "pace": 1.0,
            "speech_sample_rate": 22050,
            "audio_codec": "mp3",
        }
        batch_payload = {
            "inputs": [spoken_text[:1500]],
            "target_language_code": lang_code,
            "speaker": voice or self._speaker,
            "model": self._model,
            "pace": 1.0,
            "speech_sample_rate": 22050,
            "audio_codec": "mp3",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            # First try streaming endpoint for raw binary audio
            try:
                resp = await client.post(f"{SARVAM_BASE_URL}/text-to-speech/stream", headers=headers, json=stream_payload)
                if resp.status_code == 200 and resp.content:
                    return resp.content
            except Exception as e:
                log.debug("Sarvam /text-to-speech/stream failed: %s; falling back to standard endpoint", e)

            # Standard endpoint fallback
            resp = await client.post(f"{SARVAM_BASE_URL}/text-to-speech", headers=headers, json=batch_payload)
            if resp.status_code >= 400:
                log.warning("Sarvam TTS failed (%s): %s", resp.status_code, resp.text)
                resp.raise_for_status()
            res_data = resp.json()

        audios = res_data.get("audios", [])
        if not audios or not audios[0]:
            raise RuntimeError("Sarvam TTS returned empty audio list")

        return base64.b64decode(audios[0])

    async def stream_synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str = "",
    ) -> AsyncIterator[bytes]:
        """Stream raw binary audio chunks via Bulbul HTTP streaming endpoint."""
        if not self._key:
            raise ValueError("SARVAM_API_KEY not configured")

        spoken_text = clean_for_speech(text)
        if not spoken_text:
            return

        headers = {
            "api-subscription-key": self._key,
            "Content-Type": "application/json",
        }
        lang_code = "hi-IN" if language in ("hi", "hinglish") else "en-IN"
        payload = {
            "text": spoken_text[:1500],
            "target_language_code": lang_code,
            "speaker": voice or self._speaker,
            "model": self._model,
            "pace": 1.0,
            "speech_sample_rate": 22050,
            "audio_codec": "mp3",
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            async with client.stream(
                "POST", f"{SARVAM_BASE_URL}/text-to-speech/stream", headers=headers, json=payload
            ) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    log.warning("Sarvam stream TTS failed (%s): %s", resp.status_code, err.decode(errors="replace"))
                    resp.raise_for_status()

                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk

    async def stream_synthesize_sentences(
        self,
        full_text: str,
        language: str = "en",
        voice: str = "",
    ) -> AsyncIterator[tuple[str, bytes]]:
        """Sentence-by-sentence synthesis:
        1. Splits response into clean sentence boundaries.
        2. Sanitizes EACH sentence before speech generation to ensure safety guardrails.
        3. Synthesizes and yields (sentence_text, audio_bytes) so audio playback starts in ~250ms.
        """
        sentences = split_into_sentences(full_text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            # Safety Guardrail: Sanitize high-risk claims on EACH sentence before TTS
            sanitized_sentence, _ = sanitize_high_risk_claims(sentence, lang=language)
            try:
                audio = await self.synthesize(sanitized_sentence, language=language, voice=voice)
                if audio:
                    yield (sanitized_sentence, audio)
            except Exception as e:
                log.warning("Sentence synthesis failed for '%s': %s", sanitized_sentence[:40], e)

    async def is_healthy(self) -> bool:
        return bool(self._key)
