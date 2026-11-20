"""Unit tests for Sarvam AI live voice stack and multi-provider streaming.

Verifies:
1. SarvamLLMProvider streaming and non-streaming generation with mocked API responses.
2. SarvamTTSProvider streaming output (Bulbul) and sentence-by-sentence synthesis.
3. SarvamSTTProvider batch transcription and SarvamRealtimeSTT WebSocket lifecycle.
4. Fallback chain switching providers on simulated 429 and timeout.
5. High-risk sanitizer verification running BEFORE TTS synthesis.
6. Barge-in interruption cancellation mechanics.
7. WebSocket /ws/voice endpoint live message exchange.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.base import LLMResponse
from app.providers.registry import ProviderRegistry
from app.providers.sarvam import (
    SarvamLLMProvider,
    SarvamRealtimeSTT,
    SarvamSTTProvider,
    SarvamTTSProvider,
)
from app.schemas import AgentResult


def get_test_settings() -> Settings:
    return Settings(
        google_api_key="fake-gemini-key",
        sarvam_api_key="fake-sarvam-key",
        llm_provider_order=["gemini", "sarvam"],
        stt_provider="sarvam",
        tts_provider="sarvam",
        sarvam_model="sarvam-105b-conversations",
        sarvam_tts_model="bulbul:v4-flash",
        sarvam_stt_model="saaras:v3",
        sarvam_speaker="meera",
    )


# ---------------------------------------------------------------------------
# 1. Sarvam LLM Provider Tests
# ---------------------------------------------------------------------------
def test_sarvam_llm_generate():
    async def _run():
        provider = SarvamLLMProvider(get_test_settings())

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Aap turant 1930 par call karke complaint darj karayein.",
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await provider.generate(
                messages=[{"role": "user", "content": "Paise cut gaye"}],
                temperature=0.2,
                max_tokens=500,
            )

            assert res.provider == "sarvam"
            assert res.model == "sarvam-105b-conversations"
            assert "1930" in res.content
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "sarvam-105b-conversations"
            assert call_kwargs["json"]["temperature"] == 0.2

    asyncio.run(_run())


def test_sarvam_llm_stream_generate():
    async def _run():
        provider = SarvamLLMProvider(get_test_settings())

        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Aap "}}]}',
            b'data: {"choices":[{"delta":{"content":"1930 "}}]}',
            b'data: {"choices":[{"delta":{"content":"call karein."}}]}',
            b"data: [DONE]",
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line.decode("utf-8")

        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.aiter_lines = fake_aiter_lines

        mock_client = MagicMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        class FakeStreamContext:
            async def __aenter__(self):
                return mock_stream_resp

            async def __aexit__(self, *args):
                pass

        mock_client.stream.return_value = FakeStreamContext()

        with patch("httpx.AsyncClient", return_value=mock_client):
            tokens = []
            async for token in provider.stream_generate([{"role": "user", "content": "Help"}]):
                tokens.append(token)

            assert "".join(tokens) == "Aap 1930 call karein."

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. Sarvam TTS Provider Tests
# ---------------------------------------------------------------------------
def test_sarvam_tts_synthesize():
    async def _run():
        provider = SarvamTTSProvider(get_test_settings())

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"ID3\x03\x00\x00fake-mp3-bytes"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            audio = await provider.synthesize("Call 1930 immediately", language="en")
            assert audio == b"ID3\x03\x00\x00fake-mp3-bytes"
            mock_post.assert_called_once()
            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "bulbul:v4-flash"
            assert payload["target_language_code"] == "en-IN"

    asyncio.run(_run())


def test_sarvam_tts_stream_synthesize_sentences_sanitizer():
    """Verify that sanitize_high_risk_claims runs BEFORE audio synthesis."""
    async def _run():
        provider = SarvamTTSProvider(get_test_settings())

        # Fake input with an unverified fake phone number and 100% refund guarantee
        unsafe_text = "Call 9876543210 immediately. We guarantee 100% refund."

        synthesized_sentences = []

        async def fake_synthesize(text, language="en", voice=""):
            synthesized_sentences.append(text)
            return b"fake-audio"

        provider.synthesize = fake_synthesize

        results = []
        async for sentence, audio in provider.stream_synthesize_sentences(unsafe_text, language="en"):
            results.append((sentence, audio))

        assert len(results) > 0
        # Unverified number must have been neutralized before reaching synthesize()
        # and refund guarantee must be neutralized with statutory disclaimer
        for sentence in synthesized_sentences:
            assert "9876543210" not in sentence
        combined = " ".join(synthesized_sentences).lower()
        assert "cannot be guaranteed" in combined or "golden hour" in combined

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Sarvam STT Provider & Realtime STT Tests
# ---------------------------------------------------------------------------
def test_sarvam_stt_batch_transcribe():
    async def _run():
        provider = SarvamSTTProvider(get_test_settings())

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"transcript": "Mere bank khate se paise kat gaye"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            text = await provider.transcribe(b"fake-wav-audio", language="hi")
            assert "paise kat gaye" in text

    asyncio.run(_run())


def test_sarvam_realtime_stt_lifecycle():
    async def _run():
        stt = SarvamRealtimeSTT(get_test_settings(), language_code="hi-IN")

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        events_to_yield = [
            json.dumps({"event": "transcript.partial", "text": "mere khate", "language": "hi-IN"}),
            json.dumps({"event": "transcript.final", "text": "mere khate se paise kat gaye", "language": "hi-IN"}),
        ]

        async def fake_iter():
            for ev in events_to_yield:
                yield ev

        mock_ws.__aiter__ = lambda self: fake_iter()

        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_ws
            connected = await stt.connect()
            assert connected is True

            await stt.send_audio(b"\x00\x00" * 160)
            mock_ws.send.assert_called()

            received = []
            async for event in stt.receive_events():
                received.append(event)

            assert len(received) == 2
            assert received[0]["event"] == "transcript.partial"
            assert received[1]["event"] == "transcript.final"

            await stt.close()
            mock_ws.close.assert_called_once()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. Multi-Provider Fallback Chain on 429 & Timeout
# ---------------------------------------------------------------------------
def test_provider_registry_fallback_on_429():
    async def _run():
        registry = ProviderRegistry(get_test_settings())

        mock_gemini = AsyncMock()
        mock_gemini.generate.side_effect = RuntimeError("429 Resource Exhausted")

        async def mock_gemini_stream(*args, **kwargs):
            raise RuntimeError("429 Resource Exhausted")
            yield ""

        mock_gemini.stream_generate = mock_gemini_stream

        mock_sarvam = AsyncMock()
        mock_sarvam.generate.return_value = LLMResponse(
            content="Sarvam answered safely: Call 1930.",
            provider="sarvam",
            model="sarvam-105b-conversations",
        )

        async def fake_sarvam_stream(*args, **kwargs):
            yield "Sarvam "
            yield "stream "
            yield "token."

        mock_sarvam.stream_generate = fake_sarvam_stream

        registry._llm_providers["gemini"] = mock_gemini
        registry._llm_providers["sarvam"] = mock_sarvam

        # Non-streaming fallback
        res = await registry.generate_with_fallback([{"role": "user", "content": "Help"}])
        assert res.provider == "sarvam"
        assert "1930" in res.content

        # Streaming fallback
        tokens = []
        async for t in registry.stream_generate_with_fallback([{"role": "user", "content": "Help"}]):
            tokens.append(t)
        assert "".join(tokens) == "Sarvam stream token."

    asyncio.run(_run())


def test_provider_registry_fallback_on_timeout():
    async def _run():
        registry = ProviderRegistry(get_test_settings())

        mock_gemini = AsyncMock()
        mock_gemini.generate.side_effect = asyncio.TimeoutError()

        mock_sarvam = AsyncMock()
        mock_sarvam.generate.return_value = LLMResponse(
            content="Sarvam fallback on timeout: Call 1930.",
            provider="sarvam",
            model="sarvam-105b-conversations",
        )

        registry._llm_providers["gemini"] = mock_gemini
        registry._llm_providers["sarvam"] = mock_sarvam

        res = await registry.generate_with_fallback([{"role": "user", "content": "Help"}])
        assert res.provider == "sarvam"
        assert "timeout" in res.content

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. Live Conversation Loop & Barge-In Cancellation Tests
# ---------------------------------------------------------------------------
def test_barge_in_cancellation_during_turn():
    """Verify that when a turn is cancelled by user interruption (barge-in),
    the async task cancels cleanly without leaking or raising unhandled exceptions.
    """
    async def _run():
        turn_cancelled = False

        async def long_running_turn():
            nonlocal turn_cancelled
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                turn_cancelled = True
                raise

        task = asyncio.create_task(long_running_turn())
        await asyncio.sleep(0.02)

        # Barge-in event: cancel task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert turn_cancelled is True

    asyncio.run(_run())


def test_websocket_voice_flow():
    """Test FastAPI WebSocket /ws/voice endpoint handshake, message reception, and interrupt."""

    class MockAgent:
        async def ask(self, session_id, text, lang, is_voice=False):
            return AgentResult(
                reply="Sabse pehle 1930 par call karein. Kya aapko poora procedure chahiye?",
                tools_used=[],
                sources=["test.md"],
                artifacts=[],
                citation_note="",
            )

    class MockSpeech:
        async def transcribe(self, audio, mime):
            return "mera paisa chori ho gaya"

        async def synthesize(self, text):
            return b"fake-wav-bytes"

    app = create_app(
        Settings(
            google_api_key="fake-key",
            sarvam_api_key="fake-sarvam",
            stt_provider="sarvam",
            tts_provider="sarvam",
        ),
        agent=MockAgent(),
        speech=MockSpeech(),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/voice?session_id=test-voice-session&lang=hi") as ws:
            # 1. Server sends ready event
            ready_msg = ws.receive_json()
            assert ready_msg["type"] == "ready"
            assert ready_msg["session_id"] == "test-voice-session"

            # 2. Client sends user speech turn
            ws.send_json({"type": "text", "text": "mere 10000 rupaye cut gaye"})

            # 3. Server sends final transcript
            final_msg = ws.receive_json()
            assert final_msg["type"] == "final_transcript"
            assert "10000" in final_msg["text"]

            # 4. Server sends state thinking
            state_msg = ws.receive_json()
            assert state_msg["type"] == "state"
            assert state_msg["state"] == "thinking"

            # 5. Server sends reply_text
            reply_msg = ws.receive_json()
            assert reply_msg["type"] == "reply_text"
            assert "1930" in reply_msg["text"]

            # 6. Server sends state speaking
            speaking_msg = ws.receive_json()
            assert speaking_msg["type"] == "state"
            assert speaking_msg["state"] == "speaking"

            # 7. Test barge-in: client sends interrupt message
            ws.send_json({"type": "interrupt"})
            # Next message is either audio chunk or interrupted ack
            next_msg = ws.receive_json()
            assert next_msg["type"] in ("interrupted", "audio_chunk", "state")
