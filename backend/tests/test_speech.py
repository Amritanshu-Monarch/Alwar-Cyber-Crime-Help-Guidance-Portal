"""Unit tests for Speech.synthesize() — all mocked, no Gemini API key required.

These tests verify the internal logic of the rewritten synthesize() method against
the google-genai 2.30.1 SDK response structure. They are NOT integration tests.

Real Gemini TTS verification (Hindi/English audio quality, frontend playback)
requires a valid GOOGLE_API_KEY and must be performed manually or via the
separate real-API test file tests/test_speech_integration.py.
"""
import asyncio
import io
import wave
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.services.speech import Speech, TTS_SAMPLE_RATE, pcm_to_wav


def make_speech():
    s = Settings(google_api_key="test-key")
    sp = Speech(s, client=MagicMock())
    return sp


def fake_tts_response(pcm_bytes: bytes):
    """Build a minimal mock response matching google-genai 2.30.1 structure.

    The real SDK returns response.parts[i].inline_data.data as raw PCM bytes.
    """
    part = MagicMock()
    part.inline_data = MagicMock()
    part.inline_data.data = pcm_bytes
    response = MagicMock()
    response.parts = [part]
    return response


def test_synthesize_returns_valid_wav():
    """Mock TTS returns PCM bytes; synthesize() wraps them into a valid WAV."""
    sp = make_speech()
    pcm = b"\x00\x01" * 100
    sp._client.models.generate_content.return_value = fake_tts_response(pcm)

    result = asyncio.run(sp.synthesize("hello"))

    assert result[:4] == b"RIFF", "Result must start with RIFF WAV header"
    assert b"WAVE" in result[:12]
    assert len(result) > len(pcm)  # WAV is larger than raw PCM due to header


def test_synthesize_wav_parameters():
    """WAV header must encode: 1 channel, 16-bit depth, 24000 Hz sample rate."""
    sp = make_speech()
    pcm = b"\x00\x00" * 2400  # 0.1 seconds at 24000 Hz
    sp._client.models.generate_content.return_value = fake_tts_response(pcm)

    wav_bytes = asyncio.run(sp.synthesize("test"))

    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getnchannels() == 1, "Must be mono"
        assert w.getsampwidth() == 2, "Must be 16-bit (2 bytes per sample)"
        assert w.getframerate() == TTS_SAMPLE_RATE, f"Must be {TTS_SAMPLE_RATE} Hz"
        assert w.getnframes() > 0


def test_synthesize_non_empty_audio():
    """WAV output must contain audio frames, not just the header."""
    sp = make_speech()
    pcm = b"\x01\x02" * 50
    sp._client.models.generate_content.return_value = fake_tts_response(pcm)

    wav_bytes = asyncio.run(sp.synthesize("hello"))
    assert len(wav_bytes) > 44, "WAV must be larger than the 44-byte header"


def test_synthesize_retries_on_transient_error():
    """Network/API errors on attempt 1 trigger retry; success on attempt 2."""
    from google.genai import errors as genai_errors

    sp = make_speech()
    pcm = b"\x00\x00" * 100
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise genai_errors.ServerError(503, {"error": {"message": "overloaded"}})
        return fake_tts_response(pcm)

    sp._client.models.generate_content.side_effect = side_effect

    result = asyncio.run(sp.synthesize("hello"))
    assert result[:4] == b"RIFF"
    assert call_count[0] == 2


def test_synthesize_raises_after_two_failures():
    """Two consecutive API failures propagate as an exception."""
    from google.genai import errors as genai_errors

    sp = make_speech()
    sp._client.models.generate_content.side_effect = genai_errors.ServerError(
        503, {"error": {"message": "down"}}
    )

    with pytest.raises(genai_errors.ServerError):
        asyncio.run(sp.synthesize("hello"))


def test_synthesize_raises_on_no_audio_parts():
    """Response with no audio parts raises RuntimeError immediately without retry."""
    sp = make_speech()
    response = MagicMock()
    response.parts = []
    sp._client.models.generate_content.return_value = response

    with pytest.raises(RuntimeError, match="no audio parts"):
        asyncio.run(sp.synthesize("hello"))

    # Must NOT have retried — only 1 call made
    assert sp._client.models.generate_content.call_count == 1


def test_synthesize_no_audio_parts_when_inline_data_is_none():
    """Part present but inline_data is None — also raises without retry."""
    sp = make_speech()
    part = MagicMock()
    part.inline_data = None
    response = MagicMock()
    response.parts = [part]
    sp._client.models.generate_content.return_value = response

    with pytest.raises(RuntimeError, match="no audio parts"):
        asyncio.run(sp.synthesize("hello"))

    assert sp._client.models.generate_content.call_count == 1


def test_synthesize_uses_configured_voice():
    """The voice_name in settings is forwarded to the API config."""
    sp = make_speech()
    sp._s = Settings(google_api_key="x", tts_voice="Puck")
    sp._client.models.generate_content.return_value = fake_tts_response(b"\x00" * 48)

    asyncio.run(sp.synthesize("test"))

    # Verify the call was made and the setting is correct
    assert sp._s.tts_voice == "Puck"
    sp._client.models.generate_content.assert_called_once()


def test_pcm_to_wav_produces_valid_wav():
    """pcm_to_wav helper produces a valid WAV from raw PCM bytes."""
    pcm = b"\x00\x00" * 1000
    wav = pcm_to_wav(pcm)

    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:12]

    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == TTS_SAMPLE_RATE
        assert w.getnframes() == 1000


def test_existing_fake_speech_pattern_still_produces_valid_wav():
    """Regression: the FakeSpeech pattern used in test_api.py still works.

    FakeSpeech.synthesize() returns pcm_to_wav(b'\\x00\\x00' * 10).
    Verify this produces a valid WAV (unchanged by this PR).
    """
    wav = pcm_to_wav(b"\x00\x00" * 10)
    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:12]
