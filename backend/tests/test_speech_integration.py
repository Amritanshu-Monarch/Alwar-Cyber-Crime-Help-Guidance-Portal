"""Real Gemini API integration tests for Speech.synthesize().

REQUIRES: A valid GOOGLE_API_KEY in backend/.env
EXCLUDED from default pytest run (marked with @pytest.mark.integration).

Run manually:
    cd backend
    uv run pytest tests/test_speech_integration.py -v -m integration

These tests verify:
- Real Gemini TTS connectivity with gemini-3.8-flash-tts
- Non-silent audio output for English and Hindi text
- Correct WAV format from real API response (1 channel, 16-bit, 24000 Hz)

These tests are NOT automated in CI. Results must be manually inspected.
Do NOT report these as passing based on mocked tests.
"""
import asyncio
import io
import wave

import pytest

from app.config import get_settings
from app.services.speech import Speech, TTS_SAMPLE_RATE


@pytest.mark.integration
def test_real_tts_english():
    """Real API: English TTS returns non-silent WAV audio with correct parameters."""
    s = get_settings()
    if not s.google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")

    sp = Speech(s)
    wav_bytes = asyncio.run(sp.synthesize("Call 1930 for immediate cyber crime help."))

    assert wav_bytes[:4] == b"RIFF", "Must start with RIFF WAV header"
    assert b"WAVE" in wav_bytes[:12]

    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getnchannels() == 1, "Must be mono"
        assert w.getsampwidth() == 2, "Must be 16-bit"
        assert w.getframerate() == TTS_SAMPLE_RATE, f"Must be {TTS_SAMPLE_RATE} Hz"
        assert w.getnframes() > 1000, "Must contain substantial audio frames"
        frames = w.readframes(w.getnframes())
        assert any(b != 0 for b in frames), "Audio frames are all zero — TTS produced silence"


@pytest.mark.integration
def test_real_tts_hindi():
    """Real API: Hindi TTS returns non-silent WAV audio with correct parameters."""
    s = get_settings()
    if not s.google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")

    sp = Speech(s)
    wav_bytes = asyncio.run(sp.synthesize("अभी 1930 पर कॉल करें।"))

    assert wav_bytes[:4] == b"RIFF", "Must start with RIFF WAV header"

    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getnchannels() == 1, "Must be mono"
        assert w.getsampwidth() == 2, "Must be 16-bit"
        assert w.getframerate() == TTS_SAMPLE_RATE, f"Must be {TTS_SAMPLE_RATE} Hz"
        frames = w.readframes(w.getnframes())
        assert any(b != 0 for b in frames), "Hindi TTS produced silence"
