"""Unit tests for Settings model defaults.
These are pure unit tests — no API key or network required.
Verify that config.py model name defaults match verified Gemini model identifiers
(aistudio.google.com/docs/models, verified 2026-11-16).
"""
from app.config import Settings


def test_default_agent_model_is_valid_name():
    s = Settings(google_api_key="x")
    assert s.agent_model in ("gemini-3.8-flash", "gemini-3.5-flash-lite")


def test_default_stt_model_is_transcription_model():
    s = Settings(google_api_key="x")
    assert s.stt_model == "gemini-3.5-transcribe"


def test_default_tts_model_unchanged():
    s = Settings(google_api_key="x")
    assert s.tts_model == "gemini-3.8-flash-tts"


def test_default_fallback_model_unchanged():
    s = Settings(google_api_key="x")
    assert s.agent_fallback_model in ("gemini-3.5-flash-lite", "gemini-flash-lite-latest")


def test_env_override_works():
    s = Settings(google_api_key="x", agent_model="gemini-3.5-flash-lite")
    assert s.agent_model == "gemini-3.5-flash-lite"


def test_tts_voice_default():
    s = Settings(google_api_key="x")
    assert s.tts_voice == "Kore"
