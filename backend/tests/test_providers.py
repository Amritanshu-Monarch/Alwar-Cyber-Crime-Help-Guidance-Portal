"""Unit tests for Phase 1: Provider Abstraction & Health Monitoring.

Verifies:
1. LLMProvider, STTProvider, and TTSProvider interfaces and implementations.
2. Configuration loading and provider order parsing.
3. Fallback mechanism across LLM providers on errors.
4. Deterministic non-tool context handling for Ollama.
5. /api/health endpoint structure and provider status reporting.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.base import LLMProvider, LLMResponse, LLMToolCall
from app.providers.gemini import GeminiLLMProvider
from app.providers.groq_openrouter import GroqProvider, OpenRouterProvider
from app.providers.ollama import OllamaProvider
from app.providers.registry import ProviderRegistry
from app.providers.sarvam import SarvamLLMProvider, SarvamSTTProvider, SarvamTTSProvider


def test_provider_order_parsing():
    s = Settings(
        google_api_key="fake-key",
        llm_provider_order="gemini, sarvam, groq, ollama",
    )
    assert s.llm_provider_order == ["gemini", "sarvam", "groq", "ollama"]


def test_provider_health_checks():
    s = Settings(
        google_api_key="fake-gemini",
        sarvam_api_key="",
        groq_api_key="fake-groq",
        openrouter_api_key="",
        ollama_base_url="http://localhost:11434",
    )
    registry = ProviderRegistry(s)
    health = asyncio.run(registry.check_health())

    assert "ok" in health
    assert "providers" in health
    assert "llm_order" in health
    providers_by_name = {p["provider"]: p for p in health["providers"] if p["type"] == "llm"}

    assert providers_by_name["gemini"]["available"] is True
    assert providers_by_name["sarvam"]["available"] is False
    assert providers_by_name["groq"]["available"] is True
    assert providers_by_name["openrouter"]["available"] is False


def test_registry_fallback_on_failure():
    """Verify that when primary provider raises an error, fallback provider is invoked."""
    s = Settings(
        google_api_key="fake-gemini",
        sarvam_api_key="fake-sarvam",
        llm_provider_order=["gemini", "sarvam"],
    )
    registry = ProviderRegistry(s)

    mock_gemini = AsyncMock(spec=LLMProvider)
    mock_gemini.generate.side_effect = RuntimeError("Gemini 429 Resource Exhausted")

    mock_sarvam = AsyncMock(spec=LLMProvider)
    mock_sarvam.generate.return_value = LLMResponse(
        content="Sarvam fallback response",
        provider="sarvam",
        model="sarvam-2b",
    )

    registry._llm_providers["gemini"] = mock_gemini
    registry._llm_providers["sarvam"] = mock_sarvam

    res = asyncio.run(registry.generate_with_fallback([{"role": "user", "content": "Help"}]))

    assert mock_gemini.generate.called
    assert mock_sarvam.generate.called
    assert res.content == "Sarvam fallback response"
    assert res.provider == "sarvam"


def test_ollama_does_not_receive_tools():
    """Ensure small Ollama model receives None for tools to prevent schema confusion."""
    s = Settings(
        google_api_key="",
        llm_provider_order=["ollama"],
    )
    registry = ProviderRegistry(s)

    mock_ollama = AsyncMock(spec=LLMProvider)
    mock_ollama.generate.return_value = LLMResponse(
        content="Ollama direct grounded response",
        provider="ollama",
        model="llama3.2",
    )
    registry._llm_providers["ollama"] = mock_ollama

    tools = [{"name": "search_knowledge"}]
    res = asyncio.run(registry.generate_with_fallback([{"role": "user", "content": "Help"}], tools=tools))

    assert mock_ollama.generate.called
    call_args = mock_ollama.generate.call_args[1]
    # Tools argument must be None for Ollama
    assert call_args["tools"] is None
    assert res.content == "Ollama direct grounded response"


def test_api_health_endpoint():
    s = Settings(
        google_api_key="fake-key",
        sarvam_api_key="fake-sarvam",
        llm_provider_order=["gemini", "sarvam"],
    )
    registry = ProviderRegistry(s)

    app = create_app(settings=s, agent=AsyncMock(), speech=AsyncMock())
    app.state.provider_registry = registry

    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "providers" in data
    assert "llm_order" in data
    assert "stt_active" in data
    assert "tts_active" in data
