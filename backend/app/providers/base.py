"""Base interfaces and data structures for multi-provider abstraction:
LLM, Speech-to-Text (STT), and Text-to-Speech (TTS).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    provider: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    provider: str
    service_type: str  # "llm", "stt", "tts"
    available: bool
    details: str = ""


class LLMProvider(ABC):
    """Abstract interface for LLM text and reasoning providers."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        """Generate response given standard chat messages and optional tools."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if provider credentials/endpoint are reachable and active."""
        pass


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text transcription."""

    name: str = "base"

    @abstractmethod
    async def transcribe(self, audio: bytes, mime: str = "audio/wav", language: str = "en") -> str:
        """Transcribe speech audio bytes into clean verbatim text."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if transcription endpoint is available."""
        pass


class TTSProvider(ABC):
    """Abstract interface for Text-to-Speech audio synthesis."""

    name: str = "base"

    @abstractmethod
    async def synthesize(self, text: str, language: str = "en", voice: str = "Kore") -> bytes:
        """Synthesize text into WAV audio bytes."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if synthesis endpoint is available."""
        pass
