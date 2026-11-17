from functools import lru_cache
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# google-adk / google-genai read GOOGLE_API_KEY straight from os.environ
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    google_api_key: str = ""

    # Provider configuration
    # Comma-separated or list of providers to try in order on 429/timeout/5xx
    llm_provider_order: list[str] = ["gemini", "sarvam", "groq", "ollama"]
    stt_provider: str = "gemini"  # "gemini" or "sarvam"
    tts_provider: str = "gemini"  # "gemini" or "sarvam"

    # Sarvam AI (Indian multilingual platform with free tier credits)
    sarvam_api_key: str = ""
    sarvam_model: str = "sarvam-2b"

    # Groq & OpenRouter (OpenAI-compatible fast inference)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # Local Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    @field_validator("llm_provider_order", mode="before")
    @classmethod
    def parse_provider_order(cls, v: Union[str, list[str]]) -> list[str]:
        if isinstance(v, str):
            return [p.strip().lower() for p in v.split(",") if p.strip()]
        return [str(p).strip().lower() for p in v]

    # Changed 2026-11-16: was "gemini-3.5-flash-lite" — valid model but too weak for reliable
    # multi-tool ADK routing (misroutes Hindi questions, drops tools under load).
    # Now "gemini-3.8-flash" — current stable workhorse; verified at aistudio.google.com/docs/models.
    agent_model: str = "gemini-3.8-flash"

    # Verified 2026-11-16: gemini-3.5-flash-lite is a valid stable model and appropriate as fallback.
    # Used automatically when the primary model is overloaded (503) or rate limited (429).
    agent_fallback_model: str = "gemini-3.5-flash-lite"

    # Changed 2026-11-16: was "gemini-3.5-flash-lite" — generic model, not optimised for transcription.
    # Now "gemini-3.5-transcribe" — dedicated stable transcription model with superior accuracy
    # for Hinglish and multilingual audio; verified at aistudio.google.com/docs/models.
    stt_model: str = "gemini-3.5-transcribe"

    # Verified 2026-11-16: gemini-3.8-flash-tts is a valid stable TTS model. No change required.
    # The previous runtime failure was caused by a broken SDK call (client.interactions.create),
    # not by an incorrect model name. Fixed in speech.py.
    tts_model: str = "gemini-3.8-flash-tts"

    tts_voice: str = "Kore"
    embed_model: str = "gemini-embedding-001"
    embed_dims: int = 768

    # LLM rerank costs one extra Gemini call per search; the free tier allows ~15/min per model.
    # Opt in with RERANK=true in .env.
    rerank: bool = False

    # Verified 2026-11-16: gemini-3.5-flash-lite is valid for opt-in reranking. No change required.
    rerank_model: str = "gemini-3.5-flash-lite"

    # "null" = index.html opened straight from disk (file://); the others are common local dev servers
    allowed_origins: list[str] = [
        "null",
        "https://amritanshu-monarch.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ]
    rate_limit: str = "30/minute"
    agent_timeout_s: float = 25.0  # fallback model (last attempt)
    agent_primary_timeout_s: float = 15.0  # give up on a slow/overloaded primary sooner and fall back

    max_audio_bytes: int = 5 * 1024 * 1024
    max_text_chars: int = 1000
    max_tts_chars: int = 600

    kb_dir: Path = BASE_DIR / "kb"
    chroma_dir: Path = BASE_DIR / ".chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
