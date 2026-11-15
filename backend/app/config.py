from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# google-adk / google-genai read GOOGLE_API_KEY straight from os.environ
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    google_api_key: str = ""
    agent_model: str = "gemini-3.1-flash-lite"
    # used automatically when the primary model is overloaded (503) or rate limited (429)
    agent_fallback_model: str = "gemini-3.8-flash"
    stt_model: str = "gemini-3.1-flash-lite"
    tts_model: str = "gemini-3.8-flash-tts"
    tts_voice: str = "Kore"
    embed_model: str = "gemini-embedding-001"
    embed_dims: int = 768
    # LLM rerank costs one extra Gemini call per search; the free tier allows ~15/min per model. Opt in with RERANK=true
    rerank: bool = False
    rerank_model: str = "gemini-3.1-flash-lite"

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
    ]
    rate_limit: str = "30/minute"
    agent_timeout_s: float = 45.0  # fallback model (last attempt)
    agent_primary_timeout_s: float = 25.0  # give up on a slow/overloaded primary sooner and fall back

    max_audio_bytes: int = 5 * 1024 * 1024
    max_text_chars: int = 1000
    max_tts_chars: int = 600

    kb_dir: Path = BASE_DIR / "kb"
    chroma_dir: Path = BASE_DIR / ".chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
