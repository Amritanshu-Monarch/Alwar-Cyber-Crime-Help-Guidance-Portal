"""Real diagnostic script for all LLM and voice providers:
Gemini, Sarvam AI, Groq, and local Ollama.

Per user requirements:
1. Does NOT print or log any API keys (prints set: True/False and len).
2. Makes ONE minimal real call per provider/service and prints:
   provider, model name, HTTP status, latency (ms), and exact error body (with secrets redacted).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
import httpx

# Resolve .env path
CANDIDATE_PATHS = [
    Path.cwd() / "backend" / ".env",
    Path.cwd() / ".env",
    BACKEND_DIR / ".env",
    BACKEND_DIR.parent / ".env",
]
RESOLVED_ENV = next((p for p in CANDIDATE_PATHS if p.is_file()), None)
if RESOLVED_ENV:
    load_dotenv(RESOLVED_ENV, override=False)


def redact_secrets(text: str) -> str:
    """Mask any potential API keys, auth tokens, or query param keys."""
    if not text:
        return ""
    # Redact query params key=...
    masked = re.sub(r"key=[a-zA-Z0-9_\-]{8,}", "key=[REDACTED]", text)
    # Redact Bearer / api-subscription-key tokens
    masked = re.sub(r"(bearer|api-subscription-key|apikey|token)[:\s=]+[a-zA-Z0-9_\-]{8,}", r"\1: [REDACTED]", masked, flags=re.IGNORECASE)
    # Redact any 30+ char hex / base64 sequences
    masked = re.sub(r"\b[a-zA-Z0-9_\-]{35,}\b", "[REDACTED_TOKEN]", masked)
    return masked


async def diagnose_gemini() -> dict:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    model = os.getenv("AGENT_MODEL", "gemini-3.5-flash-lite")
    if not key:
        return {"provider": "gemini", "model": model, "status": "KEY_MISSING", "latency_ms": 0, "error": "GOOGLE_API_KEY is not set"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": "Hello, respond with 1 word."}]}]}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            lat = round((time.perf_counter() - t0) * 1000, 1)
            body = resp.text
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                return {"provider": "gemini", "model": model, "status": 200, "latency_ms": lat, "result": text[:60]}
            else:
                return {"provider": "gemini", "model": model, "status": resp.status_code, "latency_ms": lat, "error": redact_secrets(body)}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        return {"provider": "gemini", "model": model, "status": "EXCEPTION", "latency_ms": lat, "error": redact_secrets(str(e))}


async def diagnose_sarvam_chat() -> dict:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    model = os.getenv("SARVAM_MODEL", "sarvam-105b-conversations")
    if not key:
        return {"provider": "sarvam_chat", "model": model, "status": "KEY_MISSING", "latency_ms": 0, "error": "SARVAM_API_KEY is not set"}

    url = "https://api.sarvam.ai/v1/chat/completions"
    headers = {
        "api-subscription-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, respond in one word."}],
        "max_tokens": 15,
        "temperature": 0.2,
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            lat = round((time.perf_counter() - t0) * 1000, 1)
            body = resp.text
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return {"provider": "sarvam_chat", "model": model, "status": 200, "latency_ms": lat, "result": content[:60]}
            else:
                return {"provider": "sarvam_chat", "model": model, "status": resp.status_code, "latency_ms": lat, "error": redact_secrets(body)}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        return {"provider": "sarvam_chat", "model": model, "status": "EXCEPTION", "latency_ms": lat, "error": redact_secrets(str(e))}


async def diagnose_sarvam_tts() -> dict:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    model = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")
    speaker = os.getenv("SARVAM_SPEAKER", "kavya_hi_conversational")
    if not key:
        return {"provider": "sarvam_tts", "model": model, "status": "KEY_MISSING", "latency_ms": 0, "error": "SARVAM_API_KEY is not set"}

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": ["नमस्ते, 1930 पर कॉल करें।"],
        "target_language_code": "hi-IN",
        "speaker": speaker,
        "model": model,
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            lat = round((time.perf_counter() - t0) * 1000, 1)
            body = resp.text
            if resp.status_code == 200:
                audios = resp.json().get("audios", [])
                audio_len = len(audios[0]) if audios else 0
                return {"provider": "sarvam_tts", "model": model, "speaker": speaker, "status": 200, "latency_ms": lat, "result": f"Audio received ({audio_len} b64 chars)"}
            else:
                return {"provider": "sarvam_tts", "model": model, "speaker": speaker, "status": resp.status_code, "latency_ms": lat, "error": redact_secrets(body)}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        return {"provider": "sarvam_tts", "model": model, "speaker": speaker, "status": "EXCEPTION", "latency_ms": lat, "error": redact_secrets(str(e))}


async def diagnose_groq() -> dict:
    key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not key:
        return {"provider": "groq", "model": model, "status": "KEY_NOT_SET", "latency_ms": 0, "error": "GROQ_API_KEY is empty in .env (optional provider)"}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            lat = round((time.perf_counter() - t0) * 1000, 1)
            body = resp.text
            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return {"provider": "groq", "model": model, "status": 200, "latency_ms": lat, "result": content[:60]}
            else:
                return {"provider": "groq", "model": model, "status": resp.status_code, "latency_ms": lat, "error": redact_secrets(body)}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        return {"provider": "groq", "model": model, "status": "EXCEPTION", "latency_ms": lat, "error": redact_secrets(str(e))}


async def diagnose_ollama() -> dict:
    base_url = (os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            lat = round((time.perf_counter() - t0) * 1000, 1)
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "").strip()
                return {"provider": "ollama", "model": model, "status": 200, "latency_ms": lat, "result": content[:60]}
            else:
                return {"provider": "ollama", "model": model, "status": resp.status_code, "latency_ms": lat, "error": redact_secrets(resp.text)}
    except Exception as e:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        return {"provider": "ollama", "model": model, "status": "UNREACHABLE", "latency_ms": lat, "error": redact_secrets(str(e))}


async def main():
    print("=" * 80)
    print(" ALWAR CYBER CRIME GUIDANCE PORTAL — REAL PROVIDER DIAGNOSTICS")
    print("=" * 80)
    print(f"Resolved .env Path: {RESOLVED_ENV} (exists: {RESOLVED_ENV.exists() if RESOLVED_ENV else False})")
    print("-" * 80)
    print("API Key Status (Values Strictly Hidden):")
    for key_name in ["GOOGLE_API_KEY", "SARVAM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]:
        v = os.getenv(key_name, "")
        print(f"  • {key_name:<20}: set={str(bool(v)):<5} | length={len(v)}")
    print("-" * 80)
    print("Executing minimal real calls to each provider...\n")

    results = await asyncio.gather(
        diagnose_gemini(),
        diagnose_sarvam_chat(),
        diagnose_sarvam_tts(),
        diagnose_groq(),
        diagnose_ollama(),
    )

    for r in results:
        prov = r.get("provider", "unknown")
        mod = r.get("model", "n/a")
        st = r.get("status")
        lat = r.get("latency_ms", 0)
        print(f"[{prov.upper()}] Model: {mod}")
        print(f"  Status   : {st}")
        print(f"  Latency  : {lat} ms")
        if "result" in r:
            print(f"  Success  : {r['result']}")
        if "error" in r:
            print(f"  Error    : {r['error']}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
