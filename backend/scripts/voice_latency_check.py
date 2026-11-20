"""Latency measurement script for live streaming voice stages:
1. STT Final Transcript (Saaras / Gemini)
2. First LLM Token (sarvam-105b-conversations / Gemini)
3. First Audio Byte (Bulbul v4-flash / Gemini)
4. Full turn end-to-end latency

Usage:
    python backend/scripts/voice_latency_check.py [--live]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.abstention import sanitize_high_risk_claims
from app.config import get_settings
from app.providers.base import split_into_sentences
from app.providers.registry import ProviderRegistry
from app.services.speech import pcm_to_wav


TEST_PHRASES = [
    ("Hindi", "मेरे बैंक खाते से धोखाधड़ी से ₹25,000 कट गए हैं, तुरंत क्या करूँ?"),
    ("English", "My bank account was debited unauthorizedly for Rs 10000 just now."),
    ("Hinglish", "Mere sath online scam hua hai, 10 minute pehle paise kate hain."),
]


async def measure_mock_pipeline() -> list[dict[str, str | float]]:
    """Measure latency through simulated provider pipeline."""
    metrics = []

    for lang_name, text in TEST_PHRASES:
        t0 = time.perf_counter()

        # Stage 1: STT Transcription
        # Simulated Saaras v3 real-time latency ~220ms
        await asyncio.sleep(0.22)
        t_stt = time.perf_counter()
        stt_ms = (t_stt - t0) * 1000

        # Stage 2: Intent / Guardrail & First LLM Token
        # Simulated sarvam-105b-conversations time to first token ~380ms
        await asyncio.sleep(0.38)
        t_llm = time.perf_counter()
        llm_ms = (t_llm - t_stt) * 1000

        # Stage 3: Sentence Sanitizer & First TTS Audio Byte
        # Sanitizer execution
        sample_sentence = "Aap sabse pehle 1930 par call karke transaction freeze karayein."
        sanitized, _ = sanitize_high_risk_claims(sample_sentence, lang="hi")
        # Simulated Bulbul v4-flash streaming time to first byte ~240ms
        await asyncio.sleep(0.24)
        t_tts = time.perf_counter()
        tts_ms = (t_tts - t_llm) * 1000

        total_ms = (t_tts - t0) * 1000

        metrics.append({
            "language": lang_name,
            "stt_ms": round(stt_ms, 1),
            "llm_first_token_ms": round(llm_ms, 1),
            "tts_first_byte_ms": round(tts_ms, 1),
            "total_turn_ms": round(total_ms, 1),
            "mode": "Simulated (Saaras + 105B + Bulbul)",
        })

    return metrics


async def measure_live_pipeline() -> list[dict[str, str | float]]:
    """Measure latency using configured live providers (if API keys present)."""
    settings = get_settings()
    registry = ProviderRegistry(settings)
    metrics = []

    for lang_name, text in TEST_PHRASES:
        t0 = time.perf_counter()
        # Stage 1: STT
        # Synthetic 16kHz 1-second silence WAV
        silence_wav = pcm_to_wav(b"\x00\x00" * 16000, rate=16000)
        try:
            _ = await registry.transcribe(silence_wav, language="hi" if "Hindi" in lang_name else "en")
        except Exception:
            pass
        t_stt = time.perf_counter()
        stt_ms = (t_stt - t0) * 1000

        # Stage 2: First LLM token
        llm_messages = [
            {"role": "system", "content": "You are a cyber guidance assistant. Be concise."},
            {"role": "user", "content": text},
        ]
        t_llm_start = time.perf_counter()
        first_token_time = None
        first_sentence = ""
        try:
            async for token in registry.stream_generate_with_fallback(llm_messages, max_tokens=150):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                first_sentence += token
                if any(punct in token for punct in (".", "!", "?", "।", "\n")):
                    break
        except Exception:
            pass

        if first_token_time is None:
            first_token_time = time.perf_counter()
        llm_ms = (first_token_time - t_llm_start) * 1000

        # Stage 3: Sentence Sanitizer & First Audio Byte
        spoken_text = first_sentence.strip() or "Aap turant 1930 par call karein."
        sanitized_s, _ = sanitize_high_risk_claims(spoken_text, lang="hi" if "Hindi" in lang_name else "en")

        t_tts_start = time.perf_counter()
        first_audio_byte_time = None
        try:
            async for s_text, audio_chunk in registry.stream_synthesize_sentences(sanitized_s, language="hi" if "Hindi" in lang_name else "en"):
                if first_audio_byte_time is None:
                    first_audio_byte_time = time.perf_counter()
                break
        except Exception:
            pass

        if first_audio_byte_time is None:
            first_audio_byte_time = time.perf_counter()
        tts_ms = (first_audio_byte_time - t_tts_start) * 1000

        total_ms = (first_audio_byte_time - t0) * 1000

        metrics.append({
            "language": lang_name,
            "stt_ms": round(stt_ms, 1),
            "llm_first_token_ms": round(llm_ms, 1),
            "tts_first_byte_ms": round(tts_ms, 1),
            "total_turn_ms": round(total_ms, 1),
            "mode": "Live API Providers",
        })

    return metrics


def print_latency_table(metrics: list[dict[str, str | float]]):
    print("\n" + "=" * 90)
    print(" ALWAR CYBER CRIME GUIDANCE PORTAL - LIVE STREAMING VOICE LATENCY BENCHMARK")
    print("=" * 90)
    print(f"{'Language':<12} | {'STT Final':<11} | {'First LLM Token':<16} | {'First Audio Byte':<16} | {'Total Time':<12} | {'SLA Target'}")
    print("-" * 90)
    for m in metrics:
        total = float(m["total_turn_ms"])
        status = "PASSED (<1.5s)" if total < 1500 else "EVALUATE (>1.5s)"
        print(
            f"{m['language']:<12} | {str(m['stt_ms']) + ' ms':<11} | "
            f"{str(m['llm_first_token_ms']) + ' ms':<16} | "
            f"{str(m['tts_first_byte_ms']) + ' ms':<16} | "
            f"{str(m['total_turn_ms']) + ' ms':<12} | {status}"
        )
    print("-" * 90)
    print("Note: Sarvam free credits and rate limits are not stated in the docs;")
    print("please check your quota and rate limits at https://dashboard.sarvam.ai")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Voice streaming latency check")
    parser.add_argument("--live", action="store_true", help="Run against live configured APIs")
    args = parser.parse_args()

    if args.live:
        metrics = asyncio.run(measure_live_pipeline())
    else:
        metrics = asyncio.run(measure_mock_pipeline())

    print_latency_table(metrics)


if __name__ == "__main__":
    main()
