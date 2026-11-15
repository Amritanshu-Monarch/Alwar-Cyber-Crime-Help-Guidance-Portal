import asyncio
import base64
import io
import logging
import re
import wave

from google import genai
from google.genai import types

from app.config import Settings

log = logging.getLogger(__name__)

TTS_SAMPLE_RATE = 24000  # Default sample rate for mono PCM wrapped by pcm_to_wav
ALLOWED_AUDIO_MIME = {"audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/flac", "audio/aac"}

STT_PROMPT = (
    "Transcribe this audio verbatim. The speaker may use Hindi, English, or a Hindi-English mix (Hinglish). "
    "Write Hindi words in Devanagari script and English words in Latin letters. "
    "Output only the transcript. If there is no speech, output nothing."
)

# Common hallucinated strings when models encounter silence or low background noise
HALLUCINATIONS = {
    "आज तो मौसम बड़ा सुहाना है।",
    "आज तो मौसम बड़ा सुहाना है",
    "thank you for watching",
    "subtitles by",
    "please subscribe",
}


def pcm_to_wav(pcm: bytes, rate: int = TTS_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def clean_transcript(text: str) -> str:
    t = text.strip()
    # Strip enclosing quotes if any
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    # Remove bracketed audio cues like [music], (silence), [applause]
    t = re.sub(r"\[.*?\]|\(.*?\)", "", t).strip()
    # Check against known noise hallucinations
    if t.lower() in HALLUCINATIONS or t in HALLUCINATIONS:
        return ""
    # If it contains no letters or digits (e.g. only punctuation like "."), treat as empty
    if not re.search(r"[\w\u0900-\u097F]", t):
        return ""
    return t


def clean_for_speech(text: str) -> str:
    """Strip markdown formatting, links, and format numbers so speech output sounds natural."""
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    s = re.sub(r"\*([^*]+)\*", r"\1", s)        # italic
    s = re.sub(r"#{1,6}\s*", "", s)             # headings
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)  # link markdown -> title
    s = re.sub(r"[`_~]", "", s)                 # backticks, underscores
    s = re.sub(r"^[•\-\*]\s+", "", s, flags=re.MULTILINE)  # bullet points
    # Separate long numbers (like 1930 or 8764874306) with spaces for clear digit-by-digit pronunciation
    s = re.sub(r"\b\d{3,}\b", lambda m: " ".join(m.group(0)), s)
    return s.strip()


class Speech:
    def __init__(self, settings: Settings, client: genai.Client | None = None):
        self._s = settings
        self._client = client or genai.Client(api_key=settings.google_api_key or None)

    async def transcribe(self, audio: bytes, mime: str) -> str:
        def _call():
            return self._client.models.generate_content(
                model=self._s.stt_model,
                contents=[STT_PROMPT, types.Part.from_bytes(data=audio, mime_type=mime)],
                config=types.GenerateContentConfig(temperature=0.0),
            )

        for attempt in range(2):
            try:
                r = await asyncio.to_thread(_call)
                raw = (r.text or "").strip()
                return clean_transcript(raw)
            except Exception as e:
                log.warning("stt attempt %d failed: %s", attempt + 1, e)
                if attempt == 1:
                    raise
                await asyncio.sleep(0.3)
        return ""

    async def synthesize(self, text: str) -> bytes:
        clean_text = clean_for_speech(text) or text
        def _call():
            return self._client.interactions.create(
                model=self._s.tts_model,
                input=[{"type": "user_input", "content": [{"type": "text", "text": clean_text}]}],
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": self._s.tts_voice}]},
            )

        for attempt in range(2):
            try:
                r = await asyncio.to_thread(_call)
                return base64.b64decode(r.output_audio.data)
            except Exception as e:
                log.warning("tts attempt %d failed: %s", attempt + 1, e)
                if attempt == 1:
                    raise
                await asyncio.sleep(0.4)
        raise RuntimeError("speech synthesis failed after retries")
