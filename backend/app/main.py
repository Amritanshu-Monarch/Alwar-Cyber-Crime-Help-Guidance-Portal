import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.genai import errors as genai_errors
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.config import Settings, get_settings
from app.schemas import SESSION_ID_PATTERN, AgentResult, ChatIn, TtsIn, VoiceOut
from app.services.speech import ALLOWED_AUDIO_MIME

log = logging.getLogger("app")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINDI_INTENT_PATTERN = re.compile(
    r"\b(in\s+hindi|hindi\s+me|hindi\s+mein|hindi\s+mai|shikayat\s+hindi|patra\s+hindi|application\s+hindi|application\s+in\s+hindi|hindi\s+me\s+do|hindi\s+mein\s+do)\b|[ऀ-ॿ]",
    re.IGNORECASE,
)
ENGLISH_INTENT_PATTERN = re.compile(r"\b(in\s+english|english\s+me|english\s+mein)\b", re.IGNORECASE)


def create_app(settings: Settings | None = None, agent=None, speech=None) -> FastAPI:
    """Factory. Pass fakes for agent/speech in tests; real ones are built at startup otherwise."""
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            if not s.google_api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. Copy backend/.env.example to backend/.env and add your key "
                    "(https://aistudio.google.com/app/apikey)."
                )
            from app.agent.runner import AgentService
            from app.providers import ProviderRegistry
            from app.services.speech import Speech

            app.state.agent = AgentService(s)
            app.state.speech = Speech(s)
            app.state.provider_registry = ProviderRegistry(s)
            try:
                from app.services.rag import get_kb

                get_kb().ensure_fresh()
            except Exception as exc:
                cause = exc
                while cause is not None:
                    if isinstance(cause, genai_errors.ClientError) and cause.code == 429:
                        log.warning(
                            "knowledge base refresh skipped because Gemini embedding quota is exhausted; "
                            "continuing with lexical retrieval"
                        )
                        break
                    cause = cause.__cause__
                else:
                    log.warning(
                        "knowledge base refresh skipped (%s); continuing with lexical retrieval",
                        exc,
                    )
        else:
            app.state.agent, app.state.speech = agent, speech
        yield

    app = FastAPI(title="Alwar Cyber Agent API", lifespan=lifespan)
    limiter = Limiter(key_func=get_remote_address, default_limits=[s.rate_limit])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    async def run_agent(request: Request, session_id: str, text: str, lang: str) -> AgentResult:
        try:
            return await request.app.state.agent.ask(session_id, text, lang)
        except TimeoutError:
            raise HTTPException(504, "The assistant took too long. Please call 1930.")
        except genai_errors.APIError as e:
            log.error("gemini api error code=%s", e.code)
            raise HTTPException(429 if e.code == 429 else 502, "Assistant unavailable. Please call 1930.")
        except Exception:
            log.exception("agent failure")
            raise HTTPException(500, "Assistant error. Please call 1930.")

    @app.get("/api/health")
    async def health():
        if hasattr(app.state, "provider_registry"):
            return await app.state.provider_registry.check_health()
        return {"ok": True}

    # Serve the portal itself so page + API share one origin (no CORS, mic works over http://localhost)
    portal_page = Path(__file__).resolve().parents[2] / "index.html"
    if portal_page.is_file():

        @app.get("/", include_in_schema=False)
        async def portal():
            return FileResponse(portal_page)

    @app.post("/api/chat", response_model=AgentResult)
    async def chat(request: Request, body: ChatIn):
        lang = body.lang
        if HINDI_INTENT_PATTERN.search(body.message):
            lang = "hi"
        elif ENGLISH_INTENT_PATTERN.search(body.message):
            lang = "en"
        return await run_agent(request, body.session_id, body.message, lang)

    @app.post("/api/voice", response_model=VoiceOut)
    async def voice(
        request: Request,
        audio: UploadFile = File(...),
        session_id: str = Form(...),
        lang: str = Form("en"),
    ):
        if not re.fullmatch(SESSION_ID_PATTERN, session_id) or lang not in ("en", "hi"):
            raise HTTPException(422, "bad session_id or lang")
        mime = (audio.content_type or "").split(";")[0].strip().lower()
        if mime not in ALLOWED_AUDIO_MIME:
            raise HTTPException(415, f"unsupported audio type {mime!r}; send WAV")
        data = await audio.read(s.max_audio_bytes + 1)
        if not data or len(data) > s.max_audio_bytes:
            raise HTTPException(413, "audio empty or too large")
        try:
            transcript = await request.app.state.speech.transcribe(data, mime)
        except genai_errors.APIError as e:
            log.error("stt api error code=%s", e.code)
            raise HTTPException(429 if e.code == 429 else 502, "Speech recognition unavailable")
        if not transcript:
            raise HTTPException(422, "no speech detected")
        if DEVANAGARI.search(transcript):
            lang = "hi"  # user spoke Hindi: answer in Hinglish whatever the page language is
        result = await run_agent(request, session_id, transcript[: s.max_text_chars], lang)
        return VoiceOut(transcript=transcript, **result.model_dump())

    @app.post("/api/tts")
    async def tts(request: Request, body: TtsIn):
        try:
            wav = await request.app.state.speech.synthesize(body.text[: s.max_tts_chars])
        except Exception:
            log.exception("tts failure")
            raise HTTPException(502, "Speech synthesis unavailable")
        return Response(content=wav, media_type="audio/wav")

    return app


app = create_app()
