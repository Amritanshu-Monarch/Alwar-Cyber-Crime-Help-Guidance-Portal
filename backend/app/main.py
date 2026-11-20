import asyncio
import base64
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.genai import errors as genai_errors
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.agent.abstention import sanitize_high_risk_claims
from app.agent.runner import clean_reply_text
from app.config import Settings, get_settings
from app.providers.base import split_into_sentences
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
        from app.config import RESOLVED_ENV_PATH
        print(f"[startup] Loaded .env from: {RESOLVED_ENV_PATH}")
        log.info("[startup] Loaded .env from: %s", RESOLVED_ENV_PATH)
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

    async def run_agent(
        request: Request, session_id: str, text: str, lang: str, is_voice: bool = False
    ) -> AgentResult:
        try:
            import inspect

            sig = inspect.signature(request.app.state.agent.ask)
            if "is_voice" in sig.parameters:
                res = await request.app.state.agent.ask(session_id, text, lang, is_voice=is_voice)
            else:
                res = await request.app.state.agent.ask(session_id, text, lang)
            if hasattr(res, "reply") and res.reply:
                res.reply = clean_reply_text(res.reply)
            # Strip internal metadata, sources, filenames, citations, and provider tags from API output
            res.sources = []
            res.citation_note = ""
            res.tools_used = []
            res.provider = ""
            log.info("Turn completed for session %s", session_id)
            return res
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
        result = await run_agent(request, session_id, transcript[: s.max_text_chars], lang, is_voice=True)
        return VoiceOut(transcript=transcript, **result.model_dump())

    @app.post("/api/tts")
    async def tts(request: Request, body: TtsIn):
        try:
            txt = clean_reply_text(body.text[: s.max_tts_chars])
            lang_code = body.lang if getattr(body, "lang", None) in ("hi", "en") else ("hi" if DEVANAGARI.search(txt) else "hi")
            registry = getattr(request.app.state, "provider_registry", None)
            if registry is not None:
                wav = await registry.synthesize(txt, language=lang_code)
            else:
                wav = await request.app.state.speech.synthesize(txt)
        except Exception:
            log.exception("tts failure")
            raise HTTPException(502, "Speech synthesis unavailable")
        media_type = "audio/wav" if wav[:4] == b"RIFF" else "audio/mpeg"
        return Response(content=wav, media_type=media_type)

    @app.websocket("/ws/voice")
    async def websocket_voice(websocket: WebSocket):
        """Live streaming voice WebSocket endpoint.
        Handles:
        1. Bi-directional audio chunks streaming to Sarvam Saaras real-time STT.
        2. Live partial & final transcript delivery.
        3. Real-time agent pipeline execution with voice-first style.
        4. Sentence-by-sentence streaming TTS synthesis (Bulbul v4-flash).
        5. Strict high-risk sanitizer verification on each sentence before TTS.
        6. Immediate barge-in cancellation when user speaks or interrupts.
        """
        await websocket.accept()
        session_id = websocket.query_params.get("session_id") or "voice-stream"
        initial_lang = websocket.query_params.get("lang") or "hi"
        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            session_id = "voice-stream"

        registry = getattr(app.state, "provider_registry", None)
        if registry is None:
            from app.providers.registry import ProviderRegistry

            registry = ProviderRegistry(s)

        agent_svc = getattr(app.state, "agent", None)
        if agent_svc is None:
            from app.agent.runner import AgentService

            agent_svc = AgentService(s)

        from app.providers.sarvam import SarvamRealtimeSTT

        stt_client = SarvamRealtimeSTT(s, language_code="hi-IN")
        stt_connected = await stt_client.connect()

        current_task: asyncio.Task | None = None
        is_speaking = False
        audio_buffer = bytearray()

        async def send_json_safe(msg: dict):
            try:
                await websocket.send_text(json.dumps(msg))
            except Exception:
                pass

        async def run_turn(text: str, detected_lang: str):
            nonlocal is_speaking
            try:
                is_speaking = False
                await send_json_safe({"type": "state", "state": "thinking"})

                # Run full agent pipeline with voice-first directive
                result = await agent_svc.ask(session_id, text, lang=detected_lang, is_voice=True)

                clean_reply = clean_reply_text(result.reply)
                await send_json_safe({
                    "type": "reply_text",
                    "text": clean_reply,
                    "language": detected_lang,
                })

                is_speaking = True
                await send_json_safe({"type": "state", "state": "speaking"})

                # Synthesize sentence by sentence for low latency playback
                sentences = split_into_sentences(clean_reply)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    # CRITICAL SECURITY GUARD: Sanitize each sentence BEFORE sending to TTS
                    sanitized_s, _ = sanitize_high_risk_claims(sentence, lang=detected_lang)
                    try:
                        audio_bytes = await registry.synthesize(sanitized_s, language=detected_lang)
                        if audio_bytes:
                            b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                            await send_json_safe({
                                "type": "audio_chunk",
                                "sentence": sanitized_s,
                                "audio": b64_audio,
                            })
                    except asyncio.CancelledError:
                        raise
                    except Exception as tts_err:
                        log.warning("Sentence TTS failed for '%s': %s", sanitized_s[:30], tts_err)

                is_speaking = False
                await send_json_safe({"type": "state", "state": "listening"})
            except asyncio.CancelledError:
                is_speaking = False
                log.info("Agent turn cancelled by user interruption (barge-in)")
            except Exception as e:
                is_speaking = False
                log.exception("Error in voice turn: %s", e)
                await send_json_safe({"type": "error", "message": "Assistant error. Please call 1930."})
                await send_json_safe({"type": "state", "state": "listening"})

        async def sarvam_listener():
            try:
                async for event in stt_client.receive_events():
                    ev_type = event.get("event")
                    if ev_type == "transcript.partial":
                        part_text = event.get("text", "").strip()
                        if part_text:
                            await send_json_safe({
                                "type": "partial_transcript",
                                "text": part_text,
                                "language": event.get("language", initial_lang),
                            })
                    elif ev_type == "transcript.final":
                        final_text = event.get("text", "").strip()
                        if final_text:
                            ev_lang = "hi" if DEVANAGARI.search(final_text) else event.get("language", initial_lang)
                            await send_json_safe({
                                "type": "final_transcript",
                                "text": final_text,
                                "language": ev_lang,
                            })
                            nonlocal current_task
                            if current_task and not current_task.done():
                                current_task.cancel()
                            current_task = asyncio.create_task(run_turn(final_text, ev_lang))
            except Exception as e:
                log.debug("Sarvam STT listener loop ended: %s", e)

        listener_task = asyncio.create_task(sarvam_listener()) if stt_connected else None

        await send_json_safe({
            "type": "ready",
            "session_id": session_id,
            "sarvam_realtime": stt_connected,
        })

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "text" in message and message["text"]:
                    try:
                        msg = json.loads(message["text"])
                    except Exception:
                        msg = {"type": "text", "text": message["text"]}

                    m_type = msg.get("type")
                    if m_type == "audio":
                        # Audio chunk in base64
                        raw_data = base64.b64decode(msg.get("data", ""))
                        if is_speaking:
                            # User spoke while agent was speaking -> Barge-in!
                            if current_task and not current_task.done():
                                current_task.cancel()
                            is_speaking = False
                            await send_json_safe({"type": "interrupted"})
                            await send_json_safe({"type": "state", "state": "listening"})

                        if stt_connected:
                            await stt_client.send_audio(raw_data)
                        else:
                            audio_buffer.extend(raw_data)

                    elif m_type == "interrupt":
                        # Explicit user interrupt button
                        if current_task and not current_task.done():
                            current_task.cancel()
                        is_speaking = False
                        await send_json_safe({"type": "interrupted"})
                        await send_json_safe({"type": "state", "state": "listening"})

                    elif m_type == "end_turn":
                        if not stt_connected and audio_buffer:
                            # Offline / fallback STT batch transcription
                            try:
                                transcript = await registry.transcribe(
                                    bytes(audio_buffer), mime="audio/wav", language=initial_lang
                                )
                            except Exception:
                                transcript = ""
                            audio_buffer.clear()
                            if transcript:
                                det_lang = "hi" if DEVANAGARI.search(transcript) else initial_lang
                                await send_json_safe({
                                    "type": "final_transcript",
                                    "text": transcript,
                                    "language": det_lang,
                                })
                                if current_task and not current_task.done():
                                    current_task.cancel()
                                current_task = asyncio.create_task(run_turn(transcript, det_lang))

                    elif m_type == "text":
                        text_val = msg.get("text", "").strip()
                        if text_val:
                            det_lang = "hi" if DEVANAGARI.search(text_val) else initial_lang
                            await send_json_safe({
                                "type": "final_transcript",
                                "text": text_val,
                                "language": det_lang,
                            })
                            if current_task and not current_task.done():
                                current_task.cancel()
                            current_task = asyncio.create_task(run_turn(text_val, det_lang))

                elif "bytes" in message and message["bytes"]:
                    raw_bytes = message["bytes"]
                    if is_speaking:
                        # User spoke while agent was speaking -> Barge-in!
                        if current_task and not current_task.done():
                            current_task.cancel()
                        is_speaking = False
                        await send_json_safe({"type": "interrupted"})
                        await send_json_safe({"type": "state", "state": "listening"})

                    if stt_connected:
                        await stt_client.send_audio(raw_bytes)
                    else:
                        audio_buffer.extend(raw_bytes)

        except WebSocketDisconnect:
            log.info("WebSocket /ws/voice disconnected for session %s", session_id)
        finally:
            if listener_task and not listener_task.done():
                listener_task.cancel()
            if current_task and not current_task.done():
                current_task.cancel()
            await stt_client.close()

    return app


app = create_app()
