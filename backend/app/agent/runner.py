import asyncio
import logging
import re

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types

from app.agent.abstention import (
    CYBER_DOMAINS,
    get_scope_abstention_reply,
    is_out_of_scope_query,
    sanitize_high_risk_claims,
)
from app.agent.state import (
    SessionCaseState,
    extract_and_update_state,
    needs_triage_question,
)
from app.services.intent_classifier import classify_intent
from app.agent.grounding import (
    attribute_evidence,
    format_evidence_context,
    format_grounding_directive,
)
from app.agent.prompts import INSTRUCTION
from app.agent.tools import ALL_TOOLS
from app.config import Settings
from app.providers.base import split_into_sentences
from app.providers.registry import ProviderRegistry
from app.schemas import AgentResult, Artifact
from app.services.query_expander import classify_domains
from app.services.rag import get_kb

log = logging.getLogger(__name__)

APP_NAME = "alwar_cyber_agent"
USER_ID = "citizen"  # anonymous; sessions are keyed by the client-generated session_id
REPLY_LANGUAGE = {
    "en": "[Reply in English.]",
    "hi": "[Reply in Hindi in Devanagari script. Use simple, natural spoken Hindi suitable for voice clarity. Write key numbers as digits (1930, 8764874306). Keep tool language argument as hi.]",
    "hinglish": "[Reply in Hindi written in Roman (English) letters, i.e. natural spoken Hinglish like 'Aap sabse pehle 1930 par call karein'. Keep English technical terms in Latin letters (UPI, OTP, bank statement, cybercrime.gov.in). Never use Devanagari. Keep tool language argument as hi.]",
}
VOICE_DIRECTIVE = (
    "[VOICE CALL DIRECTIVE: Citizen is speaking over a live voice phone/mic call.\n"
    "1. Give the SINGLE most urgent immediate action first (e.g. call 1930 immediately or block SIM/card).\n"
    "2. If critical case details are missing, ask at most ONE short follow-up question (e.g. 'Paise kitne der pehle kate? UPI se ya card se?').\n"
    "3. Keep the spoken response short and direct (1-2 crisp sentences). Do not read long numbered lists.\n"
    "4. At the end, ask: 'Kya aapko poora step-by-step procedure chahiye?'.\n"
    "5. Never repeat any steps the citizen already confirmed they completed.\n"
    "6. Do not use markdown asterisks, bullet points, or raw website URLs.\n"
    "7. Never mention file names, YAML or Markdown sources, model names (Gemini, Sarvam), internal metadata, or knowledge-base references.]"
)
DEVANAGARI_CHARS = re.compile(r"[\u0900-\u097F]")
HINGLISH_WORDS = frozenset({
    "aap", "aapka", "aapke", "aapki", "apna", "apne", "apni",
    "mera", "mere", "meri", "mujhe", "mujhko", "hum", "humara", "hamara",
    "karein", "karna", "karo", "kare", "karu", "karte", "karta", "karti", "karne", "karenge", "karega", "karegi",
    "kaise", "kya", "kyun", "kyu", "kahan", "kiske", "kisko", "kisi", "unhone", "usne", "unka", "unke", "unki", "uska", "uske", "uski",
    "nahi", "nahin", "mat",
    "gaya", "gaye", "gayi", "hoga", "hogi", "honge", "hua", "hue", "hui",
    "thana", "shikayat", "batao", "bataiye", "chahiye",
    "paise", "rupaye", "khate", "khaton",
    "lekin", "magar", "agar",
    "raha", "rahe", "rahi",
    "bheja", "dikha", "dhamki",
    "hai", "hain", "hoon", "hun",
    "saath", "bachav",
})
COMMON_ENGLISH_WORDS = frozenset({
    "how", "what", "where", "when", "why", "who", "which",
    "can", "could", "should", "would", "is", "are", "am", "was", "were",
    "the", "this", "that", "these", "those",
    "to", "from", "in", "on", "at", "for", "with", "about", "by", "of",
    "an", "a", "my", "your", "his", "her", "their", "our",
    "i", "you", "he", "she", "it", "they", "we",
    "do", "does", "did", "have", "has", "had", "not",
    "please", "help", "report", "complaint", "police", "station",
    "bank", "account", "money", "fraud", "scam", "transfer", "unauthorized",
    "someone", "call", "message", "documents", "ready", "keep", "going",
})
ENGLISH_INTENT = re.compile(r"\b(in\s+english|english\s+me|english\s+mein)\b", re.IGNORECASE)
HINDI_INTENT = re.compile(r"\b(in\s+hindi|hindi\s+me|hindi\s+mein|hindi\s+mai|shikayat\s+hindi|patra\s+hindi)\b", re.IGNORECASE)
FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:pura|poora|complete|full|detail|details|procedure|process|steps?|"
    r"batao|bataiye|samjhao|explain|more|next|again|do|karo|karu)\b|"
    r"\b(?:step\s+by\s+step|what\s+next|how\s+to\s+continue)\b",
    re.IGNORECASE,
)


def detect_effective_language(text: str, fallback_lang: str = "en") -> str:
    """Detect conversational language: 'hi' (Devanagari), 'hinglish' (Roman Hindi), or 'en' (English)."""
    if not text:
        return fallback_lang if fallback_lang in ("en", "hi", "hinglish") else "en"
    if DEVANAGARI_CHARS.search(text):
        return "hi"
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if words & HINGLISH_WORDS:
        return "hinglish"
    if words & COMMON_ENGLISH_WORDS:
        return "en"
    if ENGLISH_INTENT.search(text):
        return "en"
    if HINDI_INTENT.search(text):
        return "hinglish"
    return fallback_lang if fallback_lang in ("en", "hi", "hinglish") else "en"

FALLBACK_STATUS_CODES = {404, 429, 500, 502, 503, 504}  # model gone / rate limited / overloaded
FALLBACK_REPLY = "Sorry, I could not answer that. Please call 1930 or 8764874306."


def build_agent(model: str) -> Agent:
    return Agent(
        name="alwar_cyber_assistant",
        model=model,
        description="Cyber crime help and guidance assistant for Alwar citizens.",
        instruction=INSTRUCTION,
        tools=ALL_TOOLS,
        generate_content_config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1000),
    )


def trim_to_sentence_boundary(text: str) -> str:
    """Ensure replies never get cut off mid-sentence."""
    if not text:
        return ""
    stripped = text.strip()
    if stripped.endswith((".", "!", "?", "।", ":", '"', "'")):
        return stripped
    # Search for the last sentence terminator followed by whitespace or string end
    matches = list(re.finditer(r"[.!?।](?:\s+|$)", stripped))
    if matches:
        last_end = matches[-1].end()
        return stripped[:last_end].strip()
    return stripped


def clean_reply_text(text: str) -> str:
    """Strip leading or echoed bracketed system directives, internal source/knowledge-base lines,
    YAML/MD file references, model identifiers, and trim mid-sentence cuts."""
    if not text:
        return ""
    # Strip any bracketed directives: [Reply in ...], [VOICE CALL DIRECTIVE: ...], [TRIAGE DIRECTIVE: ...], [OFFICIAL KNOWLEDGE ...]
    cleaned = re.sub(r"\[(?:Reply in|VOICE CALL DIRECTIVE|TRIAGE DIRECTIVE|OFFICIAL KNOWLEDGE|CRITICAL)[^\]]*\]", "", text, flags=re.IGNORECASE)
    # Strip lines that reference sources, citations or knowledge-base files
    cleaned = re.sub(r"(?im)^(?:#{1,6}\s*)?(?:Sources?|Verified Sources?|Knowledge[- ]?base(?: references)?|स्रोत|आधिकारिक ज्ञानकोष से सत्यापित)[^\n]*$", "", cleaned)
    # Strip lines mentioning yaml or md files
    cleaned = re.sub(r"(?im)^[^\n]*\b[\w-]+\.(?:yaml|yml|md)\b[^\n]*$", "", cleaned)
    # Strip inline yaml/md mentions
    cleaned = re.sub(r"`?[\w-]+\.(?:yaml|yml|md)`?", "", cleaned, flags=re.IGNORECASE)
    # Strip model self-identifications if echoed
    cleaned = re.sub(r"\b(?:I am|As|powered by|using)\s+(?:Gemini|Sarvam|Ollama|Llama|GPT)[\w\s.-]*", "", cleaned, flags=re.IGNORECASE)
    # Clean up multi-newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = trim_to_sentence_boundary(cleaned)
    return cleaned.strip()


class AgentService:
    # ponytail: in-memory sessions grow unbounded and die on restart.
    # Swap InMemorySessionService for ADK DatabaseSessionService if you need persistence or eviction.
    def __init__(self, settings: Settings):
        self._s = settings
        self._timeout = settings.agent_timeout_s
        self._primary_timeout = settings.agent_primary_timeout_s
        self._sessions = InMemorySessionService()
        self._session_context: dict[str, tuple[str, list[str], str]] = {}
        self._case_states: dict[str, SessionCaseState] = {}
        self._provider_registry = ProviderRegistry(settings)
        models = dict.fromkeys([settings.agent_model, settings.agent_fallback_model])  # ordered, deduped
        self._runners = [
            Runner(agent=build_agent(m), app_name=APP_NAME, session_service=self._sessions)
            for m in models
        ]

    async def _ensure_session(self, session_id: str) -> None:
        existing = await self._sessions.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        if existing is None:
            await self._sessions.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )

    async def _run(self, runner: Runner, session_id: str, text: str) -> AgentResult:
        await self._ensure_session(session_id)
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        reply, tools_used = "", []
        sources: set[str] = set()
        tool_hits: list[dict] = []
        artifacts: list[Artifact] = []

        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=msg
        ):
            for call in event.get_function_calls():
                tools_used.append(call.name)
            for resp in event.get_function_responses():
                body = resp.response or {}
                if resp.name == "search_knowledge":
                    hits = body.get("results", [])
                    if isinstance(hits, list):
                        tool_hits.extend(hits)
                        sources.update(r["source"] for r in hits if isinstance(r, dict) and "source" in r)
                elif resp.name == "get_complaint_guide" and body.get("status") == "ok":
                    artifacts.append(
                        Artifact(type="guide", title=body["title"], text=body["guide_text"])
                    )
                elif resp.name == "draft_complaint_letter" and body.get("status") == "ok":
                    artifacts.append(Artifact(type="letter", text=body["letter"]))
            if event.is_final_response() and event.content and event.content.parts:
                reply = "".join(p.text or "" for p in event.content.parts).strip()

        res = AgentResult(
            reply=clean_reply_text(reply) or FALLBACK_REPLY,
            tools_used=tools_used,
            sources=sorted(sources),
            artifacts=artifacts,
            citation_note="",
            provider="gemini",
        )
        # Attach internal tool hits for downstream evidence attribution in ask()
        res._tool_hits = tool_hits  # type: ignore[attr-defined]
        return res

    async def ask(self, session_id: str, text: str, lang: str = "en", is_voice: bool = False) -> AgentResult:
        # Dynamic conversational language detection: adapts per-turn to Hindi (Devanagari), Hinglish, or English
        # Controlled retrieval policy:
        # Non-conversational factual queries trigger domain-aware grounded knowledge injection.
        previous_context = self._session_context.get(session_id)
        raw_domains = classify_domains(text)
        is_follow_up = bool(
            previous_context
            and raw_domains == ["general"]
            and FOLLOW_UP_PATTERN.search(text)
        )
        if is_follow_up:
            context_text, domains, previous_lang = previous_context
            effective_lang = (
                previous_lang
                if not DEVANAGARI_CHARS.search(text)
                and not ENGLISH_INTENT.search(text)
                and not HINDI_INTENT.search(text)
                else detect_effective_language(text, fallback_lang=lang)
            )
            retrieval_text = f"{text}\nPrevious incident context: {context_text}"
        else:
            domains = raw_domains
            effective_lang = detect_effective_language(text, fallback_lang=lang)
            retrieval_text = text
        directive = REPLY_LANGUAGE.get(effective_lang, REPLY_LANGUAGE["en"])
        candidate_hits: list[dict] = []
        user_prompt = text

        if domains != ["conversation"]:
            try:
                # Cap retrieval to top 3-4 chunks to eliminate off-topic context clutter
                hits = get_kb().search(retrieval_text, k=4, domains=domains)
                if hits:
                    candidate_hits.extend(hits[:4])
                    context_blocks = format_evidence_context(candidate_hits)
                    grounding_rules = format_grounding_directive(
                        has_hits=True,
                        query_text=retrieval_text,
                        lang=effective_lang,
                    )
                    user_prompt = f"{context_blocks}\n\n{grounding_rules}"
                else:
                    user_prompt = format_grounding_directive(
                        has_hits=False,
                        query_text=retrieval_text,
                        lang=effective_lang,
                    )
            except Exception:
                log.warning("controlled pre-retrieval failed, falling back to tool calling", exc_info=True)

        # Early deterministic abstention for clear out-of-scope queries
        intent = classify_intent(text, lang=effective_lang)
        has_cyber_domain = any(d in CYBER_DOMAINS for d in domains)

        if not candidate_hits and not has_cyber_domain and domains != ["conversation"] and not is_follow_up:
            if intent.status == "out_of_scope" or is_out_of_scope_query(text, domains, candidate_hits):
                return AgentResult(
                    reply=get_scope_abstention_reply(lang=effective_lang),
                    tools_used=[],
                    sources=[],
                    artifacts=[],
                    citation_note="",
                )
            elif intent.status == "needs_clarification":
                return AgentResult(
                    reply=intent.clarification_question,
                    tools_used=[],
                    sources=[],
                    artifacts=[],
                    citation_note="",
                )

        if any(domain not in {"general", "conversation"} for domain in domains):
            context_to_store = (
                f"{previous_context[0]} {text}".strip()
                if is_follow_up and previous_context
                else text
            )
            self._session_context[session_id] = (
                context_to_store[-1500:],
                domains,
                effective_lang,
            )

        # Track user-oriented per-session incident state
        state = self._case_states.setdefault(session_id, SessionCaseState(session_id=session_id))
        state = extract_and_update_state(text, state, effective_lang=effective_lang)

        state_summary = state.to_context_summary()
        if state_summary and domains != ["conversation"]:
            user_prompt = f"{state_summary}\n\n{user_prompt}"

        if needs_triage_question(state) and domains != ["conversation"]:
            triage_directive = (
                "[TRIAGE DIRECTIVE: Essential fraud details are missing. "
                "Provide the single immediate safe action (call 1930) and ask ONE short question: "
                "'Paise kitne der pehle kate, aur transaction UPI se hua ya card se?']"
            )
            user_prompt = f"{triage_directive}\n\n{user_prompt}"
            state.triage_question_asked = True

        # Phase 6: Voice-first response style
        if is_voice and domains != ["conversation"]:
            user_prompt = f"{VOICE_DIRECTIVE}\n\n{user_prompt}"

        message = f"{directive}\n{user_prompt}"

        for i, runner in enumerate(self._runners):
            is_last = i == len(self._runners) - 1
            try:
                timeout = self._timeout if is_last else self._primary_timeout
                res = await asyncio.wait_for(self._run(runner, session_id, message), timeout)

                # Strip bracketed system directives
                res.reply = clean_reply_text(res.reply)

                # Combine pre-retrieved hits and any hits returned dynamically via search_knowledge tool
                all_hits = candidate_hits + getattr(res, "_tool_hits", [])

                # Deterministic high-risk safety guardrail post-processing (context-aware)
                retrieved_context_text = " ".join(h.get("text", "") for h in all_hits)
                sanitized_reply, _ = sanitize_high_risk_claims(
                    res.reply,
                    lang=effective_lang,
                    user_message=text,
                    retrieved_context=retrieved_context_text,
                )
                res.reply = sanitized_reply
                attributed_sources, citation_note = attribute_evidence(
                    reply=res.reply,
                    candidate_hits=all_hits,
                    domains=domains,
                    lang=effective_lang,
                    tools_used=res.tools_used,
                )
                if attributed_sources:
                    res.sources = attributed_sources
                elif res.sources and all_hits:
                    # Filter existing sources by utilization to prevent over-attribution
                    res.sources = attributed_sources

                if citation_note:
                    res.citation_note = citation_note

                log.info("Agent answered query for session %s using provider: %s", session_id, getattr(res, "provider", "gemini"))
                return res
            except genai_errors.APIError as e:
                if e.code not in FALLBACK_STATUS_CODES:
                    raise
                if not is_last:
                    log.warning("model call failed (code=%s); retrying on fallback model", e.code)
                    continue
                # Phase 5: Automatic fallback to non-Gemini provider (Sarvam, Groq, Ollama) on 429/timeout/5xx
                if hasattr(self, "_provider_registry") and self._provider_registry:
                    try:
                        non_gemini = [
                            prov
                            for name, prov in self._provider_registry.get_llm_order()
                            if name != "gemini"
                        ]
                        for prov in non_gemini:
                            if await prov.is_healthy():
                                try:
                                    llm_messages = [
                                        {"role": "system", "content": INSTRUCTION},
                                        {"role": "user", "content": message},
                                    ]
                                    provider_resp = await prov.generate(
                                        messages=llm_messages,
                                        temperature=0.2,
                                        max_tokens=1000,
                                    )
                                    if provider_resp and provider_resp.content:
                                        res = AgentResult(
                                            reply=clean_reply_text(provider_resp.content) or FALLBACK_REPLY,
                                            tools_used=[],
                                            sources=[],
                                            artifacts=[],
                                            citation_note="",
                                            provider=prov.name,
                                        )
                                        retrieved_context_text = " ".join(h.get("text", "") for h in candidate_hits)
                                        sanitized_reply, _ = sanitize_high_risk_claims(
                                            res.reply,
                                            lang=effective_lang,
                                            user_message=text,
                                            retrieved_context=retrieved_context_text,
                                        )
                                        res.reply = sanitized_reply
                                        attributed_sources, citation_note = attribute_evidence(
                                            reply=res.reply,
                                            candidate_hits=candidate_hits,
                                            domains=domains,
                                            lang=effective_lang,
                                            tools_used=[],
                                        )
                                        if attributed_sources:
                                            res.sources = attributed_sources
                                        if citation_note:
                                            res.citation_note = citation_note
                                        log.info("Agent answered query for session %s using fallback provider: %s", session_id, res.provider)
                                        return res
                                except Exception as fallback_err:
                                    log.warning("Fallback provider '%s' failed: %s", getattr(prov, "name", "unknown"), fallback_err)
                                    if hasattr(self, "_provider_registry") and self._provider_registry:
                                        self._provider_registry.record_error(getattr(prov, "name", "unknown"), "llm_fallback", fallback_err)
                    except Exception as fallback_err:
                        log.warning("Provider fallback attempt failed: %s", fallback_err)
                raise
            except TimeoutError:
                if not is_last:
                    log.warning("model call timed out; retrying on fallback model")
                    continue
                if hasattr(self, "_provider_registry") and self._provider_registry:
                    try:
                        non_gemini = [
                            prov
                            for name, prov in self._provider_registry.get_llm_order()
                            if name != "gemini"
                        ]
                        for prov in non_gemini:
                            if await prov.is_healthy():
                                try:
                                    llm_messages = [
                                        {"role": "system", "content": INSTRUCTION},
                                        {"role": "user", "content": message},
                                    ]
                                    provider_resp = await prov.generate(
                                        messages=llm_messages,
                                        temperature=0.2,
                                        max_tokens=1000,
                                    )
                                    if provider_resp and provider_resp.content:
                                        res = AgentResult(
                                            reply=clean_reply_text(provider_resp.content) or FALLBACK_REPLY,
                                            tools_used=[],
                                            sources=[],
                                            artifacts=[],
                                            citation_note="",
                                            provider=prov.name,
                                        )
                                        retrieved_context_text = " ".join(h.get("text", "") for h in candidate_hits)
                                        sanitized_reply, _ = sanitize_high_risk_claims(
                                            res.reply,
                                            lang=effective_lang,
                                            user_message=text,
                                            retrieved_context=retrieved_context_text,
                                        )
                                        res.reply = sanitized_reply
                                        attributed_sources, citation_note = attribute_evidence(
                                            reply=res.reply,
                                            candidate_hits=candidate_hits,
                                            domains=domains,
                                            lang=effective_lang,
                                            tools_used=[],
                                        )
                                        if attributed_sources:
                                            res.sources = attributed_sources
                                        if citation_note:
                                            res.citation_note = citation_note
                                        log.info("Agent answered query for session %s using fallback provider: %s", session_id, res.provider)
                                        return res
                                except Exception as fallback_err:
                                    log.warning("Fallback provider '%s' failed on timeout: %s", getattr(prov, "name", "unknown"), fallback_err)
                                    if hasattr(self, "_provider_registry") and self._provider_registry:
                                        self._provider_registry.record_error(getattr(prov, "name", "unknown"), "llm_fallback", fallback_err)
                    except Exception as fallback_err:
                        log.warning("Provider fallback attempt failed on timeout: %s", fallback_err)
                raise
        raise RuntimeError("no runners configured")  # unreachable: __init__ always builds >= 1

    async def ask_stream_sentences(
        self, session_id: str, text: str, lang: str = "en", is_voice: bool = True
    ):
        """Yield (sanitized_sentence, effective_lang) sentence by sentence for live voice playback.
        Ensures safety sanitization is strictly executed on EACH sentence BEFORE yielding.
        """
        res = await self.ask(session_id, text, lang=lang, is_voice=is_voice)
        sentences = split_into_sentences(res.reply)
        effective_lang = detect_effective_language(text, fallback_lang=lang)
        for s in sentences:
            if not s.strip():
                continue
            sanitized_s, _ = sanitize_high_risk_claims(s, lang=effective_lang)
            yield sanitized_s, effective_lang
