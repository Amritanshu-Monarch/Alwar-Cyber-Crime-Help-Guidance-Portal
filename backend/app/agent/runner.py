import asyncio
import logging

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import errors as genai_errors
from google.genai import types

from app.agent.prompts import INSTRUCTION
from app.agent.tools import ALL_TOOLS
from app.config import Settings
from app.schemas import AgentResult, Artifact

log = logging.getLogger(__name__)

APP_NAME = "alwar_cyber_agent"
USER_ID = "citizen"  # anonymous; sessions are keyed by the client-generated session_id
REPLY_LANGUAGE = {
    "en": "[Reply in English.]",
    "hi": "[Reply in Hindi written in Roman (English) letters, i.e. natural spoken Hinglish like 'Aap sabse pehle 1930 par call karein'. Never use Devanagari. Keep tool language argument as hi.]",
}
FALLBACK_STATUS_CODES = {404, 429, 500, 502, 503, 504}  # model gone / rate limited / overloaded
FALLBACK_REPLY = "Sorry, I could not answer that. Please call 1930 or 8764874306."


def build_agent(model: str) -> Agent:
    return Agent(
        name="alwar_cyber_assistant",
        model=model,
        description="Cyber crime help and guidance assistant for Alwar citizens.",
        instruction=INSTRUCTION,
        tools=ALL_TOOLS,
        generate_content_config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=700),
    )


class AgentService:
    # ponytail: in-memory sessions grow unbounded and die on restart.
    # Swap InMemorySessionService for ADK DatabaseSessionService if you need persistence or eviction.
    def __init__(self, settings: Settings):
        self._timeout = settings.agent_timeout_s
        self._primary_timeout = settings.agent_primary_timeout_s
        self._sessions = InMemorySessionService()
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
        artifacts: list[Artifact] = []

        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=msg
        ):
            for call in event.get_function_calls():
                tools_used.append(call.name)
            for resp in event.get_function_responses():
                body = resp.response or {}
                if resp.name == "search_knowledge":
                    sources.update(r["source"] for r in body.get("results", []))
                elif resp.name == "get_complaint_guide" and body.get("status") == "ok":
                    artifacts.append(
                        Artifact(type="guide", title=body["title"], text=body["guide_text"])
                    )
                elif resp.name == "draft_complaint_letter" and body.get("status") == "ok":
                    artifacts.append(Artifact(type="letter", text=body["letter"]))
            if event.is_final_response() and event.content and event.content.parts:
                reply = "".join(p.text or "" for p in event.content.parts).strip()

        return AgentResult(
            reply=reply or FALLBACK_REPLY,
            tools_used=tools_used,
            sources=sorted(sources),
            artifacts=artifacts,
        )

    async def ask(self, session_id: str, text: str, lang: str) -> AgentResult:
        # The UI language decides the reply language; models drift if left to guess.
        directive = REPLY_LANGUAGE.get(lang, REPLY_LANGUAGE["en"])
        message = f"{directive}\n{text}"
        for i, runner in enumerate(self._runners):
            is_last = i == len(self._runners) - 1
            try:
                timeout = self._timeout if is_last else self._primary_timeout
                return await asyncio.wait_for(self._run(runner, session_id, message), timeout)
            except genai_errors.APIError as e:
                if e.code not in FALLBACK_STATUS_CODES or is_last:
                    raise
                log.warning("model call failed (code=%s); retrying on fallback model", e.code)
            except TimeoutError:
                if is_last:
                    raise
                log.warning("model call timed out; retrying on fallback model")
        raise RuntimeError("no runners configured")  # unreachable: __init__ always builds >= 1
