import asyncio

import pytest
from google.genai import errors as genai_errors

from app.agent.runner import AgentService
from app.config import Settings
from app.schemas import AgentResult


def service(**kw) -> AgentService:
    kw.setdefault("llm_provider_order", ["gemini"])
    return AgentService(Settings(google_api_key="x", **kw))


def api_error(cls, code: int):
    return cls(code, {"error": {"message": "boom", "status": "X"}})


def test_overloaded_primary_falls_back_to_second_model():
    svc = service(agent_model="primary-model", agent_fallback_model="fallback-model")
    used = []

    async def fake_run(runner, session_id, text):
        used.append(runner)
        if len(used) == 1:
            raise api_error(genai_errors.ServerError, 503)
        return AgentResult(reply="ok from fallback")

    svc._run = fake_run
    result = asyncio.run(svc.ask("session-abc-123", "hi", "en"))
    assert result.reply == "ok from fallback"
    assert len(used) == 2 and used[0] is not used[1]


def test_non_retryable_error_is_raised_immediately():
    svc = service(agent_model="primary-model", agent_fallback_model="fallback-model")
    calls = []

    async def fake_run(runner, session_id, text):
        calls.append(runner)
        raise api_error(genai_errors.ClientError, 400)

    svc._run = fake_run
    with pytest.raises(genai_errors.ClientError):
        asyncio.run(svc.ask("session-abc-123", "hi", "en"))
    assert len(calls) == 1


def test_error_from_last_model_propagates():
    svc = service(agent_model="primary-model", agent_fallback_model="fallback-model")

    async def fake_run(runner, session_id, text):
        raise api_error(genai_errors.ServerError, 503)

    svc._run = fake_run
    with pytest.raises(genai_errors.ServerError):
        asyncio.run(svc.ask("session-abc-123", "hi", "en"))


def test_slow_primary_times_out_then_falls_back():
    svc = service(agent_model="primary-model", agent_fallback_model="fallback-model", agent_primary_timeout_s=0.05)
    calls = []

    async def fake_run(runner, session_id, text):
        calls.append(runner)
        if len(calls) == 1:
            await asyncio.sleep(1)
        return AgentResult(reply="fast fallback")

    svc._run = fake_run
    assert asyncio.run(svc.ask("session-abc-123", "hi", "en")).reply == "fast fallback"
    assert len(calls) == 2


def test_same_primary_and_fallback_builds_one_runner():
    assert len(service(agent_model="same", agent_fallback_model="same")._runners) == 1


def test_language_directive_is_prefixed():
    svc = service()
    seen = []

    async def fake_run(runner, session_id, text):
        seen.append(text)
        return AgentResult(reply="ok")

    svc._run = fake_run
    asyncio.run(svc.ask("session-abc-123", "namaste", "hi"))
    assert seen[0].startswith("[Reply in Hindi") and seen[0].endswith("namaste")
