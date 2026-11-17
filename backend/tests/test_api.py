import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import AgentResult, Artifact
from app.services.speech import pcm_to_wav

SID = "test-session-1234"


class FakeAgent:
    async def ask(self, session_id, text, lang):
        return AgentResult(
            reply=f"echo:{lang}:{text}",
            tools_used=["search_knowledge"],
            sources=["portal.md"],
            artifacts=[Artifact(type="letter", text="LETTER")],
        )


class FakeSpeech:
    def __init__(self, transcript="my money was stolen"):
        self.transcript = transcript

    async def transcribe(self, audio, mime):
        return self.transcript

    async def synthesize(self, text):
        return pcm_to_wav(b"\x00\x00" * 10)


def settings(**kw):
    return Settings(google_api_key="x", max_audio_bytes=1000, rate_limit="1000/minute", **kw)


@pytest.fixture
def client():
    with TestClient(create_app(settings(), FakeAgent(), FakeSpeech())) as c:
        yield c


def wav_file(n=100):
    return {"audio": ("a.wav", b"\x00" * n, "audio/wav")}


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_chat_returns_agent_result(client):
    r = client.post("/api/chat", json={"session_id": SID, "message": "help"})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "echo:en:help"
    assert body["artifacts"][0] == {"type": "letter", "text": "LETTER", "title": ""}


def test_chat_passes_ui_language(client):
    r = client.post("/api/chat", json={"session_id": SID, "message": "help", "lang": "hi"})
    assert r.json()["reply"] == "echo:hi:help"
    assert client.post("/api/chat", json={"session_id": SID, "message": "x", "lang": "fr"}).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": SID, "message": ""},
        {"session_id": SID, "message": "x" * 1001},
        {"session_id": "short", "message": "hi"},
        {"session_id": "bad id with spaces!!", "message": "hi"},
        {"message": "hi"},
    ],
)
def test_chat_validation(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_voice_happy_path(client):
    r = client.post("/api/voice", data={"session_id": SID}, files=wav_file())
    assert r.status_code == 200
    assert r.json()["transcript"] == "my money was stolen"
    assert r.json()["reply"] == "echo:en:my money was stolen"
    r = client.post("/api/voice", data={"session_id": SID, "lang": "hi"}, files=wav_file())
    assert r.json()["reply"] == "echo:hi:my money was stolen"
    assert client.post("/api/voice", data={"session_id": SID, "lang": "zz"}, files=wav_file()).status_code == 422


def test_voice_rejects_bad_mime(client):
    files = {"audio": ("a.exe", b"abc", "application/octet-stream")}
    assert client.post("/api/voice", data={"session_id": SID}, files=files).status_code == 415


def test_voice_accepts_webm(client):
    """audio/webm is accepted — Chrome and Firefox MediaRecorder default format.
    Gemini STT supports audio/webm (confirmed aistudio.google.com/docs/audio 2026-11-16).
    """
    files = {"audio": ("recording.webm", b"\x00" * 100, "audio/webm")}
    r = client.post("/api/voice", data={"session_id": SID}, files=files)
    # 200 (transcript returned) or 422 (no speech detected) both mean MIME was accepted
    assert r.status_code in (200, 422), f"Expected 200/422 for audio/webm, got {r.status_code}"


def test_voice_accepts_webm_with_codec_suffix(client):
    """audio/webm;codecs=opus (Chrome MediaRecorder exact output) is accepted after codec suffix strip."""
    files = {"audio": ("recording.webm", b"\x00" * 100, "audio/webm;codecs=opus")}
    r = client.post("/api/voice", data={"session_id": SID}, files=files)
    assert r.status_code in (200, 422), f"Expected 200/422 for audio/webm;codecs=opus, got {r.status_code}"


def test_voice_rejects_truly_unsupported_mime(client):
    """Truly unsupported MIME types (video/mp4, application/pdf) still return 415."""
    for bad_mime in ("video/mp4", "application/pdf", "text/plain", "image/jpeg"):
        files = {"audio": ("file", b"abc", bad_mime)}
        r = client.post("/api/voice", data={"session_id": SID}, files=files)
        assert r.status_code == 415, f"Expected 415 for {bad_mime}, got {r.status_code}"


def test_voice_rejects_too_large_and_empty(client):
    assert client.post("/api/voice", data={"session_id": SID}, files=wav_file(2000)).status_code == 413
    assert client.post("/api/voice", data={"session_id": SID}, files=wav_file(0)).status_code == 413


def test_voice_no_speech():
    with TestClient(create_app(settings(), FakeAgent(), FakeSpeech(transcript=""))) as c:
        assert c.post("/api/voice", data={"session_id": SID}, files=wav_file()).status_code == 422


def test_tts_returns_wav(client):
    r = client.post("/api/tts", json={"text": "hello"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_rate_limit():
    s = Settings(google_api_key="x", rate_limit="2/minute")
    with TestClient(create_app(s, FakeAgent(), FakeSpeech())) as c:
        codes = [c.post("/api/chat", json={"session_id": SID, "message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_frontend_chat_timeout_and_fallback_distinction():
    """Verify that index.html sets a safe fetch timeout >= 30000ms, renders citations, and distinguishes offline fallback."""
    import re
    from pathlib import Path

    index_path = Path(__file__).resolve().parent.parent.parent / "index.html"
    assert index_path.exists(), "index.html not found"
    content = index_path.read_text(encoding="utf-8")

    # Safe timeout check (Task 2)
    match = re.search(r"setTimeout\s*\(\s*\(\)\s*=>\s*controller\.abort\(\)\s*,\s*(\d+)\s*\)", content)
    assert match, "Abort timeout not found in index.html"
    timeout_ms = int(match.group(1))
    assert timeout_ms >= 30000, f"Frontend timeout {timeout_ms}ms is too short for backend LLM execution"

    # Citation note visibility in addBubble (Task 3)
    assert "citationNote" in content, "citationNote parameter not found in addBubble"
    assert "📌" in content, "Citation badge rendering not found in index.html"

    # Offline fallback distinction (Task 2)
    assert "isFallback" in content, "isFallback parameter not found in addBubble"
    assert "Offline Guidance" in content or "ऑफलाइन गाइड" in content, "Fallback distinction message not found"
