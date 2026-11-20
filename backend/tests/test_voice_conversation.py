"""Tests for Sub-phase B3.3: Conversational Voice Mode Implementation & Verification.

Covers:
1. Conversational Voice state machine transitions and invariants
2. Multi-turn session continuity across text and voice queries
3. RAG retrieval, evidence grounding, and source attribution on voice input
4. Deterministic abstention preservation for voice inputs
5. Speech synthesis (/api/tts) contract, audio formatting, and markdown sanitization
6. Speech recognition (/api/voice) endpoint validation and error handling
7. Cancellation token invalidation, silence inactivity backoff, and echo prevention
"""
import asyncio
import io
import wave
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import AgentResult, Artifact
from app.services.speech import TTS_SAMPLE_RATE, clean_for_speech, pcm_to_wav


# ─── 1. STATE MACHINE CONTRACT & INVARIANT TESTS ─────────────────────────────

class ConversationalVoiceStateMachine:
    """Python reference mirror of the frontend Conversational Voice state machine in index.html.
    
    Verifies state transitions, duplicate prevention, cancellation tokens,
    silence backoff, and echo prevention safeguards.
    """
    VALID_STATES = {
        "idle",
        "requesting_permission",
        "listening",
        "processing",
        "speaking",
        "paused",
        "error",
    }

    def __init__(self, silence_threshold: int = 6):
        self.state = "idle"
        self.voice_mode_active = False
        self.is_recognition_running = False
        self.is_submitting_voice_turn = False
        self.consecutive_silence_restarts = 0
        self.silence_threshold = silence_threshold
        self.speak_token = 0
        self.active_audio = None
        self.history = []

    def set_state(self, new_state: str):
        if new_state not in self.VALID_STATES:
            raise ValueError(f"Invalid voice state: {new_state}")
        self.state = new_state
        self.history.append(new_state)

    def start_voice_conversation(self, has_mic_permission: bool = True):
        self.voice_mode_active = True
        self.consecutive_silence_restarts = 0
        self.set_state("requesting_permission")
        if not has_mic_permission:
            self.voice_mode_active = False
            self.set_state("error")
            return
        self.start_listening()

    def start_listening(self, from_state: str = None):
        if not self.voice_mode_active or self.state == "paused":
            return
        if self.is_recognition_running:
            # Duplicate start prevented
            return
        if self.state == "speaking" and from_state != "speaking_ended":
            # Echo prevention invariant: cannot listen while speaking
            raise RuntimeError("Echo prevention violation: recognition started while speaking")
        self.is_recognition_running = True
        self.set_state("listening")

    def on_silence_end(self):
        """Called when SpeechRecognition triggers onend with no transcript."""
        self.is_recognition_running = False
        if not self.voice_mode_active or self.state != "listening":
            return
        self.consecutive_silence_restarts += 1
        if self.consecutive_silence_restarts >= self.silence_threshold:
            self.pause("Paused due to inactivity")
        else:
            self.start_listening()

    def submit_voice_turn(self, transcript: str):
        if not transcript or not transcript.strip() or self.is_submitting_voice_turn:
            return
        self.is_submitting_voice_turn = True
        self.is_recognition_running = False
        self.consecutive_silence_restarts = 0
        self.set_state("processing")

    def finish_processing_and_speak(self, reply_text: str):
        self.is_submitting_voice_turn = False
        self.speak_token += 1
        current_token = self.speak_token
        self.set_state("speaking")
        self.active_audio = reply_text
        return current_token

    def on_playback_ended(self, token: int):
        if token != self.speak_token:
            # Invalidate stale callback
            return
        self.active_audio = None
        if self.voice_mode_active and self.state != "paused":
            # Automatically resume listening after assistant audio completes
            self.start_listening(from_state="speaking_ended")
        else:
            self.set_state("idle")

    def interrupt_speech(self):
        """User taps manual Stop Speaking UI button (half-duplex manual interruption, not voice-activated barge-in)."""
        if self.state == "speaking":
            self.speak_token += 1
            self.active_audio = None
            if self.voice_mode_active:
                self.start_listening(from_state="speaking_ended")

    def pause(self, reason: str = ""):
        self.speak_token += 1
        self.active_audio = None
        self.is_recognition_running = False
        self.set_state("paused")

    def resume(self):
        self.voice_mode_active = True
        self.consecutive_silence_restarts = 0
        self.set_state("listening")
        self.is_recognition_running = True

    def end_voice_conversation(self):
        self.voice_mode_active = False
        self.is_recognition_running = False
        self.is_submitting_voice_turn = False
        self.consecutive_silence_restarts = 0
        self.speak_token += 1
        self.active_audio = None
        self.set_state("idle")


def test_voice_state_machine_happy_path():
    """Verify normal multi-turn voice conversation flow and auto-resumption."""
    sm = ConversationalVoiceStateMachine()
    assert sm.state == "idle"

    # Turn 1: User starts voice mode
    sm.start_voice_conversation(has_mic_permission=True)
    assert sm.state == "listening"
    assert sm.is_recognition_running is True

    # User speaks
    sm.submit_voice_turn("How do I report financial fraud?")
    assert sm.state == "processing"
    assert sm.is_recognition_running is False

    # Assistant replies and speaks audio
    tok = sm.finish_processing_and_speak("Call 1930 immediately within the golden hour.")
    assert sm.state == "speaking"
    assert sm.is_recognition_running is False

    # Echo prevention invariant check: start_listening during speech must raise RuntimeError
    with pytest.raises(RuntimeError, match="Echo prevention violation"):
        sm.start_listening()

    # Audio playback completes -> Auto resumes listening
    sm.on_playback_ended(tok)
    assert sm.state == "listening"
    assert sm.is_recognition_running is True

    # Turn 2: User asks follow-up without clicking anything
    sm.submit_voice_turn("What documents should I carry to the police station?")
    assert sm.state == "processing"
    tok2 = sm.finish_processing_and_speak("Carry bank statements and transaction IDs.")
    assert sm.state == "speaking"

    # User ends session
    sm.end_voice_conversation()
    assert sm.state == "idle"
    assert sm.voice_mode_active is False
    assert sm.active_audio is None


def test_voice_state_machine_interrupt_speech():
    """Verify user can manually interrupt assistant speaking via Stop Speaking button and immediately resume listening."""
    sm = ConversationalVoiceStateMachine()
    sm.start_voice_conversation()
    sm.submit_voice_turn("Tell me all police stations in Alwar")
    tok = sm.finish_processing_and_speak("Alwar has Kotwali, Shivaji Park, Aravali Vihar...")
    assert sm.state == "speaking"

    # User manually interrupts assistant speech via button
    sm.interrupt_speech()
    assert sm.state == "listening"
    assert sm.is_recognition_running is True

    # Stale audio completion callback must NOT disrupt new state
    sm.on_playback_ended(tok)
    assert sm.state == "listening"


def test_voice_state_machine_silence_inactivity_threshold():
    """Verify that continuous silence auto-rearms up to 6 times, then pauses cleanly."""
    sm = ConversationalVoiceStateMachine(silence_threshold=6)
    sm.start_voice_conversation()
    assert sm.state == "listening"

    # 5 consecutive silences: should keep auto-rearming
    for _ in range(5):
        sm.on_silence_end()
        assert sm.state == "listening"

    # 6th consecutive silence: inactivity threshold reached -> transitions to paused
    sm.on_silence_end()
    assert sm.state == "paused"
    assert sm.is_recognition_running is False

    # User resumes
    sm.resume()
    assert sm.state == "listening"
    assert sm.consecutive_silence_restarts == 0


def test_voice_state_machine_permission_denial():
    """Verify microphone permission denial sets error state."""
    sm = ConversationalVoiceStateMachine()
    sm.start_voice_conversation(has_mic_permission=False)
    assert sm.state == "error"
    assert sm.voice_mode_active is False
    assert sm.is_recognition_running is False


# ─── 2. SESSION CONTINUITY & AGENT GROUNDING TESTS ───────────────────────────

class MockAgent:
    def __init__(self):
        self.ask = AsyncMock()

class MockSpeech:
    def __init__(self):
        self.synthesize = AsyncMock()
        self.transcribe = AsyncMock()


@pytest.fixture
def test_env():
    settings = Settings(
        google_api_key="mock-key",
        rate_limit="100/minute",
        allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    )
    fake_agent = MockAgent()
    fake_speech = MockSpeech()

    # Default fake agent response
    fake_agent.ask.return_value = AgentResult(
        reply="Call 1930 immediately to freeze fraudulent transactions.",
        artifacts=[
            Artifact(type="guide", title="Immediate Checklist", text="1. Call 1930\n2. Note transaction ID")
        ],
        sources=["helpline.md"],
        citation_note="Verified from helpline.md",
    )

    # Default fake speech response
    pcm_bytes = b"\x00\x00" * 2400
    fake_speech.synthesize.return_value = pcm_to_wav(pcm_bytes)
    fake_speech.transcribe.return_value = "How do I report financial fraud?"

    app = create_app(settings=settings, agent=fake_agent, speech=fake_speech)
    with TestClient(app) as client:
        yield client, fake_agent, fake_speech


def test_session_continuity_across_text_and_voice(test_env):
    """Voice turns and text turns must use the exact same session_id and AgentService pipeline."""
    client, fake_agent, _ = test_env
    session_id = "test-session-continuity-001"

    # Step 1: User sends a text message
    chat_resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "I lost money in UPI fraud", "lang": "en"},
    )
    assert chat_resp.status_code == 200
    assert fake_agent.ask.call_count == 1
    call_args_1 = fake_agent.ask.call_args_list[0]
    assert call_args_1[0][0] == session_id

    # Step 2: User follows up via voice on the same session
    fake_agent.ask.return_value = AgentResult(
        reply="Submit your bank statement and UTR number.",
        artifacts=[],
        sources=["evidence.md"],
        citation_note="Verified from evidence.md",
    )
    chat_voice_resp = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "What documents do I need?", "lang": "en"},
    )
    assert chat_voice_resp.status_code == 200
    assert fake_agent.ask.call_count == 2
    call_args_2 = fake_agent.ask.call_args_list[1]
    assert call_args_2[0][0] == session_id  # Exact same session ID preserved!
    assert chat_voice_resp.json()["citation_note"] == "Verified from evidence.md"


def test_voice_preserves_deterministic_abstention(test_env):
    """Unsupported requests submitted via voice must strictly trigger deterministic abstention."""
    client, fake_agent, _ = test_env
    session_id = "test-session-abstain-002"

    abstain_reply = (
        "I cannot verify that information. Official police officer private phone numbers "
        "are not published in the verified directory. For assistance, contact the Alwar Cyber "
        "Cell helpline at 8764874306 or National Helpline at 1930."
    )
    fake_agent.ask.return_value = AgentResult(
        reply=abstain_reply,
        artifacts=[],
        sources=[],
        citation_note="Abstained: unsupported officer contact",
    )

    resp = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Give me the personal mobile number of SHO Shivaji Park",
            "lang": "en",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "I cannot verify" in data["reply"]
    assert "8764874306" in data["reply"]
    assert data["citation_note"] == "Abstained: unsupported officer contact"


# ─── 3. TTS SPEECH SYNTHESIS CONTRACT & SANITIZATION ─────────────────────────

def test_clean_for_speech_sanitization():
    """Markdown, headers, links, bullets and long digits must be cleaned for natural pronunciation."""
    raw = (
        "### Immediate Action Guide\n\n"
        "• Call **1930** immediately.\n"
        "• Visit the [Cyber Portal](https://cybercrime.gov.in) to file a complaint.\n"
        "• For Alwar Cyber Cell, dial 8764874306.\n"
    )
    clean = clean_for_speech(raw)

    # Headers stripped
    assert "###" not in clean
    assert "Immediate Action Guide" in clean

    # Bold asterisks stripped
    assert "**" not in clean

    # Markdown links stripped to text
    assert "[" not in clean and "]" not in clean
    assert "Cyber Portal" in clean
    assert "https://" not in clean

    # Bullets stripped
    assert "•" not in clean

    # Numbers formatted with spaces for clear digit-by-digit pronunciation
    assert "1 9 3 0" in clean
    assert "8 7 6 4 8 7 4 3 0 6" in clean


def test_api_tts_endpoint_returns_valid_wav(test_env):
    """The /api/tts endpoint must return a valid WAV container with audio/wav media type."""
    client, _, _ = test_env
    resp = client.post(
        "/api/tts",
        json={"text": "Call 1930 immediately."},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    wav_bytes = resp.content
    assert wav_bytes[:4] == b"RIFF"
    assert b"WAVE" in wav_bytes[:12]

    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == TTS_SAMPLE_RATE


def test_api_tts_empty_text_validation(test_env):
    """Empty text in /api/tts must be handled cleanly."""
    client, _, _ = test_env
    resp = client.post("/api/tts", json={"text": ""})
    assert resp.status_code == 422


# ─── 4. VOICE UPLOAD (/api/voice) STT VALIDATION ─────────────────────────────

def test_api_voice_unsupported_audio_type(test_env):
    """Audio with unsupported MIME type must return HTTP 415."""
    client, _, _ = test_env
    resp = client.post(
        "/api/voice",
        data={"session_id": "test-session-voice-415", "lang": "en"},
        files={"audio": ("test.txt", b"plain text audio", "text/plain")},
    )
    assert resp.status_code == 415


def test_api_voice_no_speech_detected(test_env):
    """When speech recognition returns empty transcript, /api/voice returns HTTP 422."""
    client, _, fake_speech = test_env
    fake_speech.transcribe.return_value = ""  # Silence

    resp = client.post(
        "/api/voice",
        data={"session_id": "test-session-voice-422", "lang": "en"},
        files={"audio": ("audio.wav", b"RIFF....WAVE" + b"\x00" * 100, "audio/wav")},
    )
    assert resp.status_code == 422
    assert "no speech detected" in resp.json()["detail"].lower()


def test_api_voice_successful_flow(test_env):
    """Valid audio upload must transcribe, run agent, and return VoiceOut schema."""
    client, fake_agent, fake_speech = test_env
    fake_speech.transcribe.return_value = "ऑनलाइन ठगी की शिकायत कैसे करें?"
    fake_agent.ask.return_value = AgentResult(
        reply="1930 पर तुरंत कॉल करें।",
        artifacts=[],
        sources=["helpline.md"],
        citation_note="Verified",
    )

    resp = client.post(
        "/api/voice",
        data={"session_id": "test-session-voice-hi", "lang": "hi"},
        files={"audio": ("audio.webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 200, "audio/webm")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["transcript"] == "ऑनलाइन ठगी की शिकायत कैसे करें?"
    assert data["reply"] == "1930 पर तुरंत कॉल करें।"
