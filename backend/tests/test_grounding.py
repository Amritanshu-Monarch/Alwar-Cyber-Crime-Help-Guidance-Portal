"""Comprehensive tests for Sub-phase B1: Evidence Grounding and Source Attribution.

Verifies:
1. Queries with relevant retrieved evidence.
2. Queries with no retrieved evidence.
3. General cyber-safety guidance.
4. Source attribution and citation formatting.
5. Multiple source documents.
6. Hindi and English responses.
7. Backward compatibility of the API response schema.
8. Non-empty retrieval does not falsely attribute ungrounded claims.
9. Explicit communication of limitations when official evidence is insufficient.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.facts import CONTROL_ROOM
from app.agent.grounding import (
    attribute_evidence,
    extract_distinctive_terms,
    format_evidence_context,
    format_grounding_directive,
    is_chunk_utilized,
    is_unverified_limitation_expressed,
    requests_detailed_answer,
)
from app.agent.runner import AgentService
from app.config import Settings
from app.main import create_app
from app.schemas import AgentResult, Artifact, ChatIn, VoiceOut
from app.services.speech import pcm_to_wav


# Sample fixture hits
SAMPLE_HITS = [
    {
        "source": "mrm_portal_recovery.yaml",
        "section": "Lien Removal & Unfreeze",
        "text": "Immediately call 1930 to freeze funds on NCRP and report on cybercrime.gov.in. Use MRM portal for lien removal.",
        "authority": "official_government",
        "jurisdiction": "national",
        "domain": "reporting",
    },
    {
        "source": "bank_account_freeze.yaml",
        "section": "Bank Freeze Procedures",
        "text": "State cyber cell directs bank to freeze suspicious accounts under Section 102 CrPC / 106 BNSS. Submit KYC and UTR to IO.",
        "authority": "official_government",
        "jurisdiction": "national",
        "domain": "reporting",
    },
    {
        "source": "morphed_photos_stopncii.yaml",
        "section": "StopNCII Hash Creation",
        "text": "Victims of morphed or intimate photos should create a hash at StopNCII.org to prevent viral distribution.",
        "authority": "official_government",
        "jurisdiction": "national",
        "domain": "deepfake",
    },
]


# ==============================================================================
# 1. Evidence Prompt Construction Tests
# ==============================================================================

def test_evidence_prompt_construction_with_hits():
    """Verify format_evidence_context and directive when relevant evidence exists."""
    ctx = format_evidence_context(SAMPLE_HITS[:2])
    assert "[VERIFIED OFFICIAL KNOWLEDGE CONTEXT (Ground Truth)]:" in ctx
    assert "--- Document [1]: mrm_portal_recovery.yaml ---" in ctx
    assert "--- Document [2]: bank_account_freeze.yaml ---" in ctx
    assert "Authority: official_government" in ctx
    assert "Section: Lien Removal & Unfreeze" in ctx

    directive = format_grounding_directive(has_hits=True, query_text="How to unfreeze account?", lang="en")
    assert "THREE-TIER INFORMATION DISTINCTION" in directive
    assert "VERIFIED OFFICIAL FACTS" in directive
    assert "GENERAL CYBER-SAFETY GUIDANCE" in directive
    assert "UNVERIFIED / INSUFFICIENT EVIDENCE" in directive
    assert "ANTI-HALLUCINATION" in directive
    assert CONTROL_ROOM in directive
    assert "1930" in directive


def test_detailed_procedure_requests_are_not_truncated():
    """Explicit full-procedure requests receive complete verified steps."""
    assert requests_detailed_answer("pura procedure do") is True
    directive = format_grounding_directive(
        has_hits=True,
        query_text="pura procedure do",
        lang="hinglish",
    )
    assert "complete verified numbered steps" in directive
    assert "Max 2-3 short" not in directive


def test_grounding_directive_prioritizes_relevant_evidence_for_help_requests():
    directive = format_grounding_directive(
        has_hits=True,
        query_text="My Telegram account was hacked. Please help me recover it.",
        lang="en",
    )
    assert "Use only the relevant document" in directive
    assert "Do not mix steps from unrelated platforms or incident types" in directive
    assert "If the evidence contains an applicable step, include it" in directive


def test_evidence_prompt_construction_without_hits():
    """Verify directive when no official records match the query."""
    directive = format_grounding_directive(has_hits=False, query_text="Who is the SHO?", lang="en")
    assert "NO MATCHING OFFICIAL RECORDS" in directive
    assert "ANTI-HALLUCINATION GUARD" in directive
    assert "Do NOT guess, fabricate, or hallucinate" in directive
    assert CONTROL_ROOM in directive
    assert "1930" in directive


# ==============================================================================
# 2. Source Attribution & Citation Formatting Tests
# ==============================================================================

def test_source_attribution_single_relevant_evidence():
    """Queries with a single relevant retrieved evidence document are attributed correctly."""
    reply = "You must call 1930 right away to register the transaction and follow MRM portal steps for lien removal."
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="en")
    assert sources == ["mrm_portal_recovery.yaml"]
    assert note == "Verified from official knowledge base: mrm_portal_recovery.yaml."


def test_source_attribution_multiple_relevant_sources():
    """Queries supported by multiple source documents attribute all utilized sources in sorted order."""
    reply = "Call 1930 for MRM portal lien removal, and provide your KYC and UTR to the IO under Section 106 BNSS to unfreeze your bank account."
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="en")
    assert sources == ["bank_account_freeze.yaml", "mrm_portal_recovery.yaml"]
    assert note == "Verified from official knowledge base: bank_account_freeze.yaml, mrm_portal_recovery.yaml."


def test_platform_specific_source_requires_platform_in_reply():
    """Do not cite Facebook or Instagram recovery docs for a Telegram-only answer."""
    hits = [
        {
            "source": "15_facebook_hacked_recovery.yaml",
            "section": "Facebook Recovery",
            "text": "Recover a hacked Facebook account and secure unknown sessions.",
        },
        {
            "source": "18_telegram_hacked_scam.yaml",
            "section": "Telegram Recovery",
            "text": "Terminate unknown Telegram sessions and enable Two-Step Verification.",
        },
    ]

    sources, _ = attribute_evidence(
        "Telegram settings mein unknown sessions terminate karein aur Two-Step Verification enable karein.",
        hits,
        ["account_hacking", "social_media"],
        lang="hinglish",
    )

    assert sources == ["18_telegram_hacked_scam.yaml"]


def test_no_retrieved_evidence_leaves_sources_empty():
    """Queries with no matching evidence do not fabricate sources and communicate lack of KB records."""
    reply = "I only assist with cyber crime reporting and online safety in Alwar."
    sources, note = attribute_evidence(reply, [], ["general"], lang="en")
    assert sources == []
    assert note == "No matching official records found in knowledge base."


def test_unutilized_hit_does_not_get_false_attribution():
    """CRITICAL: The presence of a retrieved chunk does NOT prove the claim is grounded.
    If the reply does not actually use the source's content, that source must not be attributed.
    """
    # Financial chunks are present in candidates, but response only discusses morphed photos StopNCII
    reply = "Victims of morphed pictures should create a digital hash on StopNCII to stop viral spread."
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["deepfake"], lang="en")
    assert "mrm_portal_recovery.yaml" not in sources
    assert "bank_account_freeze.yaml" not in sources
    assert sources == ["morphed_photos_stopncii.yaml"]


def test_unrelated_reply_with_hits_attributes_no_sources():
    """If the agent gives an irrelevant or generic answer that fails to use any retrieved chunk,
    do not treat the mere presence of candidate hits as proof of grounding.
    """
    reply = "You should check your calendar and wait until next week."
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="en")
    assert sources == []
    assert "Information not verified against specific official knowledge base records." in note


# ==============================================================================
# 3. Explicit Communication of Limitations & Insufficient Evidence
# ==============================================================================

def test_explicit_unverified_detail_limitation_english():
    """When the user asks for a detail not present in records, the agent's explicit disclaimer
    is recognized and attributed with an unverified limitation note.
    """
    reply = (
        "I could not verify that specific detail from available official information. "
        f"Please confirm with the Alwar Cyber Control Room at {CONTROL_ROOM} or 1930."
    )
    assert is_unverified_limitation_expressed(reply) is True
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="en")
    assert sources == []
    assert "Specific details could not be verified from available official records." in note
    assert CONTROL_ROOM in note


def test_explicit_unverified_detail_limitation_hindi():
    """Hindi limitation communication properly sets the localized unverified citation note."""
    reply = "उपलब्ध आधिकारिक रिकॉर्ड में यह विशिष्ट विवरण सत्यापित नहीं हो सका। कृपया 1930 पर संपर्क करें।"
    assert is_unverified_limitation_expressed(reply) is True
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="hi")
    assert sources == []
    assert "उपलब्ध आधिकारिक रिकॉर्ड से विशिष्ट विवरण सत्यापित नहीं हो सके" in note


def test_mixed_verified_and_unverified_detail_preserves_attribution():
    """When a response contains one verified claim AND an unverified claim,
    it must NOT discard the valid source attribution for the verified claim.
    """
    reply = (
        "Call 1930 immediately to report on the MRM portal for lien removal. "
        "However, I could not verify the personal phone number of the SHO from available official records. "
        f"Please confirm with the Alwar Cyber Control Room at {CONTROL_ROOM}."
    )
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="en")
    assert sources == ["mrm_portal_recovery.yaml"]
    assert "Verified from official knowledge base: mrm_portal_recovery.yaml." in note
    assert "Additional specific details could not be verified" in note


# ==============================================================================
# 4. General Cyber-Safety Guidance Tests
# ==============================================================================

def test_general_cyber_safety_guidance_english():
    """General cyber safety advice is distinguished from statutory/official filing procedures."""
    reply = "Always create a strong, unique password and enable two-factor authentication (2FA) to secure your account."
    sources, note = attribute_evidence(reply, [], ["cyber_safety"], lang="en")
    assert sources == []
    assert note == "General cyber-safety guidance (unverified against specific statutory/official filing records)."


def test_general_cyber_safety_guidance_hindi():
    """General cyber safety advice in Hindi receives the localized general guidance note."""
    reply = "अपना पासवर्ड किसी के साथ साझा न करें और हमेशा 2FA सुरक्षा चालू रखें।"
    sources, note = attribute_evidence(reply, [], ["cyber_safety"], lang="hi")
    assert sources == []
    assert note == "सामान्य साइबर सुरक्षा दिशानिर्देश (विशिष्ट आधिकारिक रिकॉर्ड से असत्यापित)।"


# ==============================================================================
# 5. Hindi and English Source Attribution Tests
# ==============================================================================

def test_hindi_response_source_attribution():
    """Hindi/Hinglish factual reply grounded in official KB formats citation note in Hindi."""
    reply = "Aap sabse pehle 1930 par call karein aur MRM portal ke madhyam se lien removal request darj karein."
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["reporting"], lang="hi")
    assert sources == ["mrm_portal_recovery.yaml"]
    assert "आधिकारिक ज्ञानकोष से सत्यापित: mrm_portal_recovery.yaml." in note


def test_conversational_greetings_leave_citation_empty():
    """Conversational greetings produce empty sources and empty citation_note default."""
    for greet in ["Hello! How can I help you?", "Namaste! Main Alwar Cyber Help assistant hoon."]:
        sources, note = attribute_evidence(greet, [], ["conversation"], lang="en")
        assert sources == []
        assert note == ""


# ==============================================================================
# 6. Tool-Based Official Source Attribution Tests
# ==============================================================================

def test_tool_official_attribution():
    """Verify tool execution maps to verified official registry and procedure sources."""
    # 1. Guide tool
    s1, n1 = attribute_evidence("Here is your checklist.", [], ["reporting"], "en", tools_used=["get_complaint_guide"])
    assert s1 == ["official_complaint_guide"]
    assert "Checklist verified" in n1

    # 2. Station finder tool
    s2, n2 = attribute_evidence("Kotwali Police Station details displayed.", [], ["local_police"], "en", tools_used=["find_police_station"])
    assert s2 == ["alwar_police_stations_registry"]
    assert "Police station details verified" in n2

    # 3. Draft complaint letter tool
    s3, n3 = attribute_evidence("Your complaint letter is drafted below.", [], ["reporting"], "en", tools_used=["draft_complaint_letter"])
    assert s3 == ["rajasthan_police_complaint_format"]
    assert "Formal complaint letter drafted" in n3


def test_tool_invocation_without_output_utilization_does_not_attribute():
    """Verify tool invocation alone without response utilization does not produce false tool citation."""
    s, n = attribute_evidence("I am looking into that.", [], ["local_police"], "en", tools_used=["find_police_station"])
    assert s == []
    assert "alwar_police_stations_registry" not in s


# ==============================================================================
# 7. Backward Compatibility of API Response Schema Tests
# ==============================================================================

def test_agent_result_schema_defaults_and_mutation():
    """Verify AgentResult schema: citation_note safe default, mutable list factories."""
    res = AgentResult(reply="Hello")
    assert res.reply == "Hello"
    assert res.tools_used == []
    assert res.sources == []
    assert res.artifacts == []
    assert res.citation_note == ""

    # Verify mutating one instance list does not affect default factory of another
    res.tools_used.append("search_knowledge")
    res.sources.append("doc1.yaml")
    res2 = AgentResult(reply="World")
    assert res2.tools_used == []
    assert res2.sources == []
    assert res2.citation_note == ""


def test_voice_out_inherits_citation_note():
    """Verify VoiceOut inherits citation_note and serializes cleanly."""
    vo = VoiceOut(
        transcript="sample voice query",
        reply="sample reply",
        tools_used=["get_contacts"],
        sources=["portal.md"],
        artifacts=[],
        citation_note="Verified from official knowledge base: portal.md.",
    )
    data = vo.model_dump()
    assert data["transcript"] == "sample voice query"
    assert data["citation_note"] == "Verified from official knowledge base: portal.md."
    assert data["sources"] == ["portal.md"]


def test_api_chat_and_voice_include_citation_note():
    """Verify FastAPI /api/chat and /api/voice return citation_note in JSON without breaking existing fields."""
    class GroundedFakeAgent:
        async def ask(self, session_id, text, lang):
            return AgentResult(
                reply=f"Verified answer for {text}",
                tools_used=["search_knowledge"],
                sources=["mrm_portal_recovery.yaml"],
                artifacts=[Artifact(type="guide", title="Guide", text="Step 1")],
                citation_note="Verified from official knowledge base: mrm_portal_recovery.yaml.",
            )

    class FakeSpeech:
        async def transcribe(self, audio, mime):
            return "stolen money"

        async def synthesize(self, text):
            return pcm_to_wav(b"\x00\x00" * 10)

    app = create_app(Settings(google_api_key="x"), GroundedFakeAgent(), FakeSpeech())
    with TestClient(app) as client:
        # 1. /api/chat backward compatibility
        r_chat = client.post("/api/chat", json={"session_id": "session-12345", "message": "frozen account", "lang": "en"})
        assert r_chat.status_code == 200
        chat_body = r_chat.json()
        assert "reply" in chat_body
        assert "tools_used" in chat_body
        assert "sources" in chat_body
        assert "artifacts" in chat_body
        assert "citation_note" in chat_body
        assert chat_body["citation_note"] == "Verified from official knowledge base: mrm_portal_recovery.yaml."
        assert chat_body["sources"] == ["mrm_portal_recovery.yaml"]

        # 2. /api/voice backward compatibility
        wav_payload = {"audio": ("sample.wav", b"\x00" * 100, "audio/wav")}
        r_voice = client.post("/api/voice", data={"session_id": "session-12345", "lang": "en"}, files=wav_payload)
        assert r_voice.status_code == 200
        voice_body = r_voice.json()
        assert voice_body["transcript"] == "stolen money"
        assert voice_body["citation_note"] == "Verified from official knowledge base: mrm_portal_recovery.yaml."
        assert voice_body["sources"] == ["mrm_portal_recovery.yaml"]


# ==============================================================================
# 8. End-to-End AgentService Grounding Integration Test
# ==============================================================================

def test_agent_service_ask_end_to_end_grounding(monkeypatch):
    """Verify AgentService.ask builds grounded evidence context and attributes sources properly."""
    class FakeKB:
        def search(self, query, k=3, domains=None):
            return [SAMPLE_HITS[0]]

    monkeypatch.setattr("app.agent.runner.get_kb", lambda: FakeKB())

    svc = AgentService(Settings(google_api_key="x", agent_model="model-a", agent_fallback_model="model-a"))
    captured_messages = []

    async def fake_run(runner, session_id, message):
        captured_messages.append(message)
        # Simulate grounded model response utilizing MRM portal facts
        return AgentResult(
            reply="Aap sabse pehle 1930 par call karein aur MRM portal ke zariye account lien hatwane ki request karein.",
            tools_used=[],
            sources=[],
            artifacts=[],
        )

    svc._run = fake_run

    # Execute ask with a query in the reporting domain
    result = asyncio.run(svc.ask("session-12345", "mere paise freeze ho gaye mrm portal se kaise milenge", "hi"))

    # 1. Assert pre-retrieval injected verified context and grounding rules into message
    assert len(captured_messages) == 1
    sent_prompt = captured_messages[0]
    assert "[VERIFIED OFFICIAL KNOWLEDGE CONTEXT (Ground Truth)]:" in sent_prompt
    assert "THREE-TIER INFORMATION DISTINCTION" in sent_prompt

    # 2. Assert result has grounded source and citation note
    assert result.sources == ["mrm_portal_recovery.yaml"]
    assert "आधिकारिक ज्ञानकोष से सत्यापित: mrm_portal_recovery.yaml." in result.citation_note
