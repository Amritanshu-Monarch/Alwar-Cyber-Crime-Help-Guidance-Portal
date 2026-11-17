"""Comprehensive Unit and Integration Tests for Sub-phase B3: Deterministic Abstention.

Verifies:
1. Pure supported queries: authoritative evidence grounding with preserved source attribution.
2. Pure unsupported queries: explicit abstention with zero source hallucination, directing to 1930 / 8764874306.
3. Partially supported / mixed queries: supported claims answered with source attribution, unsupported details disclaimed.
4. Paraphrased unsupported queries: correctly abstains without inventing facts.
5. Irrelevant / out-of-domain queries: polite scope abstention without manufacturing cybercrime context.
6. Safety-critical guardrails:
   - Unknown 10-digit mobile numbers replaced with official Alwar Cyber Control Room (8764874306).
   - Authorized numbers (1930, 112, 1090, 8764874306) and transaction references (UTR/Txn) preserved.
   - Fabricated filing fees corrected with statutory 100% free clarification.
   - Fabricated 100% refund guarantees corrected with golden-hour dependency clarification.
   - Fabricated judicial outcomes (guaranteed arrest/bail denial) corrected with court discretion clarification.
7. Multilingual abstention detection: English, Hindi (Devanagari), and Hinglish (Latin script).
8. End-to-end AgentService.ask execution: early out-of-scope abstention and post-generation sanitization.
"""

from __future__ import annotations

import asyncio
import pytest

from app.agent.abstention import (
    CYBER_DOMAINS,
    DEV_ABSTENTION_SUBSTRINGS,
    VERIFIED_PHONE_NUMBERS,
    get_scope_abstention_reply,
    is_abstention_expressed,
    is_out_of_scope_query,
    sanitize_high_risk_claims,
)
from app.agent.facts import CONTROL_ROOM
from app.agent.grounding import (
    attribute_evidence,
    format_grounding_directive,
    is_unverified_limitation_expressed,
)
from app.agent.runner import AgentService
from app.config import Settings
from app.schemas import AgentResult

SAMPLE_HITS = [
    {
        "source": "upi_fraud_immediate_actions.md",
        "section": "Immediate Action",
        "authority": "official_government",
        "jurisdiction": "national",
        "text": "Call National Cyber Crime Helpline 1930 immediately within the golden hour to freeze the fraudster's beneficiary account.",
        "score": 0.89,
    },
    {
        "source": "alwar_police_directory.md",
        "section": "Kotwali Police Station",
        "authority": "rajasthan_police",
        "jurisdiction": "alwar",
        "text": "Alwar Kotwali Police Station handles jurisdiction in central Alwar. For cyber crimes, contact Alwar Cyber Control Room at 8764874306.",
        "score": 0.85,
    },
]


# ==============================================================================
# 1. Pure Supported Queries
# ==============================================================================

def test_pure_supported_query_attribution():
    """Authoritative answer utilizing knowledge base receives verified source attribution without abstention note."""
    reply = "Victims should immediately call 1930 within the golden hour to freeze suspect UPI accounts."
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["financial_fraud"], lang="en")
    
    assert sources == ["upi_fraud_immediate_actions.md"]
    assert "Verified from official knowledge base: upi_fraud_immediate_actions.md" in note
    assert "could not be verified" not in note
    assert is_abstention_expressed(reply) is False


# ==============================================================================
# 2. Pure Unsupported Official Queries
# ==============================================================================

def test_pure_unsupported_official_query_english():
    """Agent abstains from inventing unverified official information and points to official control room."""
    reply = (
        "I could not verify that specific official detail from available records. "
        f"Please confirm directly with the Alwar Cyber Control Room at {CONTROL_ROOM} or call 1930."
    )
    assert is_abstention_expressed(reply) is True
    assert is_unverified_limitation_expressed(reply) is True

    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["local_police"], lang="en")
    assert sources == []
    assert "Specific details could not be verified from available official records." in note
    assert CONTROL_ROOM in note


def test_pure_unsupported_official_query_hindi():
    """Hindi response explicitly expressing factual limitation has verified sources left empty."""
    reply = (
        "उपलब्ध आधिकारिक रिकॉर्ड में यह विशिष्ट विवरण सत्यापित नहीं हो सका। "
        f"कृपया अलवर साइबर कंट्रोल रूम ({CONTROL_ROOM}) या 1930 पर संपर्क करें।"
    )
    assert is_abstention_expressed(reply) is True
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["local_police"], lang="hi")
    assert sources == []
    assert "उपलब्ध आधिकारिक रिकॉर्ड से विशिष्ट विवरण सत्यापित नहीं हो सके।" in note
    assert CONTROL_ROOM in note


def test_pure_unsupported_official_query_hinglish():
    """Hinglish response expressing limitation is detected and assigned proper citation note."""
    reply = (
        "Ye specific detail official records me verify nahi ho saka. "
        f"Aap Alwar Cyber Control Room {CONTROL_ROOM} ya 1930 par sampark karein."
    )
    assert is_abstention_expressed(reply) is True
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["local_police"], lang="hi")
    assert sources == []
    assert "विशिष्ट विवरण सत्यापित नहीं हो सके" in note


# ==============================================================================
# 3. Partially Supported / Mixed Queries
# ==============================================================================

def test_mixed_query_preserves_verified_attribution_and_notes_limitation():
    """Mixed query: Agent answers supported part (UPI golden hour 1930) and disclaims unsupported part (SHO personal mobile).
    Preserves source attribution for the grounded chunk, while citation note mentions both verification and limitation.
    """
    reply = (
        "Call 1930 immediately within the golden hour to freeze the fraudster's account. "
        "However, the personal mobile number of the SHO could not be verified from available official records."
    )
    assert is_abstention_expressed(reply) is True

    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["financial_fraud", "local_police"], lang="en")
    # Verified source MUST NOT be discarded!
    assert sources == ["upi_fraud_immediate_actions.md"]
    # Citation note must acknowledge BOTH official verification AND the limitation
    assert "Verified from official knowledge base: upi_fraud_immediate_actions.md" in note
    assert "Additional specific details could not be verified from available official records." in note


def test_mixed_query_hindi():
    """Mixed query in Hindi retains verified source and adds limitation clause in Hindi note."""
    reply = (
        "गोल्डन आवर में तुरंत 1930 पर कॉल करें ताकि ठग के खाते में रकम होल्ड हो सके। "
        "थानाधिकारी का व्यक्तिगत मोबाइल नंबर उपलब्ध आधिकारिक रिकॉर्ड में सत्यापित नहीं हो सका।"
    )
    assert is_abstention_expressed(reply) is True

    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["financial_fraud", "local_police"], lang="hi")
    assert sources == ["upi_fraud_immediate_actions.md"]
    assert "आधिकारिक ज्ञानकोष से सत्यापित: upi_fraud_immediate_actions.md" in note
    assert "कुछ विशिष्ट विवरण उपलब्ध आधिकारिक रिकॉर्ड से सत्यापित नहीं हो सके।" in note


# ==============================================================================
# 4. Paraphrased Unsupported Queries
# ==============================================================================

def test_paraphrased_unsupported_sho_mobile_query():
    """Paraphrased queries for unverified facts (e.g. personal mobile) result in abstention detection."""
    reply = (
        "Direct personal phone line of the Kotwali SHO is not available in official records. "
        f"You can contact Alwar Cyber Control Room at {CONTROL_ROOM}."
    )
    assert is_abstention_expressed(reply) is True
    sources, note = attribute_evidence(reply, SAMPLE_HITS, ["local_police"], lang="en")
    assert sources == []
    assert CONTROL_ROOM in note


# ==============================================================================
# 5. Out-of-Domain and Irrelevant Queries
# ==============================================================================

@pytest.mark.parametrize(
    "query",
    [
        "How do I bake a chocolate cake?",
        "Tell me a funny joke",
        "Who won the cricket match yesterday?",
        "What is the capital of France?",
        "Write a python script for binary tree inversion",
        "पिज्जा बनाने की विधि क्या है?",
        "मुझे एक अच्छा चुटकुला सुनाओ",
        "Aaj ka mausam kaisa rahega?",
    ],
)
def test_out_of_domain_queries_detected(query):
    """Irrelevant queries outside digital safety and police domains are classified as out-of-scope."""
    domains = ["general"]
    assert is_out_of_scope_query(query, domains, candidate_hits=[]) is True


def test_in_scope_queries_not_marked_out_of_scope():
    """Queries mentioning cybercrime, banking fraud, police stations, or helplines remain in-scope."""
    assert is_out_of_scope_query("Someone stole my money on UPI", ["financial_fraud"], candidate_hits=[]) is False
    assert is_out_of_scope_query("Where is Alwar Kotwali police station?", ["local_police"], candidate_hits=[]) is False
    assert is_out_of_scope_query("What is 1930 helpline?", ["reporting"], candidate_hits=[]) is False
    assert is_out_of_scope_query("mere paise kat gaye", ["financial_fraud"], candidate_hits=[]) is False


def test_scope_abstention_replies():
    """Scope abstention replies are polite, concise, and direct to 1930 and Alwar Cyber Control Room."""
    en_reply = get_scope_abstention_reply(lang="en")
    assert "I can only assist with cyber crime reporting" in en_reply
    assert "1930" in en_reply
    assert CONTROL_ROOM in en_reply

    hi_reply = get_scope_abstention_reply(lang="hi")
    assert "Main keval cyber crime reporting" in hi_reply
    assert "1930" in hi_reply
    assert CONTROL_ROOM in hi_reply


# ==============================================================================
# 6. Safety-Critical Guardrails: High-Risk Claims Sanitization
# ==============================================================================

def test_sanitize_replaces_hallucinated_phone_number():
    """Hallucinated 10-digit mobile number claiming to be police/helpline is replaced with Alwar Control Room."""
    bad_reply = "You can call DSP Cyber Crime on personal number 9829012345 for assistance."
    clean, modified = sanitize_high_risk_claims(bad_reply, lang="en")
    
    assert modified is True
    assert "9829012345" not in clean
    assert CONTROL_ROOM in clean


def test_sanitize_preserves_verified_numbers():
    """Official authorized numbers (1930, 112, 1090, 8764874306, 0144-2337456) are preserved."""
    good_reply = f"Contact National Helpline 1930, emergency 112, or Alwar Cyber Control Room at {CONTROL_ROOM}."
    clean, modified = sanitize_high_risk_claims(good_reply, lang="en")
    
    assert modified is False
    assert clean == good_reply


def test_sanitize_preserves_utr_transaction_references():
    """10-digit transaction identifiers preceded by UTR or Ref are not mistakenly modified as phone numbers."""
    reply = "Your complaint has been noted with UTR 9876543210 for bank dispute tracking."
    clean, modified = sanitize_high_risk_claims(reply, lang="en")
    
    assert modified is False
    assert "9876543210" in clean


def test_sanitize_corrects_fabricated_filing_fee_assertion():
    """Fabricated complaint filing fee is corrected with statutory free-of-cost clarification."""
    bad_reply = "You must pay a registration fee of Rs 500 at the police station to file the FIR."
    clean, modified = sanitize_high_risk_claims(bad_reply, lang="en")
    
    assert modified is True
    assert "free of cost" in clean
    assert "No official fee is charged" in clean


def test_sanitize_does_not_duplicate_fee_disclaimer():
    """If the reply already clarifies no fee is charged, no redundant warning is appended."""
    good_reply = "There is no fee for filing a cyber crime complaint. It is completely free of cost."
    clean, modified = sanitize_high_risk_claims(good_reply, lang="en")
    
    assert modified is False
    assert clean == good_reply


def test_sanitize_corrects_fabricated_100_percent_refund_guarantee():
    """Fabricated 100% refund or fixed deadline claim is appended with golden hour dependency disclaimer."""
    bad_reply = "Once the complaint is lodged, you have a 100% refund guarantee within 24 hours."
    clean, modified = sanitize_high_risk_claims(bad_reply, lang="en")
    
    assert modified is True
    assert "cannot be guaranteed" in clean
    assert "golden hour" in clean


def test_sanitize_corrects_guaranteed_arrest_claim():
    """Fabricated promise of guaranteed arrest or bail denial is appended with judicial discretion disclaimer."""
    bad_reply = "Guaranteed arrest of the scammer will occur tomorrow and bail will be denied strictly."
    clean, modified = sanitize_high_risk_claims(bad_reply, lang="en")
    
    assert modified is True
    assert "subject to police investigation and court discretion" in clean


# ==============================================================================
# 7. Multilingual Abstention Expressions
# ==============================================================================

def test_multilingual_abstention_patterns():
    """Verify various forms of abstention across English, Devanagari Hindi, and Hinglish."""
    # English
    assert is_abstention_expressed("This detail cannot be verified from available official records.") is True
    assert is_abstention_expressed("Specific detail could not be verified.") is True
    assert is_abstention_expressed("There is no fixed deadline or refund guarantee.") is True
    assert is_abstention_expressed("Filing a complaint has no fee.") is True

    # Devanagari Hindi
    assert is_abstention_expressed("यह जानकारी आधिकारिक रिकॉर्ड में उपलब्ध नहीं है।") is True
    assert is_abstention_expressed("यह विवरण सत्यापित नहीं हो सका।") is True
    assert is_abstention_expressed("साइबर अपराध रिपोर्टिंग निःशुल्क है।") is True
    assert is_abstention_expressed("पैसे वापसी की गारंटी नहीं दी जा सकती।") is True

    # Hinglish
    assert is_abstention_expressed("Ye specific record official data me verify nahi ho saka.") is True
    assert is_abstention_expressed("Refund ki koi guarantee nahi hai, golden hour par depend karta hai.") is True
    assert is_abstention_expressed("Complaint file karne ki koi fee nahi lagti.") is True


# ==============================================================================
# 8. End-to-End AgentService.ask Abstention & Guardrail Flow
# ==============================================================================

def test_agent_service_ask_early_out_of_scope_abstention(monkeypatch):
    """End-to-end: Out-of-domain query triggers immediate scope abstention without invoking Gemini runner."""
    class FakeKB:
        def search(self, query, k=3, domains=None):
            return []

    monkeypatch.setattr("app.agent.runner.get_kb", lambda: FakeKB())

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    
    # Track whether model _run was called
    model_invoked = False

    async def fake_run(runner, session_id, message):
        nonlocal model_invoked
        model_invoked = True
        return AgentResult(reply="I love pizza!", tools_used=[], sources=[], artifacts=[])

    svc._run = fake_run

    res = asyncio.run(svc.ask("session-abc", "How do I make chocolate cake?", "en"))
    
    # Model should NEVER be called for clear out-of-domain queries
    assert model_invoked is False
    assert "I can only assist with cyber crime reporting" in res.reply
    assert "1930" in res.reply
    assert res.sources == []
    assert res.citation_note == ""


def test_agent_service_ask_high_risk_sanitization(monkeypatch):
    """End-to-end: If model hallucinates a fake phone number and 100% refund, post-sanitization catches both."""
    class FakeKB:
        def search(self, query, k=3, domains=None):
            return [SAMPLE_HITS[0]]

    monkeypatch.setattr("app.agent.runner.get_kb", lambda: FakeKB())

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))

    async def fake_run(runner, session_id, message):
        # Simulate model responding with grounded hit + fabricated mobile and refund guarantee
        return AgentResult(
            reply=(
                "Call 1930 within the golden hour. "
                "You can also call Inspector Verma at 9876512340. "
                "There is a 100% refund guarantee within 24 hours."
            ),
            tools_used=[],
            sources=[],
            artifacts=[],
        )

    svc._run = fake_run

    res = asyncio.run(svc.ask("session-xyz", "Help me with UPI fraud and inspector contact", "en"))
    
    # 1. Hallucinated mobile 9876512340 must be replaced with CONTROL_ROOM
    assert "9876512340" not in res.reply
    assert CONTROL_ROOM in res.reply
    # 2. Refund guarantee must be clarified
    assert "cannot be guaranteed" in res.reply
    # 3. Grounded source upi_fraud_immediate_actions.md must be attributed
    assert res.sources == ["upi_fraud_immediate_actions.md"]
    # 4. Citation note must acknowledge verification and limitation
    assert "Verified from official knowledge base: upi_fraud_immediate_actions.md" in res.citation_note
    assert "Additional specific details could not be verified" in res.citation_note
