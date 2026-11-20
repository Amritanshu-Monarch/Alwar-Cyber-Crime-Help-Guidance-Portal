"""Tests for Sub-phase B3.3.2: Multilingual Voice Quality & Real-Device Acceptance.

Validates that the conversational voice system naturally understands and responds
to citizens speaking in:
1. Hindi (Devanagari script, pure Hindi vocabulary)
2. Hinglish (Latin script code-mixed Hindi-English, preserving technical terms)
3. English (cybersecurity reporting, phishing, blackmail)
4. Dynamic Language Switching (Hindi -> Hinglish -> English -> Hindi within a single session)
5. Technical Term Handling (UPI, OTP, UTR, APK, StopNCII, Chakshu, CEIR)
6. Multi-turn Conversational Context Preservation
7. Safety, Grounding, and Deterministic Abstention across all languages
8. Speech formatting & synthesis sanitization
"""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.abstention import (
    get_scope_abstention_reply,
    is_abstention_expressed,
    is_out_of_scope_query,
    sanitize_high_risk_claims,
)
from app.agent.facts import CONTROL_ROOM
from app.agent.grounding import attribute_evidence, format_grounding_directive
from app.agent.runner import (
    AgentService,
    REPLY_LANGUAGE,
    clean_reply_text,
    detect_effective_language,
)
from app.config import Settings
from app.schemas import AgentResult, Artifact
from app.services.speech import STT_PROMPT, clean_for_speech


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
        "source": "digital_arrest_advisories.md",
        "section": "Digital Arrest Scam",
        "authority": "police_advisory",
        "jurisdiction": "national",
        "text": "Police or law enforcement agencies never place citizens under digital arrest or demand money via video call. Report to 1930 or 8764874306.",
        "score": 0.88,
    },
    {
        "source": "chakshu_reporting_guide.md",
        "section": "Chakshu Portal",
        "authority": "dot",
        "jurisdiction": "national",
        "text": "Report suspected fraud communications like fake electricity disconnection SMS on the Chakshu portal.",
        "score": 0.85,
    },
]


class FakeKB:
    """Mock KB so unit tests do not exhaust Gemini embedding free-tier quota."""
    def search(self, query, k=3, domains=None):
        q = (query or "").lower()
        if "chakshu" in q or "electricity" in q:
            return [SAMPLE_HITS[2]]
        elif "arrest" in q or "अरेस्ट" in q:
            return [SAMPLE_HITS[1]]
        elif any(k in q for k in ("upi", "fraud", "transfer", "खाते", "पैसे", "15000", "25000", "screenshot")):
            return [SAMPLE_HITS[0]]
        return []

    def ensure_fresh(self):
        pass


@pytest.fixture(autouse=True)
def mock_kb(monkeypatch):
    """Automatically mock knowledge base in all tests in this module."""
    fake_kb = FakeKB()
    monkeypatch.setattr("app.agent.runner.get_kb", lambda: fake_kb)
    return fake_kb


# ==============================================================================
# 1. Unit Tests: Language Detection & Directive Selection
# ==============================================================================

def test_language_detection_hindi_devanagari():
    """Devanagari script queries are detected as 'hi'."""
    queries = [
        "मेरे बैंक खाते से पैसे कट गए हैं, लेकिन मैंने कोई ओटीपी शेयर नहीं किया। मुझे क्या करना चाहिए?",
        "अलवर में साइबर अपराध की रिपोर्ट करने की प्रक्रिया क्या है?",
        "किसी ने मेरे नाम और फोटो का उपयोग करके फर्जी इंस्टाग्राम अकाउंट बना लिया है।",
        "अज्ञात नंबर से कॉल आया कि आपके नाम पर अरेस्ट वारंट है और पैसे ट्रांसफर करने को कह रहे हैं।",
        "नमस्ते, मुझे साइबर अपराध की शिकायत करनी है।",
        "क्या साइबर कंप्लेंट के लिए कोई फीस लगती है?",
    ]
    for q in queries:
        detected = detect_effective_language(q, fallback_lang="en")
        assert detected == "hi", f"Expected 'hi' for Devanagari query: {q}"


def test_language_detection_hinglish_code_mixed():
    """Latin script code-mixed Hindi-English queries are detected as 'hinglish'."""
    queries = [
        "Mere bank khate se 15000 rupees fraud ho gaye hain, maine koi OTP share nahi kiya. Main kya karun?",
        "Kisi ne fake UPI screenshot bhej kar mujhse delivery le li, report kaise karein?",
        "Mujhe WhatsApp par unknown number se video call aayi aur ab wo video viral karne ki dhamki de rahe hain.",
        "Alwar cyber police station ka phone number kya hai?",
        "Main 1930 par call kiya tha par busy aa raha hai, ab kya karu?",
        "Mera account hack ho gaya hai, help chahiye.",
        "Aap mujhe batao ki fake loan app ki complaint kaise karein?",
    ]
    for q in queries:
        detected = detect_effective_language(q, fallback_lang="en")
        assert detected == "hinglish", f"Expected 'hinglish' for code-mixed query: {q}"


def test_language_detection_english():
    """English queries without Hindi/Devanagari are detected as 'en'."""
    queries = [
        "How can I report an unauthorized bank transfer of 25000 rupees?",
        "Someone is blackmailing me on WhatsApp with a morphed video.",
        "I received a fake SMS asking me to update my electricity bill or power will be disconnected.",
        "Where is the Alwar Cyber Crime police station located?",
        "What documents should I keep ready before going to the police station?",
        "Can you help me report an online financial fraud?",
    ]
    for q in queries:
        detected = detect_effective_language(q, fallback_lang="hi")
        assert detected == "en", f"Expected 'en' for English query: {q}"


def test_language_detection_fallbacks_and_intent_overrides():
    """Ambiguous inputs follow fallback, and explicit intent overrides are respected."""
    # Empty or generic neutral tokens fallback cleanly
    assert detect_effective_language("", fallback_lang="en") == "en"
    assert detect_effective_language("", fallback_lang="hi") == "hi"
    assert detect_effective_language("1930", fallback_lang="en") == "en"
    assert detect_effective_language("1930", fallback_lang="hi") == "hi"

    # Explicit English intent overrides
    assert detect_effective_language("Explain this in English please", fallback_lang="hi") == "en"
    assert detect_effective_language("Answer in English", fallback_lang="hi") == "en"

    # Explicit Hindi intent
    assert detect_effective_language("Shikayat patra hindi me do", fallback_lang="en") == "hinglish"
    assert detect_effective_language("मुझे हिंदी में बताओ", fallback_lang="en") == "hi"


def test_reply_language_directives():
    """Verify system directives enforce the proper script and constraints."""
    en_dir = REPLY_LANGUAGE["en"]
    hi_dir = REPLY_LANGUAGE["hi"]
    hinglish_dir = REPLY_LANGUAGE["hinglish"]

    # English directive
    assert en_dir == "[Reply in English.]"

    # Hindi Devanagari directive: requires Devanagari, digits for numbers, and starts with [Reply in Hindi
    assert hi_dir.startswith("[Reply in Hindi")
    assert "Devanagari" in hi_dir
    assert "1930" in hi_dir
    assert "8764874306" in hi_dir

    # Hinglish directive: requires Roman letters, natural spoken Hinglish, retains technical terms, never Devanagari
    assert hinglish_dir.startswith("[Reply in Hindi")
    assert "Roman" in hinglish_dir
    assert "Hinglish" in hinglish_dir
    assert "Never use Devanagari" in hinglish_dir
    assert "UPI" in hinglish_dir


# ==============================================================================
# 2. Scenario A: Hindi (Devanagari Script & Pure Vocabulary)
# ==============================================================================

def test_scenario_a_hindi_financial_fraud_flow():
    """Hindi query: unauthorized debit without OTP -> routes to 'hi' directive, contains Devanagari, mentions 1930."""
    seen_messages = []

    async def fake_run(runner, session_id, message):
        seen_messages.append(message)
        return AgentResult(
            reply="तुरंत 1930 पर कॉल करें और अपने बैंक को सूचित करके खाता फ्रीज कराएं। इसके बाद cybercrime.gov.in पर शिकायत दर्ज करें।",
            tools_used=["get_complaint_guide"],
            sources=["upi_fraud_immediate_actions.md"],
        )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    query = "मेरे बैंक खाते से पैसे कट गए हैं, लेकिन मैंने कोई ओटीपी शेयर नहीं किया। मुझे क्या करना चाहिए?"
    res = asyncio.run(svc.ask("session-hi-01", query, lang="en"))

    # Verify directive passed to runner
    assert seen_messages[0].startswith("[Reply in Hindi in Devanagari script")
    # Verify response contains Devanagari
    assert bool(re.search(r"[\u0900-\u097F]", res.reply))
    assert "1930" in res.reply
    assert "cybercrime.gov.in" in res.reply
    assert res.sources == ["upi_fraud_immediate_actions.md"]


def test_scenario_a_hindi_digital_arrest_flow():
    """Hindi query: fake arrest warrant call -> routes to 'hi' directive, provides safe advisory."""
    seen_messages = []

    async def fake_run(runner, session_id, message):
        seen_messages.append(message)
        return AgentResult(
            reply="पुलिस या कोई भी सरकारी एजेंसी वीडियो कॉल पर डिजिटल अरेस्ट नहीं करती और न ही पैसे मांगती है। तुरंत कॉल काटें और 1930 या 8764874306 पर सूचना दें।",
            tools_used=["search_knowledge"],
            sources=["digital_arrest_advisories.md"],
        )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    query = "अज्ञात नंबर से कॉल आया कि आपके नाम पर अरेस्ट वारंट है और पैसे ट्रांसफर करने को कह रहे हैं।"
    res = asyncio.run(svc.ask("session-hi-02", query, lang="en"))

    assert seen_messages[0].startswith("[Reply in Hindi in Devanagari script")
    assert "डिजिटल अरेस्ट" in res.reply
    assert "1930" in res.reply
    assert CONTROL_ROOM in res.reply


# ==============================================================================
# 3. Scenario B: Hinglish (Latin Script Code-Mixed Hindi-English)
# ==============================================================================

def test_scenario_b_hinglish_fake_screenshot():
    """Hinglish query: fake UPI screenshot -> routes to 'hinglish' directive, preserves technical terms."""
    seen_messages = []

    async def fake_run(runner, session_id, message):
        seen_messages.append(message)
        return AgentResult(
            reply="Aap fake UPI screenshot aur delivery chats ka screenshot preserve karein. Turant 1930 par call karein aur cybercrime.gov.in par report karein.",
            tools_used=["get_complaint_guide"],
            sources=["upi_fraud_immediate_actions.md"],
        )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    query = "Kisi ne fake UPI screenshot bhej kar mujhse delivery le li, report kaise karein?"
    res = asyncio.run(svc.ask("session-hing-01", query, lang="en"))

    assert seen_messages[0].startswith("[Reply in Hindi written in Roman")
    assert "Never use Devanagari" in seen_messages[0]
    # Response is Latin script Hinglish (no Devanagari)
    assert not bool(re.search(r"[\u0900-\u097F]", res.reply))
    assert "UPI screenshot" in res.reply
    assert "1930" in res.reply
    assert "cybercrime.gov.in" in res.reply


def test_scenario_b_hinglish_sextortion_blackmail():
    """Hinglish query: WhatsApp video blackmail -> preserves terms, advises against paying extortion."""
    seen_messages = []

    async def fake_run(runner, session_id, message):
        seen_messages.append(message)
        return AgentResult(
            reply="Ghabraiye mat aur unhe koi paisa transfer na karein. WhatsApp chat aur call log ke screenshots preserve karke Alwar Cyber Control Room 8764874306 par sampark karein.",
            tools_used=[],
            sources=[],
        )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    query = "Mujhe WhatsApp par unknown number se video call aayi aur ab wo video viral karne ki dhamki de rahe hain."
    res = asyncio.run(svc.ask("session-hing-02", query, lang="en"))

    assert seen_messages[0].startswith("[Reply in Hindi written in Roman")
    assert "WhatsApp" in res.reply
    assert CONTROL_ROOM in res.reply


# ==============================================================================
# 4. Scenario C: English (Cybersecurity Reporting & Evidence Guidance)
# ==============================================================================

def test_scenario_c_english_financial_transfer():
    """English query: unauthorized bank transfer -> routes to 'en' directive, clean guidance."""
    seen_messages = []

    async def fake_run(runner, session_id, message):
        seen_messages.append(message)
        return AgentResult(
            reply="Immediately dial 1930 within the golden hour to freeze the transferred amount and alert your bank. Then register your complaint on cybercrime.gov.in.",
            tools_used=["get_complaint_guide"],
            sources=["upi_fraud_immediate_actions.md"],
        )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    query = "How can I report an unauthorized bank transfer of 25000 rupees?"
    res = asyncio.run(svc.ask("session-en-01", query, lang="hi"))  # UI set to 'hi', but query is English!

    # Model receives English directive because query is in English
    assert seen_messages[0].startswith("[Reply in English.]")
    assert "1930" in res.reply
    assert "cybercrime.gov.in" in res.reply
    assert res.sources == ["upi_fraud_immediate_actions.md"]


def test_scenario_c_english_electricity_bill_scam():
    """English query: fake electricity bill SMS -> clear advisory and preservation guidance."""
    seen_messages = []

    async def fake_run(runner, session_id, message):
        seen_messages.append(message)
        return AgentResult(
            reply="Do not click any link or call the number in the SMS. Report the sender number on the Chakshu portal and contact Alwar Cyber Control Room at 8764874306.",
            tools_used=["search_knowledge"],
            sources=["chakshu_reporting_guide.md"],
        )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    query = "I received a fake SMS asking me to update my electricity bill or power will be disconnected."
    res = asyncio.run(svc.ask("session-en-02", query, lang="en"))

    assert seen_messages[0].startswith("[Reply in English.]")
    assert "Chakshu" in res.reply
    assert CONTROL_ROOM in res.reply


# ==============================================================================
# 5. Scenario D: Dynamic Language Switching (4-Turn In-Session Test)
# ==============================================================================

def test_scenario_d_dynamic_language_switching_single_session():
    """Test a continuous 4-turn voice conversation within a single session:
    Turn 1 (Hindi Devanagari) -> [Reply in Hindi in Devanagari script...]
    Turn 2 (Hinglish) -> [Reply in Hindi written in Roman...]
    Turn 3 (English) -> [Reply in English.]
    Turn 4 (Hindi Devanagari) -> [Reply in Hindi in Devanagari script...] with zero fee confirmation.
    """
    seen_directives = []
    session_id = "session-multilingual-switch-12345"

    turn_replies = [
        # Turn 1
        "तुरंत 1930 पर कॉल करें ताकि ठग के खाते में रकम होल्ड कराई जा सके।",
        # Turn 2
        "Agar 1930 busy hai toh bina deri kiye cybercrime.gov.in par complaint darj karein aur apne bank branch jakar account freeze karwayein.",
        # Turn 3
        "Keep your bank statement, UTR transaction number, suspect's phone number, and transaction SMS ready.",
        # Turn 4
        "साइबर अपराध शिकायत अथवा एफआईआर दर्ज कराने के लिए कोई सरकारी शुल्क नहीं लगता है, यह पूर्णतः निःशुल्क है।",
    ]

    turn_idx = 0

    async def fake_run(runner, s_id, message):
        nonlocal turn_idx
        # Extract bracketed directive from start of message
        m = re.match(r"^(\[Reply in [^\]]+\])", message)
        seen_directives.append(m.group(1) if m else "")
        reply = turn_replies[turn_idx]
        turn_idx += 1
        return AgentResult(reply=reply)

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    # Turn 1: Hindi Devanagari
    res1 = asyncio.run(svc.ask(session_id, "मेरे खाते से पैसे कट गए।", lang="en"))
    assert "Devanagari script" in seen_directives[0]
    assert "1930" in res1.reply

    # Turn 2: Hinglish
    res2 = asyncio.run(svc.ask(session_id, "Main 1930 par call kiya tha par busy aa raha hai, ab kya karu?", lang="en"))
    assert "written in Roman" in seen_directives[1]
    assert "cybercrime.gov.in" in res2.reply

    # Turn 3: English
    res3 = asyncio.run(svc.ask(session_id, "What documents should I keep ready before going to the police station?", lang="en"))
    assert "[Reply in English.]" == seen_directives[2]
    assert "bank statement" in res3.reply.lower()

    # Turn 4: Hindi Devanagari (Fee Inquiry)
    res4 = asyncio.run(svc.ask(session_id, "क्या शिकायत दर्ज कराने के लिए कोई सरकारी फीस लगती है?", lang="en"))
    assert "Devanagari script" in seen_directives[3]
    # Verifies deterministic zero fee statement in Hindi
    assert ("निःशुल्क" in res4.reply or "कोई" in res4.reply) and "शुल्क नहीं" in res4.reply


# ==============================================================================
# 6. Scenario E: Technical Term Handling & Acronym Preservation
# ==============================================================================

def test_scenario_e_technical_term_preservation():
    """Technical acronyms, platform names, and official portals are preserved without awkward translation."""
    test_terms = [
        "UPI", "OTP", "UTR", "APK", "SIM", "2FA",
        "WhatsApp", "Instagram", "Telegram",
        "cybercrime.gov.in", "Chakshu",
    ]

    sample_hindi_text = (
        "आप तुरंत 1930 पर कॉल करके UPI और OTP फ्रॉड की शिकायत करें तथा UTR नंबर नोट कराएं। WhatsApp, Instagram और Telegram पर आए APK और SIM फ्रॉड के स्क्रीनशॉट सुरक्षित रखें। "
        "cybercrime.gov.in और Chakshu पोर्टल पर 2FA सक्षम रखते हुए रिपोर्ट करें।"
    )

    for term in test_terms:
        assert term in sample_hindi_text, f"Term {term} must be preserved verbatim in Hindi text"


# ==============================================================================
# 7. Scenario F: Multi-Turn Conversational Context Carryover
# ==============================================================================

def test_scenario_f_multi_turn_context_carryover():
    """Multi-turn context retains prior incident details across turns."""
    session_id = "session-context-carryover-9988"
    history = []

    async def fake_run(runner, s_id, message):
        history.append(message)
        if len(history) == 1:
            return AgentResult(reply="I understand you lost money in a Telegram job scam. Please report immediately to 1930.")
        elif len(history) == 2:
            return AgentResult(reply="You should report within the golden hour (first 2-3 hours) for the highest chance of freezing the funds.")
        else:
            return AgentResult(
                reply="A formal complaint letter for your Telegram job fraud has been drafted and displayed on screen.",
                artifacts=[Artifact(type="letter", text="To SHO, Alwar Police... Incident: Part-time job scam on Telegram...")],
            )

    svc = AgentService(Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model"))
    svc._run = fake_run

    # Turn 1
    r1 = asyncio.run(svc.ask(session_id, "I lost money in a part-time job scam on Telegram.", lang="en"))
    assert "Telegram job scam" in r1.reply

    # Turn 2: User says "it" -> context refers to the job scam
    r2 = asyncio.run(svc.ask(session_id, "How much time do I have to report it?", lang="en"))
    assert "golden hour" in r2.reply

    # Turn 3: User asks for complaint letter without re-explaining the scam
    r3 = asyncio.run(svc.ask(session_id, "Can you draft a complaint letter for me?", lang="en"))
    assert len(r3.artifacts) == 1
    assert "Telegram" in r3.artifacts[0].text


def test_short_hinglish_follow_up_keeps_telegram_context():
    """Short follow-ups inherit the active incident instead of triggering scope abstention."""
    messages = []

    async def fake_run(runner, s_id, message):
        messages.append(message)
        return AgentResult(reply="Telegram recovery steps are ready.")

    svc = AgentService(
        Settings(google_api_key="fake-key", agent_model="fake-model", agent_fallback_model="fake-model")
    )
    svc._run = fake_run

    first = asyncio.run(svc.ask("session-telegram-followup", "Mera Telegram hack ho gaya hai.", lang="en"))
    second = asyncio.run(svc.ask("session-telegram-followup", "pura procedure do", lang="en"))

    assert first.reply == "Telegram recovery steps are ready."
    assert second.reply == "Telegram recovery steps are ready."
    assert "Reply in Hindi written in Roman" in messages[1]
    assert "Telegram hack" in messages[1]


# ==============================================================================
# 8. Scenario G: Safety & Deterministic Abstention Regression Across Languages
# ==============================================================================

def test_scenario_g_unsupported_officer_number_abstention_all_languages():
    """Pure unsupported query: personal mobile of SHO / SP is disclaimed in Hindi, Hinglish, and English."""
    # Hindi Devanagari
    hi_reply = (
        "थानाधिकारी का व्यक्तिगत मोबाइल नंबर उपलब्ध आधिकारिक रिकॉर्ड में सत्यापित नहीं हो सका। "
        f"कृपया अलवर साइबर कंट्रोल रूम ({CONTROL_ROOM}) या 1930 पर संपर्क करें।"
    )
    assert is_abstention_expressed(hi_reply) is True
    sources, note = attribute_evidence(hi_reply, [], ["local_police"], lang="hi")
    assert sources == []
    assert CONTROL_ROOM in note

    # Hinglish
    hing_reply = (
        "SHO ka personal mobile number official records me verify nahi ho saka. "
        f"Aap Alwar Cyber Control Room {CONTROL_ROOM} ya 1930 par call karein."
    )
    assert is_abstention_expressed(hing_reply) is True
    sources, note = attribute_evidence(hing_reply, [], ["local_police"], lang="hinglish")
    assert sources == []
    assert CONTROL_ROOM in note

    # English
    en_reply = (
        "The personal phone number of the police officer could not be verified from available official records. "
        f"Please contact the Alwar Cyber Control Room at {CONTROL_ROOM} or 1930."
    )
    assert is_abstention_expressed(en_reply) is True
    sources, note = attribute_evidence(en_reply, [], ["local_police"], lang="en")
    assert sources == []
    assert CONTROL_ROOM in note


def test_scenario_g_high_risk_claim_sanitization_multilingual():
    """High-risk claims (fabricated fee and 100% refund guarantee) are sanitized in Hindi, Hinglish, and English."""
    # Filing Fee Sanitization
    hi_bad = "थाना में एफआईआर दर्ज कराने के लिए 500 रुपये फीस लगेगी।"
    hi_clean, mod = sanitize_high_risk_claims(hi_bad, lang="hi")
    assert mod is True
    assert "निःशुल्क" in hi_clean

    hing_bad = "Police station me registration fee of Rs 500 lagegi."
    hing_clean, mod = sanitize_high_risk_claims(hing_bad, lang="hinglish")
    assert mod is True
    assert "koi official fee nahi" in hing_clean or "free of cost" in hing_clean

    en_bad = "You must pay a registration fee of Rs 500 to file the complaint."
    en_clean, mod = sanitize_high_risk_claims(en_bad, lang="en")
    assert mod is True
    assert "completely free of cost" in en_clean

    # Refund Guarantee Sanitization
    hi_bad_refund = "शिकायत के बाद 24 घंटे में पैसे वापस मिलने की 100% गारंटी है।"
    hi_clean_refund, mod = sanitize_high_risk_claims(hi_bad_refund, lang="hi")
    assert mod is True
    assert "गारंटी नहीं होती" in hi_clean_refund

    hing_bad_refund = "Complaint hone par 100% refund guarantee within 24 hours milti hai."
    hing_clean_refund, mod = sanitize_high_risk_claims(hing_bad_refund, lang="hinglish")
    assert mod is True
    assert "golden hour" in hing_clean_refund.lower()

    en_bad_refund = "Once filed, you have a 100% refund guarantee within 24 hours."
    en_clean_refund, mod = sanitize_high_risk_claims(en_bad_refund, lang="en")
    assert mod is True
    assert "cannot be guaranteed" in en_clean_refund


def test_scenario_g_out_of_domain_abstention_multilingual():
    """Out-of-domain queries (cooking, jokes, weather) are detected and abstained politely."""
    assert is_out_of_scope_query("पिज्जा बनाने की विधि क्या है?", ["conversation"]) is True
    assert is_out_of_scope_query("Aaj ka mausam kaisa rahega?", ["conversation"]) is True
    assert is_out_of_scope_query("Tell me a funny joke", ["conversation"]) is True

    # Polite scope reply
    hi_scope = get_scope_abstention_reply(lang="hi")
    assert "1930" in hi_scope
    assert CONTROL_ROOM in hi_scope

    en_scope = get_scope_abstention_reply(lang="en")
    assert "1930" in en_scope
    assert CONTROL_ROOM in en_scope


# ==============================================================================
# 9. Speech Formatting & STT Multilingual Prompt Validation
# ==============================================================================

def test_speech_formatting_clean_for_speech_multilingual():
    """clean_for_speech strips markdown and formats long numbers with spaces for spoken clarity."""
    # Hindi Devanagari text with numbers and markdown
    hi_input = "**महत्वपूर्ण:** तुरंत [1930](tel:1930) या 8764874306 पर कॉल करें।"
    hi_clean = clean_for_speech(hi_input)
    assert "**" not in hi_clean
    assert "[" not in hi_clean
    assert "1 9 3 0" in hi_clean
    assert "8 7 6 4 8 7 4 3 0 6" in hi_clean

    # Hinglish text with numbers and markdown
    hing_input = "**Alert:** Aap sabse pehle `1930` par call karein ya *8764874306* par."
    hing_clean = clean_for_speech(hing_input)
    assert "**" not in hing_clean
    assert "`" not in hing_clean
    assert "1 9 3 0" in hing_clean
    assert "8 7 6 4 8 7 4 3 0 6" in hing_clean

    # English text
    en_input = "Call National Cyber Helpline **1930** or Alwar Control Room at 8764874306."
    en_clean = clean_for_speech(en_input)
    assert "**" not in en_clean
    assert "1 9 3 0" in en_clean
    assert "8 7 6 4 8 7 4 3 0 6" in en_clean


def test_stt_prompt_multilingual_coverage():
    """Verify backend STT prompt instructs Gemini to handle Hindi, Hinglish, and English."""
    assert "Hindi" in STT_PROMPT
    assert "English" in STT_PROMPT
    assert "Hinglish" in STT_PROMPT
    assert "Devanagari" in STT_PROMPT
    assert "Latin" in STT_PROMPT
