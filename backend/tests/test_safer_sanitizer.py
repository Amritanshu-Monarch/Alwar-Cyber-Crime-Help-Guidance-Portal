"""Unit tests for Phase 4: Safer Sanitizer and Entailment-based Grounding.

Tests:
1. Context-aware phone sanitization:
   - Preserves user-provided suspect phone numbers.
   - Preserves user-provided victim phone numbers.
   - Preserves numbers from retrieved chunks.
   - Preserves UTRs, Transaction IDs, Complaint IDs, Ack numbers.
   - Overwrites unverified numbers claiming to be official police contacts.
2. Tricky Hindi/Hinglish fee, refund, and judicial promise neutralizations.
3. Entailment-style citation and factual claim validation.
"""

from app.agent.abstention import (
    CONTROL_ROOM,
    VERIFIED_DOMAINS,
    VERIFIED_PHONE_NUMBERS,
    sanitize_high_risk_claims,
)
from app.agent.grounding import (
    attribute_evidence,
    verify_factual_claims_entailment,
)


# ==============================================================================
# 1. Phone Number & Identifier Preservation Tests
# ==============================================================================

def test_preserves_user_provided_suspect_number():
    """Suspect phone number provided by user must NOT be overwritten by CONTROL_ROOM."""
    user_msg = "Fraudster called me from 9876543210 asking for OTP."
    agent_reply = "I noted suspect number 9876543210. Immediately report this number on Chakshu and call 1930."
    sanitized, modified = sanitize_high_risk_claims(
        agent_reply,
        lang="en",
        user_message=user_msg,
        retrieved_context="",
    )
    assert "9876543210" in sanitized
    assert CONTROL_ROOM not in sanitized


def test_preserves_number_from_retrieved_context():
    """Phone numbers appearing in retrieved knowledge chunks must be preserved."""
    chunk_context = "Alwar special cyber helpline is 9123456789 for district residents."
    agent_reply = "You can contact the special desk at 9123456789 or 1930."
    sanitized, _ = sanitize_high_risk_claims(
        agent_reply,
        lang="en",
        user_message="Help me with special desk",
        retrieved_context=chunk_context,
    )
    assert "9123456789" in sanitized


def test_preserves_utr_transaction_and_complaint_ids():
    """10-digit or 12-digit numbers labelled as UTR, Txn ID, or Ack ID must never be overwritten."""
    agent_reply = "Keep your UTR 9876543210 and Transaction ID 8765432109 ready for the police report."
    sanitized, _ = sanitize_high_risk_claims(
        agent_reply,
        lang="en",
        user_message="I lost money",
        retrieved_context="",
    )
    assert "9876543210" in sanitized
    assert "8765432109" in sanitized


def test_replaces_hallucinated_official_officer_contact():
    """An unverified number in the context of calling police or an officer must be sanitized."""
    agent_reply = "Please call SHO Alwar Kotwali at 9876543210 to lodge your complaint."
    sanitized, modified = sanitize_high_risk_claims(
        agent_reply,
        lang="en",
        user_message="Who is the SHO?",
        retrieved_context="",
    )
    assert modified is True
    assert "9876543210" not in sanitized
    assert CONTROL_ROOM in sanitized


def test_replaces_hallucinated_officer_contact_in_hindi():
    """Hindi response hallucinating an officer phone number is sanitized to CONTROL_ROOM."""
    agent_reply = "थानाधिकारी से संपर्क करने के लिए 9876543210 पर कॉल करें।"
    sanitized, modified = sanitize_high_risk_claims(
        agent_reply,
        lang="hi",
        user_message="थानाधिकारी का नंबर क्या है?",
        retrieved_context="",
    )
    assert modified is True
    assert "9876543210" not in sanitized
    assert CONTROL_ROOM in sanitized


# ==============================================================================
# 2. Fee Assertion Neutralization Tests (English, Hindi, Hinglish)
# ==============================================================================

def test_neutralizes_tricky_hinglish_fee_claims():
    """Hinglish phrasings asserting filing fees receive mandatory free-of-cost correction."""
    phrasings = [
        "Shikayat darj karane ke 500 rs charge lagega portal par.",
        "Aapko registration fee of rs 200 pay karna hoga.",
        "FIR darj karne ka kharcha aayega thoda sa.",
    ]
    for text in phrasings:
        sanitized, modified = sanitize_high_risk_claims(text, lang="hinglish")
        assert modified is True
        assert "free of cost" in sanitized.lower() or "koi official fee nahi" in sanitized.lower()


def test_neutralizes_tricky_hindi_fee_claims():
    """Hindi phrasings asserting fees receive statutory free-of-cost correction."""
    phrasings = [
        "शिकायत दर्ज कराने के लिए 500 रुपये जमा करने होंगे।",
        "सरकारी शुल्क लगेगा पोर्टल पर पंजीकरण के लिए।",
        "एफआईआर दर्ज कराने के लिए पैसे लगेंगे।",
    ]
    for text in phrasings:
        sanitized, modified = sanitize_high_risk_claims(text, lang="hi")
        assert modified is True
        assert "निःशुल्क" in sanitized or "कोई आधिकारिक शुल्क नहीं" in sanitized


def test_does_not_double_correct_legitimate_no_fee_statements():
    """Statements correctly affirming that filing is free must not receive duplicate corrections."""
    reply = "Cyber crime reporting on cybercrime.gov.in has no fee, it is completely free of cost."
    sanitized, modified = sanitize_high_risk_claims(reply, lang="en")
    assert modified is False
    assert sanitized == reply


# ==============================================================================
# 3. Refund Guarantee & Turnaround Neutralization Tests
# ==============================================================================

def test_neutralizes_tricky_hinglish_refund_guarantees():
    """Hinglish phrasings asserting 100% refund guarantee receive disclaimer."""
    phrasings = [
        "Aapko 24 ghante me refund pakka ho jayega.",
        "Guaranteed paisa wapas milenge 100%.",
        "Poore paise wapas honge guarantee ke sath.",
    ]
    for text in phrasings:
        sanitized, modified = sanitize_high_risk_claims(text, lang="hinglish")
        assert modified is True
        assert "no fixed time guarantee" in sanitized.lower() or "koi fixed time guarantee nahi hoti" in sanitized.lower()


def test_neutralizes_tricky_hindi_refund_guarantees():
    """Hindi phrasings asserting refund guarantees receive golden hour disclaimer."""
    phrasings = [
        "24 घंटे में पैसे वापस मिल जाएंगे।",
        "पैसे वापसी की पक्की गारंटी है।",
        "पूरी रकम वापस होगी 100% गारंटी के साथ।",
    ]
    for text in phrasings:
        sanitized, modified = sanitize_high_risk_claims(text, lang="hi")
        assert modified is True
        assert "कोई निश्चित समय-सीमा या 100% गारंटी नहीं होती" in sanitized


# ==============================================================================
# 4. Judicial Promise Neutralization Tests
# ==============================================================================

def test_neutralizes_judicial_and_arrest_promises():
    """Statements guaranteeing arrest or bail rejection receive police/court discretion disclaimer."""
    phrasings = [
        ("Accused ko turant arrest kar liya jayega pakka.", "hinglish", "faisla police investigation"),
        ("पक्की गिरफ्तारी होगी और जमानत नहीं मिलेगी यह निश्चित है।", "hi", "न्यायालय के विवेक"),
        ("The accused will definitely be jailed and bail will be denied strictly.", "en", "court discretion"),
    ]
    for text, lang, expected_token in phrasings:
        sanitized, modified = sanitize_high_risk_claims(text, lang=lang)
        assert modified is True
        assert expected_token.lower() in sanitized.lower()


# ==============================================================================
# 5. Entailment-style Factual Claim Verification Tests
# ==============================================================================

def test_entailment_accepts_verified_numbers_and_chunks():
    """Phone numbers in verified contact list or retrieved chunks pass entailment."""
    hits = [{"source": "special.yaml", "text": "Contact desk at 9123456789 for help."}]
    reply = f"Call National Helpline 1930 or Alwar Cyber Control Room {CONTROL_ROOM} or 9123456789."
    valid, unverified = verify_factual_claims_entailment(reply, hits)
    assert valid is True
    assert unverified == []


def test_entailment_flags_hallucinated_phone_and_portal():
    """Fake phone numbers and fake portals are caught by entailment check."""
    hits = [{"source": "upi.yaml", "text": "Report on cybercrime.gov.in immediately."}]
    reply = "Contact officer at 9988776655 or visit fake-cyber-alwar.com to register."
    valid, unverified = verify_factual_claims_entailment(reply, hits)
    assert valid is False
    assert any("phone:9988776655" in u for u in unverified)
    assert any("portal:fake-cyber-alwar.com" in u for u in unverified)


def test_entailment_flags_unverified_fee():
    """A positive fee claim is flagged as unverified unless stated as free."""
    hits = [{"source": "upi.yaml", "text": "Report on cybercrime.gov.in immediately."}]
    reply = "You must pay registration charge of rs 500."
    valid, unverified = verify_factual_claims_entailment(reply, hits)
    assert valid is False
    assert "unverified_fee" in unverified
