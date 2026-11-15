"""Comprehensive regression tests for domain-aware RAG, query expansion,
Unicode tokenization, local Alwar mapping, and anti-hallucination grounding.
"""

import pytest

from app.agent import tools
from app.agent.facts import LOCALITY_TO_STATION, STATIONS
from app.services.query_expander import classify_domains, expand_query, tokenize
from app.services.rag import get_kb


def test_unicode_tokenizer_preserves_devanagari_and_acronyms():
    tokens = tokenize("मेरे खाते से 50,000 रुपये UPI कट गए! 1930 FIR")
    assert "upi" in tokens
    assert "1930" in tokens
    assert "fir" in tokens
    assert "50000" in tokens
    # Devanagari tokens preserved
    assert any("कट" in t or "खाते" in t for t in tokens)


def test_classify_domains_conversation():
    assert classify_domains("hi") == ["conversation"]
    assert classify_domains("Hello") == ["conversation"]
    assert classify_domains("Namaste") == ["conversation"]
    assert classify_domains("thanks a lot") == ["conversation"]


def test_classify_domains_deepfake_no_false_financial():
    # Extortion demand without financial loss should NOT classify as financial_fraud
    domains = classify_domains("Someone made a deepfake video and is demanding money.")
    assert "deepfake" in domains
    assert "financial_fraud" not in domains


def test_classify_domains_financial_loss():
    domains = classify_domains("Mere UPI se 50000 rupaye kat gaye.")
    assert "financial_fraud" in domains


def test_classify_domains_devanagari_fake_profile():
    domains = classify_domains("मेरे इंस्टाग्राम की फेक प्रोफाइल बनाई गई है")
    assert "social_media" in domains or "harassment" in domains


def test_classify_domains_malware_apk():
    domains = classify_domains("Someone installed a malicious APK on my phone.")
    assert "malware" in domains


def test_classify_domains_harassment_threat():
    domains = classify_domains("Kisi ne WhatsApp par dhamki di.")
    assert "harassment" in domains


def test_expand_query_hinglish():
    exp = expand_query("paise kat gaye mere UPI se")
    assert "unauthorized debit" in exp or "upi fraud" in exp


def test_expand_query_deepfake():
    exp = expand_query("kisine nakli video banayi deepfake")
    assert "morphed media" in exp or "non consensual" in exp


@pytest.fixture(scope="module")
def kb():
    k = get_kb()
    k.ensure_fresh()
    return k


def test_retrieval_test_1_deepfake_no_financial_contamination(kb):
    hits = kb.search("Someone made a deepfake video of me.")
    assert hits, "Expected retrieval hits for deepfake query"
    # Primary hit must be deepfake / morphed media
    assert hits[0]["domain"] in ("deepfake", "social_media")
    # Must NOT contain financial fraud / bank statement chunks in top results
    assert all(h["domain"] != "financial_fraud" for h in hits)
    assert not any("Bank statement" in h["text"] for h in hits)


def test_retrieval_test_2_upi_financial_fraud(kb):
    hits = kb.search("Mere UPI se 50000 rupaye kat gaye.")
    assert hits, "Expected retrieval hits for UPI fraud"
    assert hits[0]["domain"] == "financial_fraud"
    assert any("1930" in h["text"] for h in hits)


def test_retrieval_test_3_hindi_fake_profile(kb):
    hits = kb.search("मेरे इंस्टाग्राम की फेक प्रोफाइल बनाई गई है")
    assert hits, "Expected retrieval hits for fake profile query"
    sources = [h["source"] for h in hits]
    assert any("fake-profile" in s or "cyber-harassment" in s for s in sources)


def test_retrieval_test_4_malicious_apk(kb):
    hits = kb.search("Someone installed a malicious APK on my phone.")
    assert hits, "Expected retrieval hits for APK malware query"
    assert any(h["domain"] == "malware" for h in hits)
    assert any("Airplane Mode" in h["text"] or "AnyDesk" in h["text"] for h in hits)


def test_retrieval_test_5_harassment_threat(kb):
    hits = kb.search("Kisi ne WhatsApp par dhamki di.")
    assert hits, "Expected retrieval hits for harassment query"
    assert any(h["domain"] in ("harassment", "social_media", "local_police") for h in hits)


def test_retrieval_test_6_deepfake_demanding_money_not_bank_fraud(kb):
    hits = kb.search("Someone made a deepfake video and is demanding money.")
    assert hits, "Expected retrieval hits for deepfake extortion query"
    assert hits[0]["domain"] in ("deepfake", "social_media", "sextortion")
    assert all(h["domain"] != "financial_fraud" for h in hits)


def test_retrieval_test_7_conversation_returns_empty(kb):
    hits = kb.search("Hi")
    assert hits == []


def test_retrieval_test_8_off_topic_pizza_recipe_rejected(kb):
    hits = kb.search("What is the best pizza recipe?")
    assert hits == []


def test_local_alwar_police_station_mapping():
    # Test specific locality mapping
    r = tools.find_police_station("Kala Kuan")
    assert r["status"] == "ok"
    station_names = [s["name"] for s in r["stations"]]
    assert "Aravali Vihar Police Station" in station_names

    r2 = tools.find_police_station("Hope Circus")
    assert r2["status"] == "ok"
    assert any("Kotwali" in s["name"] for s in r2["stations"])

    r3 = tools.find_police_station("Shivaji Park")
    assert r3["status"] == "ok"
    assert any("Shivaji Park" in s["name"] for s in r3["stations"])


def test_mixed_complaint_letter_deepfake_extortion():
    r = tools.draft_complaint_letter(
        applicant_name="Sunita Devi",
        phone="9876543210",
        police_station="Mahila Thana Alwar",
        incident_category="Deepfake & Financial Extortion",
        incident_details="Someone morphed my photos into a deepfake video and demanded Rs 20000 on WhatsApp",
        incident_date="Yesterday night",
        language="en",
    )
    assert r["status"] == "ok"
    letter = r["letter"]
    assert "Sunita Devi" in letter
    assert "Mahila Thana Alwar" in letter
    # Enclosures must preserve media evidence and extortion details
    assert "deepfake video" in letter.lower()
    assert "extortion demand" in letter.lower()
