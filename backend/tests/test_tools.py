from app.agent import tools
from app.agent.facts import STATIONS

FULL = dict(
    applicant_name="Ramesh Sharma",
    phone="9829000000",
    police_station="Kotwali Police Station",
    incident_category="Job / Loan Scam",
    incident_details="Lost Rs 5000 via UPI to caller 98xxxxxx",
    incident_date="12 Oct 2026, 4pm",
)


def test_letter_english_contains_all_fields():
    r = tools.draft_complaint_letter(**FULL, language="en")
    assert r["status"] == "ok"
    for needle in ("Ramesh Sharma", "9829000000", "Kotwali Police Station", "Job / Loan Scam", "Rs 5000", "12 Oct 2026"):
        assert needle in r["letter"]


def test_letter_hindi():
    r = tools.draft_complaint_letter(**FULL, language="hi")
    assert r["status"] == "ok"
    assert "थानाधिकारी" in r["letter"]


def test_letter_rejects_missing_details():
    r = tools.draft_complaint_letter(**{**FULL, "incident_details": "  "}, language="en")
    assert r["status"] == "error"


def test_letter_rejects_bad_language():
    assert tools.draft_complaint_letter(**FULL, language="fr")["status"] == "error"


def test_letter_unknown_category_falls_back_to_other():
    r = tools.draft_complaint_letter(**{**FULL, "incident_category": "<script>"}, language="en")
    assert "<script>" not in r["letter"] and "Other" in r["letter"]


def test_stations_all_and_filtered():
    assert len(tools.find_police_station("")["stations"]) == len(STATIONS)
    r = tools.find_police_station("kotwali")
    assert [s["name"] for s in r["stations"]] == ["Kotwali Police Station"]
    assert r["stations"][0]["map_url"].startswith("https://maps.google.com/?q=")
    assert tools.find_police_station("zzz")["status"] == "no_results"


def test_every_guide_renders_in_both_languages():
    from app.agent.guides import GUIDES

    for key in GUIDES:
        for lang in ("en", "hi"):
            r = tools.get_complaint_guide(key, lang)
            assert r["status"] == "ok", (key, lang)
            assert r["title"] in r["guide_text"]
            assert "1." in r["guide_text"] and "•" in r["guide_text"]


def test_guide_financial_fraud_puts_1930_first():
    text = tools.get_complaint_guide("financial_fraud", "en")["guide_text"]
    assert text.index("Call 1930") < text.index("cybercrime.gov.in")
    assert "UTR" in text


def test_guide_unknown_type_lists_valid_options_and_bad_lang_falls_back():
    r = tools.get_complaint_guide("nonsense", "en")
    assert r["status"] == "error" and "financial_fraud" in r["message"]
    assert tools.get_complaint_guide("financial_fraud", "xx")["status"] == "ok"


def test_search_no_results_tells_agent_not_to_answer_from_memory(monkeypatch):
    class Empty:
        def search(self, q):
            return []

    monkeypatch.setattr(tools, "get_kb", lambda: Empty())
    r = tools.search_knowledge("write python code")
    assert r["status"] == "no_results" and "do not answer" in r["message"]


def test_contacts_have_core_numbers():
    c = tools.get_contacts()["contacts"]
    assert c["national_cyber_helpline"]["phone"] == "1930"
    assert c["alwar_cyber_control_room"]["phone"] == "8764874306"


def test_search_knowledge_wraps_hits_and_errors(monkeypatch):
    class Hit:
        def search(self, q):
            return [{"text": "call 1930", "source": "portal.md"}]

    class Boom:
        def search(self, q):
            raise RuntimeError("down")

    monkeypatch.setattr(tools, "get_kb", lambda: Hit())
    assert tools.search_knowledge("fraud")["status"] == "ok"
    monkeypatch.setattr(tools, "get_kb", lambda: Boom())
    assert tools.search_knowledge("fraud")["status"] == "error"


def test_letter_deepfake_has_media_enclosures_not_bank():
    r = tools.draft_complaint_letter(
        applicant_name="Rajveer",
        phone="[Phone Number]",
        police_station="Neb Thana",
        incident_category="Social Media Harassment / Fake Profile",
        incident_details="Deepfake video created by Kritarth",
        incident_date="4 November 2026",
        language="en",
    )
    assert r["status"] == "ok"
    assert "Rajveer" in r["letter"]
    assert "Neb Thana" in r["letter"]
    assert "Kritarth" in r["letter"]
    assert "deepfake" in r["letter"].lower()
    assert "Bank statement" not in r["letter"]
    assert "morphed media" in r["letter"].lower() or "video" in r["letter"].lower()

