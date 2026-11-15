"""ADK function tools. Plain functions: ADK builds the Gemini schema from signature + docstring.
Every tool returns a dict with a `status` key and never raises (the model must be able to recover)."""

import logging
from datetime import date
from urllib.parse import quote_plus

from app.agent.facts import CONTACTS, INCIDENT_CATEGORIES, MAPS_URL, STATIONS
from app.agent.guides import GUIDES, render_guide
from app.services.rag import get_kb

log = logging.getLogger(__name__)

LETTER_FIELD_MAX = 600


def search_knowledge(query: str) -> dict:
    """Search the cyber crime knowledge base: how to report, fraud types, what to tell police and the bank,
    documents needed, applications and services (Chakshu, lost phone, tenant verification), cyber safety.
    The knowledge base is in English.

    Args:
        query: A short, self-contained English search query that includes the key facts from the
            conversation, e.g. "documents needed to file complaint for UPI fraud". Translate Hindi to English.

    Returns:
        dict with status and results (text, source, section). status "no_results" means the question is
        outside the knowledge base: do not answer it from your own knowledge.
    """
    try:
        hits = get_kb().search(query)
    except Exception:
        log.exception("knowledge search failed")
        return {"status": "error", "message": "knowledge base unavailable"}
    if not hits:
        return {
            "status": "no_results",
            "message": "Nothing relevant found. Probably outside cyber crime help; do not answer from memory.",
        }
    return {"status": "ok", "results": hits}


def get_complaint_guide(incident_type: str, language: str) -> dict:
    """Get the exact checklist for HOW to complain or apply: where and how to report, documents to keep
    ready, what to tell, and helplines. The full checklist is shown to the user on screen.

    Args:
        incident_type: Pick by topic. financial_fraud = money was actually lost to UPI/OTP/card/online payment fraud.
            suspicious_call_or_sms = how to report a fake or suspicious call, SMS or WhatsApp message
            when no money was lost (Chakshu); use this for any "fake call" or "fake SMS" question.
            lost_phone_or_sim_fraud = lost or stolen phone, SIM misuse. account_hacking = hacked social
            media or email. harassment_or_fake_profile = harassment, fake profile. sextortion_or_morphed_images
            = blackmail, intimate or morphed images. job_loan_investment_scam = job, loan app, investment or
            trading scam. tenant_verification = tenant or servant verification application.
            other_cyber_crime = anything else cyber related.
        language: "en" for English or "hi" for Hindi (match the language you are replying in).

    Returns:
        dict with status, title and guide_text.
    """
    lang = language if language in ("en", "hi") else "en"
    rendered = render_guide(incident_type, lang)
    if rendered is None:
        return {"status": "error", "message": f"unknown incident_type; use one of {sorted(GUIDES)}"}
    title, text = rendered
    return {"status": "ok", "title": title, "guide_text": text}


def get_contacts() -> dict:
    """Get official helpline numbers and official links: Alwar Cyber Control Room, 1930, 112,
    cybercrime.gov.in, Chakshu (report fake call/SMS), WhatsApp cyber desk.

    Returns:
        dict with status and contacts.
    """
    return {"status": "ok", "contacts": CONTACTS}


def find_police_station(area: str) -> dict:
    """Find police stations in Alwar with a Google Maps link.

    Args:
        area: Part of a station or area name to filter by, e.g. "Kotwali". Use an empty string for all stations.

    Returns:
        dict with status and matching stations (name, area, map_url).
    """
    needle = area.strip().lower()
    found = [
        {**s, "map_url": MAPS_URL + quote_plus(f"{s['name']} Alwar")}
        for s in STATIONS
        if not needle or needle in s["name"].lower() or needle in s["area"].lower()
    ]
    return {"status": "ok" if found else "no_results", "stations": found}


def _clip(value: str) -> str:
    return " ".join(value.split())[:LETTER_FIELD_MAX]


def draft_complaint_letter(
    applicant_name: str,
    phone: str,
    police_station: str,
    incident_category: str,
    incident_details: str,
    incident_date: str,
    language: str,
) -> dict:
    """Draft a formal cyber-crime complaint letter addressed to the Station House Officer (SHO).
    Only call this once you have every detail from the user. Never invent values.

    Args:
        applicant_name: Full name of the complainant.
        phone: Complainant's mobile number.
        police_station: Alwar police station the letter is addressed to.
        incident_category: One of: Online Financial Fraud / OTP Theft, WhatsApp / Fake Call Impersonation,
            Social Media Harassment / Fake Profile, ATM / Credit Card Fraud, Job / Loan Scam, Other.
        incident_details: What happened: transaction ID, bank, amount lost, caller number, etc.
        incident_date: Date/time of the incident as the user said it.
        language: "en" for English or "hi" for Hindi.

    Returns:
        dict with status and the full letter text.
    """
    if language not in ("en", "hi"):
        return {"status": "error", "message": "language must be 'en' or 'hi'"}
    f = {
        "name": _clip(applicant_name),
        "phone": _clip(phone),
        "station": _clip(police_station),
        "category": _clip(incident_category) if incident_category in INCIDENT_CATEGORIES else "Other",
        "details": _clip(incident_details),
        "when": _clip(incident_date),
        "today": date.today().strftime("%d/%m/%Y"),
    }
    if not all(f[k] for k in ("name", "phone", "station", "details", "when")):
        return {"status": "error", "message": "missing required details; ask the user"}
    return {"status": "ok", "letter": (_LETTER_HI if language == "hi" else _LETTER_EN).format(**f)}


_LETTER_EN = """To,
The Station House Officer (SHO),
Police Station: {station}, Alwar District (Rajasthan)

Subject: Formal complaint regarding {category}.

Respected Sir,
I am writing to bring to your notice an incident of cyber crime. My name is {name}, Contact No: {phone}.

Incident date: {when}
Incident description:
{details}

I request you to kindly register my complaint and initiate the necessary legal investigation. I have also reported / will report this on the National Cyber Crime Helpline 1930 and cybercrime.gov.in.

Enclosures:
1. Bank statement / transaction receipt
2. Screenshots / call log evidence

Sincerely,
Name: {name}
Phone: {phone}
Date: {today}"""

_LETTER_HI = """सेवा में,
श्रीमान थानाधिकारी महोदय,
पुलिस थाना: {station}, जिला अलवर (राजस्थान)

विषय: {category} के संबंध में शिकायत पत्र।

महोदय,
सविनय निवेदन है कि मेरा नाम {name} है, मोबाइल नंबर {phone} है।

घटना की तिथि: {when}
घटना का विवरण:
{details}

अतः श्रीमान जी से निवेदन है कि मेरी शिकायत दर्ज कर मामले की जांच कर उचित कानूनी कार्रवाई करने की कृपा करें। मैंने राष्ट्रीय साइबर हेल्पलाइन 1930 / cybercrime.gov.in पर भी सूचना दी है / दूँगा/दूँगी।

संलग्न दस्तावेज़:
1. बैंक स्टेटमेंट / ट्रांजैक्शन रसीद
2. स्क्रीनशॉट / कॉल विवरण

भवदीय,
नाम: {name}
फ़ोन: {phone}
दिनांक: {today}"""

ALL_TOOLS = [
    search_knowledge,
    get_complaint_guide,
    get_contacts,
    find_police_station,
    draft_complaint_letter,
]
