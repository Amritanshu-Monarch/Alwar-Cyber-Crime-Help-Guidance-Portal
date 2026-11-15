"""ADK function tools. Plain functions: ADK builds the Gemini schema from signature + docstring.
Every tool returns a dict with a `status` key and never raises (the model must be able to recover)."""

import logging
from datetime import date
from urllib.parse import quote_plus

from app.agent.facts import (
    CATEGORY_HI,
    CONTACTS,
    INCIDENT_CATEGORIES,
    LOCALITY_TO_STATION,
    MAPS_URL,
    STATION_NAME_HI,
    STATIONS,
)
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
            "message": "Nothing relevant found in official records: do not answer from memory.",
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
    """Find police stations in Alwar with locality mapping and a Google Maps link.

    Args:
        area: Part of a station or area name to filter by, e.g. "Kotwali", "Kala Kuan", "NEB", "Shivaji Park".
            Use an empty string for all stations.

    Returns:
        dict with status, matching stations (name, area, map_url), and verified jurisdiction notes.
    """
    needle = area.strip().lower()

    # Check if needle matches a mapped locality in Alwar
    locality_match = None
    for loc_key, station_name in LOCALITY_TO_STATION.items():
        if loc_key in needle or needle in loc_key:
            locality_match = station_name
            break

    found = [
        {**s, "map_url": MAPS_URL + quote_plus(f"{s['name']} Alwar")}
        for s in STATIONS
        if not needle
        or needle in s["name"].lower()
        or needle in s["area"].lower()
        or (locality_match and locality_match.lower() == s["name"].lower())
    ]

    note = ""
    if not found:
        note = "If your exact locality is not found, you can file a Zero FIR at any nearby police station under BNSS Section 173, or contact Alwar Cyber Control Room at 8764874306."
    elif locality_match:
        note = f"Locality '{area.strip()}' falls under the jurisdiction of {locality_match}."

    return {
        "status": "ok" if found else "no_results",
        "stations": found,
        "note": note,
    }


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
            Social Media Harassment / Fake Profile, Deepfake / Morphed Media Harassment,
            Sextortion / Video Call Blackmail, Deepfake & Financial Extortion, ATM / Credit Card Fraud,
            Job / Loan Scam, Hacked Account / Unauthorized Access, Other Cyber Crime.
        incident_details: What happened: transaction ID, bank, amount lost, caller number, etc.
        incident_date: Date/time of the incident as the user said it.
        language: "en" for English or "hi" for Hindi.

    Returns:
        dict with status and the full letter text.
    """
    if language not in ("en", "hi"):
        return {"status": "error", "message": "language must be 'en' or 'hi'"}

    name_clean = _clip(applicant_name)
    phone_clean = _clip(phone) if phone and _clip(phone) not in ("None", "null", "undefined") else "[Phone Number]"
    station_clean = _clip(police_station)
    details_clean = _clip(incident_details)
    when_clean = _clip(incident_date)

    if not name_clean or not station_clean or not details_clean or not when_clean:
        return {"status": "error", "message": "missing required details; ask the user"}

    text_context = (details_clean + " " + _clip(incident_category)).lower()
    has_actual_payment = any(k in text_context for k in ("paid", "transferred", "kat gaye", "bhej diye", "cut gaye", "deducted", "lost", "कट गए", "दे दिए", "भेज दिए")) and any(k in text_context for k in ("rs", "rupee", "inr", "paisa", "paise", "रुपए", "पैसे", "bank", "upi"))
    is_extortion_demand = any(k in text_context for k in ("demand", "demanding", "asking for money", "ransom", "paisa mang", "paise mang", "रंगदारी", "फिरौती", "मांग"))
    is_deepfake_or_harassment = any(k in text_context for k in ("deepfake", "morph", "nude", "video", "photo", "harass", "blackmail", "sextortion", "अश्लील", "न्यूड", "ब्लैकमेल", "फोटो", "वीडियो"))
    is_hacking = any(k in text_context for k in ("hack", "takeover", "unauthorized access", "compromise", "login", "हैक")) or (
        any(k in text_context for k in ("whatsapp", "instagram", "facebook", "telegram", "email", "व्हाट्सएप"))
        and any(k in text_context for k in ("hack", "takeover", "otp", "stolen", "banned", "access", "बैन", "हैक", "कब्जा"))
    )
    is_lost_phone = any(k in text_context for k in ("lost phone", "stolen phone", "imei", "ceir", "फोन चोरी", "मोबाइल खो"))

    # Determine enclosure items based on strict case classification
    if is_deepfake_or_harassment and not has_actual_payment:
        if language == "hi":
            enclosures = (
                "1. स्क्रीनशॉट, धमकी भरे संदेश एवं संदिग्ध सोशल मीडिया प्रोफाइल लिंक\n"
                "2. कथित डीपफेक वीडियो / मॉर्फ्ड मीडिया की डिजिटल प्रति (पेन ड्राइव / सीडी में)\n"
                "3. रंगदारी / फिरौती मांग (extortion demand) और आरोपी के संपर्क नंबर / सोशल मीडिया हैंडल विवरण\n"
                "4. 1930 / cybercrime.gov.in शिकायत पावती प्रति"
            )
        else:
            enclosures = (
                "1. Screenshots of chats, threatening messages, and suspect profile URLs\n"
                "2. Digital copy of alleged deepfake video / morphed media (stored on pen drive / CD)\n"
                "3. Details of extortion demand and contact numbers / handles of the accused\n"
                "4. 1930 / cybercrime.gov.in complaint acknowledgement receipt"
            )
    elif (is_deepfake_or_harassment or is_hacking) and has_actual_payment:
        if language == "hi":
            enclosures = (
                "1. धमकी भरे चैट, संदेश एवं संदिग्ध प्रोफाइल लिंक के स्क्रीनशॉट\n"
                "2. कथित डीपफेक वीडियो / मॉर्फ्ड मीडिया / हैक अकाउंट अलर्ट की डिजिटल प्रति\n"
                "3. रंगदारी / फिरौती मांग (extortion demand) और आरोपी के भुगतान खाते (UPI ID / बैंक विवरण)\n"
                "4. बैंक खाता स्टेटमेंट / डेबिट रसीद (UTR संख्या सहित)\n"
                "5. 1930 / cybercrime.gov.in शिकायत पावती प्रति"
            )
        else:
            enclosures = (
                "1. Screenshots of threatening chats, messages, and suspect profile URLs\n"
                "2. Digital copy of alleged deepfake video / morphed media / hacked account alerts\n"
                "3. Details of extortion demand (suspect account / UPI ID / phone numbers)\n"
                "4. Bank statement / debit transaction receipt showing UTR\n"
                "5. 1930 / cybercrime.gov.in complaint acknowledgement receipt"
            )
    elif is_hacking and not has_actual_payment:
        if language == "hi":
            enclosures = (
                "1. हैक किए गए व्हाट्सएप / सोशल मीडिया अकाउंट एवं अनधिकृत सेशन के स्क्रीनशॉट\n"
                "2. हैकर द्वारा भेजे गए चैट, फिरौती / पैसे मांगने के संदेश एवं मोबाइल नंबर विवरण\n"
                "3. मोबाइल नंबर / सिम स्वामित्व प्रमाण (आधार कार्ड / सिम खरीद रसीद / मोबाइल बिल)\n"
                "4. 1930 / cybercrime.gov.in शिकायत पावती प्रति"
            )
        else:
            enclosures = (
                "1. Screenshots of hacked WhatsApp / social media account and unauthorized access alerts\n"
                "2. Screenshots of hacker's chats, ransom/money demands, and contact numbers\n"
                "3. Proof of mobile number / SIM ownership (Aadhaar ID, mobile bill, or recharge receipt)\n"
                "4. 1930 / cybercrime.gov.in complaint acknowledgement receipt"
            )
    elif is_lost_phone:
        if language == "hi":
            enclosures = (
                "1. मोबाइल खरीद का मूल बिल (IMEI 1 एवं IMEI 2 नंबर सहित)\n"
                "2. डुप्लीकेट सिम जारी होने की रसीद / ऑपरेटर कन्फर्मेशन\n"
                "3. संचार साथी (CEIR) पोर्टल अनुरोध रसीद (Request ID)\n"
                "4. शिकायतकर्ता का पहचान पत्र (आधार कार्ड)"
            )
        else:
            enclosures = (
                "1. Original mobile purchase invoice showing IMEI 1 & IMEI 2 numbers\n"
                "2. Duplicate SIM issuance receipt / telecom confirmation\n"
                "3. Sanchar Saathi (CEIR) portal blocking request receipt\n"
                "4. Complainant's identity proof (Aadhaar card)"
            )
    else:
        if language == "hi":
            enclosures = (
                "1. बैंक स्टेटमेंट / ट्रांजैक्शन रसीद (UTR संख्या सहित)\n"
                "2. फ्रॉड पेमेंट स्क्रीनशॉट / कॉल विवरण\n"
                "3. 1930 / cybercrime.gov.in शिकायत पावती प्रति"
            )
        else:
            enclosures = (
                "1. Bank statement / debit SMS showing transaction ID (UTR)\n"
                "2. Screenshots of fraudulent payment requests, QR codes, or chats\n"
                "3. 1930 / cybercrime.gov.in complaint acknowledgement receipt"
            )

    cat_raw = _clip(incident_category)
    if cat_raw in INCIDENT_CATEGORIES:
        category = cat_raw
    elif "deepfake" in cat_raw.lower() or "morph" in cat_raw.lower():
        category = "Deepfake / Morphed Media Harassment"
    elif "blackmail" in cat_raw.lower() or "sextortion" in cat_raw.lower():
        category = "Sextortion / Video Call Blackmail"
    elif "harass" in cat_raw.lower() or "fake profile" in cat_raw.lower():
        category = "Social Media Harassment / Fake Profile"
    elif "hack" in cat_raw.lower() or "whatsapp" in cat_raw.lower():
        category = "Hacked Account / Unauthorized Access"
    else:
        category = "Other"

    # Format Station and Category for Hindi if requested
    station_display = station_clean
    category_display = category
    if language == "hi":
        category_display = CATEGORY_HI.get(category, category)
        # Check if station matches a known station
        for eng_name, hi_name in STATION_NAME_HI.items():
            if eng_name.lower() in station_clean.lower() or station_clean.lower() in eng_name.lower():
                station_display = hi_name
                break
        if "police station" in station_display.lower():
            station_display = station_display.replace("Police Station", "पुलिस थाना").replace("police station", "पुलिस थाना")

    f = {
        "name": name_clean,
        "phone": phone_clean,
        "station": station_display,
        "category": category_display,
        "details": details_clean,
        "when": when_clean,
        "today": date.today().strftime("%d/%m/%Y"),
        "enclosures": enclosures,
    }
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

I request you to kindly register my complaint under appropriate provisions of the Information Technology Act 2000 and Bharatiya Nyaya Sanhita (BNS) 2023, and initiate the necessary legal investigation. I have also reported / will report this on the National Cyber Crime Helpline 1930 and cybercrime.gov.in.

Enclosures:
{enclosures}

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

अतः श्रीमान जी से निवेदन है कि सूचना प्रौद्योगिकी अधिनियम (IT Act) 2000 एवं भारतीय न्याय संहिता (BNS) 2023 के तहत मेरी शिकायत दर्ज कर मामले की जांच कर उचित कानूनी कार्रवाई करने की कृपा करें। मैंने राष्ट्रीय साइबर हेल्पलाइन 1930 / cybercrime.gov.in पर भी सूचना दी है / दूँगा/दूँगी।

संलग्न दस्तावेज़:
{enclosures}

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
