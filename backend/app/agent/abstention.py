"""Deterministic Abstention and Safety-Critical Guardrails for the Alwar Cyber Crime Agent.

Implements Task 6 (Deterministic Abstention) of Phase 2:
1. Distinguishes supported, partially supported, unsupported, and out-of-domain queries.
2. Enforces deterministic abstention on safety-critical domains:
   - Reporting numbers and official contacts (no invented phone numbers).
   - Complaint filing fees (strictly 100% free by law; no fabricated fees).
   - Turnaround deadlines & recovery guarantees (no false 100% refund promises).
   - Legal & judicial outcomes (no judicial guarantees of conviction or bail denial).
3. Preserves valid source attribution for supported claims in mixed queries.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.facts import CONTROL_ROOM, OFFICIAL_FACTS

# Authoritative verified emergency and government phone numbers
VERIFIED_PHONE_NUMBERS = frozenset({
    "1930",
    "112",
    "1090",
    CONTROL_ROOM,  # "8764874306"
    "0144-2337456",
    "01442337456",
    "2337456",
})

# Authoritative government and verified safety portals
VERIFIED_DOMAINS = frozenset({
    "cybercrime.gov.in",
    "sancharsaathi.gov.in",
    "police.rajasthan.gov.in",
    "stopncii.org",
    "rbi.org.in",
    "gac.gov.in",
    "ceir.sancharsaathi.gov.in",
})

# Recognized domain tags that represent valid cyber safety / crime topics
CYBER_DOMAINS = frozenset({
    "deepfake",
    "sextortion",
    "harassment",
    "social_media",
    "financial_fraud",
    "account_hacking",
    "phishing",
    "malware",
    "telecom",
    "device",
    "local_police",
    "reporting",
    "evidence",
    "legal_procedure",
    "cyber_safety",
})

# Explicit abstention regex patterns (English, Hindi, Hinglish)
ABSTENTION_PATTERNS = [
    re.compile(r"\b(?:could\s+not|cannot|can\s+not|unable\s+to)\s+(?:be\s+)?verif(?:y|ied)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:available|present|found)\s+(?:in\s+)?(?:available|official)?\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+matching\s+official\s+records|evidence\s+is\s+insufficient)\b", re.IGNORECASE),
    re.compile(r"\b(?:specific\s+detail|specific\s+official\s+fact)\s+(?:could\s+not|cannot)\s+be\s+verified\b", re.IGNORECASE),
    re.compile(r"\b(?:personal\s+(?:mobile|phone)|direct\s+(?:number|phone|line))\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+fixed\s+deadline|cannot\s+(?:be\s+)?guarantee(?:d)?|no\s+refund\s+guarantee)\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+fee|free\s+of\s+cost|no\s+charge\s+for\s+filing)\b", re.IGNORECASE),
    # Hinglish patterns
    re.compile(r"\b(?:satyapit\s+nahi|verify\s+nahi|verified\s+nahi|records?\s+me\s+nahi|records?\s+mein\s+nahi)\b", re.IGNORECASE),
    re.compile(r"\b(?:available\s+nahi\s+hai|guarantee\s+nahi\s+(?:hai|hoti)|koi\s+(?:official\s+)?fee\s+nahi)\b", re.IGNORECASE),
]

DEV_ABSTENTION_SUBSTRINGS = (
    "सत्यापित नहीं",
    "सत्यापित नहीं हो सका",
    "सत्यापित नहीं किया जा सका",
    "रिकॉर्ड में नहीं",
    "रिकॉर्ड में उपलब्ध नहीं",
    "विवरण सत्यापित नहीं",
    "विवरण उपलब्ध नहीं",
    "उपलब्ध नहीं",
    "पुष्टि नहीं की जा सकती",
    "आधिकारिक रिकॉर्ड में दर्ज नहीं",
    "कोई शुल्क नहीं",
    "निःशुल्क",
    "गारंटी नहीं दी जा सकती",
)

# High-risk claim trigger patterns that require statutory clarification if asserted positively
FEE_ASSERTION_PATTERNS = [
    re.compile(
        r"\b(?:fee\s+of\s+rs|registration\s+fee|pay\s+(?:rs|rupees?|\$)\s*\d+|deposit\s+charge|"
        r"complaint\s*(?:charge|fee)|(?:fir|shikayat|complaint)\s*(?:darj\s*)?(?:karane|karne\s*)?(?:ke|ka)\s*(?:paise|kharcha|fee|charge)|"
        r"(?:rs|rupees?|\d+)\s*(?:lagega|denge|dena\s*hoga|charge\s*lagega)|portal\s*(?:fee|charge)|kharcha\s*aayega)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:शुल्क\s+देना\s+होगा|फीस\s+लगेगी|पंजीकरण\s+शुल्क\s+\d+|रुपये\s+जमा\s+करने\s+होंगे|"
        r"शिकायत\s*की\s*फीस|सरकारी\s*शुल्क\s*लगेगा|पैसे\s*लगेंगे|खर्चा\s*आएगा)",
    ),
]

REFUND_GUARANTEE_PATTERNS = [
    re.compile(
        r"\b(?:100%\s*(?:refund|money\s*back|wapas|guarantee)|guaranteed\s+(?:refund|money\s*back|recovery)|"
        r"refund\s+within\s+(?:24|48)\s*hours|poore\s*paise\s*wapas\s*(?:milenge|honge)\s*guarantee|"
        r"24\s*ghante\s*me\s*refund\s*(?:hoga|pakka)|guaranteed\s*paisa\s*wapas)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:रिफंड\s*की\s*100%\s*गारंटी|24\s*घंटे\s*में\s*पैसे\s*वापस|100%\s*गारंटी|पूरी\s*रकम\s*वापस\s*होगी|"
        r"पैसे\s*वापसी\s*की\s*पक्की\s*गारंटी)",
    ),
]

JUDICIAL_PROMISE_PATTERNS = [
    re.compile(
        r"\b(?:guaranteed\s*arrest|accused\s*will\s*definitely\s*be\s*(?:jailed|arrested)|bail\s*will\s*be\s*denied\s*strictly|"
        r"accused\s*ko\s*turant\s*arrest\s*kar\s*liya\s*jayega|jail\s*bhej\s*diya\s*jayega\s*(?:zaroor|pakka)|"
        r"bail\s*pakka\s*reject\s*hogi)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:पक्की\s*गिरफ्तारी\s*होगी|जमानत\s*नहीं\s*मिलेगी\s*यह\s*निश्चित\s*है|निश्चित\s*रूप\s*से\s*गिरफ्तार\s*किया\s*जाएगा|"
        r"जेल\s*भेजा\s*जाएगा\s*निश्चित)",
    ),
]

OUT_OF_DOMAIN_PATTERNS = [
    "pizza", "recipe", "cake", "cook", "cooking", "biryani", "food", "restaurant", "hotel",
    "joke", "weather", "cricket", "score", "movie", "football", "capital of", "oscars", "award",
    "code in", "python script", "binary tree", "song", "lyrics", "horoscope",
    "chutkula", "kahani", "shayari", "gaana", "mausam",
    "पिज्जा", "रेसिपी", "चुटकुला", "गाना", "मौसम", "क्रिकेट", "कहानी", "शायरी", "सब्जी", "खाना", "फिल्म"
]

CYBER_KEYWORDS = frozenset({
    "police", "thana", "fraud", "scam", "cheat", "bank", "paisa", "paise", "money",
    "call", "message", "otp", "sim", "link", "account", "crime", "cyber", "report",
    "alwar", "station", "fir", "harass", "threat", "hack", "safe", "bachav", "1930",
    "112", "1090", "sho", "sp", "inspector", "arrest", "court", "law", "freeze", "lien",
    "fee", "complaint",
    "पुलिस", "थाना", "धोखा", "ठगी", "बैंक", "पैसे", "कॉल", "मैसेज", "ओटीपी", "सिम", "लिंक",
    "शिकायत", "दर्ज", "एफआईआर", "साइबर", "शुल्क", "फीस", "सबूत"
})


def is_abstention_expressed(reply: str) -> bool:
    """Determine whether the reply expresses an explicit factual limitation or abstention."""
    if not reply:
        return False
    if any(p.search(reply) for p in ABSTENTION_PATTERNS):
        return True
    return any(sub in reply for sub in DEV_ABSTENTION_SUBSTRINGS)


def is_out_of_scope_query(
    query: str,
    domains: list[str],
    candidate_hits: list[dict[str, Any]] | None = None,
) -> bool:
    """Detect whether a query is completely unrelated to cybercrime, digital safety, or police procedures."""
    if not query or not query.strip():
        return True

    # If the domain classifier detected specific cybercrime domains, it is in-scope
    if any(d in CYBER_DOMAINS for d in domains):
        return False

    # If knowledge search retrieved relevant verified chunks, it is in-scope
    if candidate_hits:
        return False

    q_lower = query.lower()

    # Check for general greetings
    conversational_tokens = {"hi", "hello", "namaste", "hey", "help", "madad", "shukriya", "thanks", "thank you"}
    tokens = set(re.findall(r"[\w]+", q_lower))
    if tokens and tokens.issubset(conversational_tokens):
        return False

    # Check for known out-of-domain patterns (recipes, jokes, programming, weather, sports, general entertainment)
    if any(ind in q_lower for ind in OUT_OF_DOMAIN_PATTERNS):
        return True

    # If query contains any domain terms, keep in-scope
    if any(term in q_lower for term in CYBER_KEYWORDS):
        return False

    # If no cyber domains, no candidate hits, not a greeting, and has no cyber keywords, it is out-of-scope
    return True


def get_scope_abstention_reply(lang: str = "en") -> str:
    """Standard brief, polite scope abstention for out-of-domain inquiries."""
    if lang == "hi":
        return (
            "Main keval cyber crime reporting, online fraud, digital safety aur Alwar police procedures mein sahayata kar sakta hoon. "
            f"Cyber sahayata ke liye National Helpline 1930 ya Alwar Cyber Control Room ({CONTROL_ROOM}) par call karein."
        )
    return (
        "I can only assist with cyber crime reporting, online fraud, digital safety, and police procedures in Alwar. "
        f"For cyber assistance, please contact National Helpline 1930 or Alwar Cyber Control Room at {CONTROL_ROOM}."
    )


OFFICIAL_CONTACT_CONTEXT = re.compile(
    r"\b(?:officer|inspector|sho|sub-inspector|police|helpline|desk|call|contact|dial|control\s*room|"
    r"अधिकारी|थाना|इंस्पेक्टर|कॉल|संपर्क|नंबर|number)\b",
    re.IGNORECASE,
)


def sanitize_high_risk_claims(
    reply: str,
    lang: str = "en",
    user_message: str = "",
    retrieved_context: str = "",
) -> tuple[str, bool]:
    """Inspect generated text for high-risk hallucinations (unverified phone numbers, fees, false refund guarantees).

    Context-aware phone number protection:
    - Does NOT overwrite numbers typed by the user (suspect's number, complainant's own contact).
    - Does NOT overwrite numbers present in verified retrieved chunks.
    - Preserves transaction IDs, UTRs, and complaint acknowledgement numbers.
    - Only replaces a number if it appears in a context claiming to be an official police/helpline contact
      and is not in VERIFIED_PHONE_NUMBERS.

    Returns:
        tuple[str, bool]: (sanitized_reply, was_modified)
    """
    clean_reply = (reply or "").strip()
    if not clean_reply:
        return clean_reply, False

    modified = False

    # 1. Context-aware phone number sanitization
    for m in re.finditer(r"\b(?:\+91[\-\s]?)?([6-9]\d{9})\b", clean_reply):
        num = m.group(1)
        if num in VERIFIED_PHONE_NUMBERS:
            continue
        if user_message and num in user_message:
            continue
        if retrieved_context and num in retrieved_context:
            continue

        start_idx = m.start()
        prefix = clean_reply[max(0, start_idx - 30) : start_idx].lower()
        if any(tag in prefix for tag in ("utr", "txn", "transaction", "ack", "complaint id", "ref", "id", "case", "receipt")):
            continue

        # Check if surrounding context claims to be an official police/officer contact
        window = clean_reply[max(0, start_idx - 60) : min(len(clean_reply), m.end() + 60)].lower()
        if OFFICIAL_CONTACT_CONTEXT.search(window):
            clean_reply = clean_reply.replace(num, CONTROL_ROOM)
            modified = True

    # 2. Check for fabricated filing fee assertions
    if any(p.search(clean_reply) for p in FEE_ASSERTION_PATTERNS):
        if not re.search(r"\b(?:no\s+fee|free\s+of\s+cost|कोई\s+शुल्क\s+नहीं|निःशुल्क|koi\s+(?:official\s+)?fee\s+nahi)\b", clean_reply, re.IGNORECASE):
            if lang == "hi":
                fee_correction = " (ध्यान दें: साइबर अपराध शिकायत अथवा एफआईआर दर्ज कराने के लिए कोई आधिकारिक शुल्क नहीं लगता है, यह पूर्णतः निःशुल्क है।)"
            elif lang == "hinglish":
                fee_correction = " (Dhyan dein: Cyber crime shikayat ya FIR darj karane ke liye koi official fee nahi lagti, yeh poori tarah free of cost hai.)"
            else:
                fee_correction = " (Note: Filing a cyber crime complaint or FIR on cybercrime.gov.in or at police stations is completely free of cost. No official fee is charged.)"
            clean_reply = clean_reply + fee_correction
            modified = True

    # 3. Check for fabricated 100% refund guarantees or turnaround deadlines
    if any(p.search(clean_reply) for p in REFUND_GUARANTEE_PATTERNS):
        if not re.search(r"\b(?:cannot\s+be\s+guaranteed|no\s+guarantee|गारंटी\s+नहीं|guarantee\s+nahi)\b", clean_reply, re.IGNORECASE):
            if lang == "hi":
                guarantee_correction = " (ध्यान दें: रिफंड की कोई निश्चित समय-सीमा या 100% गारंटी नहीं होती; राशि वापसी इस पर निर्भर करती है कि गोल्डन आवर में ठग के खाते में रकम होल्ड हो सकी या नहीं।)"
            elif lang == "hinglish":
                guarantee_correction = " (Dhyan dein: Refund ki koi fixed time guarantee nahi hoti; recovery is baat par depend karti hai ki golden hour mein suspect account me funds freeze huye ya nahi.)"
            else:
                guarantee_correction = " (Note: Full recovery or fixed refund deadlines cannot be guaranteed; fund return depends strictly on whether stolen money was successfully frozen in suspect accounts during the golden hour.)"
            clean_reply = clean_reply + guarantee_correction
            modified = True

    # 4. Check for guaranteed arrest or judicial bail promises
    if any(p.search(clean_reply) for p in JUDICIAL_PROMISE_PATTERNS):
        if lang == "hi":
            judicial_correction = " (ध्यान दें: गिरफ्तारी और जमानत का निर्णय पुलिस जांच और न्यायालय के विवेक पर निर्भर करता है।)"
        elif lang == "hinglish":
            judicial_correction = " (Dhyan dein: Arrest aur bail ka faisla police investigation aur court ke discretion par depend karta hai.)"
        else:
            judicial_correction = " (Note: Arrests and judicial bail outcomes are subject to police investigation and court discretion; no specific judicial outcome can be guaranteed in advance.)"
        clean_reply = clean_reply + judicial_correction
        modified = True

    return clean_reply, modified
