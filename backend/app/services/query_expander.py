"""Domain classification, query normalization, and Unicode-aware tokenization.
Handles English, Hindi (Devanagari), and natural spoken Hinglish idioms.
"""

import re
import unicodedata

# Unicode-aware word token pattern matching Latin, Devanagari, and digits
UNICODE_WORD = re.compile(r"[\w]+", re.UNICODE)

# Common generic stopwords in English and Hindi/Hinglish (domain terms like upi/otp/1930 are explicitly excluded)
STOPWORDS = frozenset(
    (
        "a an the and or of to in on for is are was were be been i my me you your it this that with at by from "
        "as do does did how what when where which who can should would will am not no "
        "hai hain ho gaye gaya gayi tha thi the ke ki ka ko se me mein par aur ya bhi toh mera meri mere "
        "kisi kuch apne please help sir madam tell mujhe hamara hamare hum"
    ).split()
)

# Crucial domain terms that must NEVER be discarded as stopwords
PROTECTED_TERMS = frozenset(
    {
        "1930", "112", "1090", "upi", "otp", "sim", "kyc", "fir", "sho", "bank", "sms", "call", "apk",
        "pin", "cvv", "imei", "ceir", "chakshu", "tafcop", "bns", "bnss", "it", "alwar", "thana", "gpay",
        "phonepe", "paytm", "cyber", "police", "arrest", "zero", "loan", "fraud", "scam"
    }
)

# Common Hinglish/Hindi colloquial idioms mapped to search concepts
EXPANSION_RULES = [
    # Financial fraud / unauthorized debit
    (
        re.compile(r"\b(paise?\s+(kat|nikal|ud|chale)\s*(gaye?|liye?)|khate?\s+se\s+paise?|account\s+se\s+paisa)\b", re.IGNORECASE),
        "unauthorized debit upi fraud bank transaction dispute 1930 golden hour",
    ),
    (
        re.compile(r"\b(upi\s+se\s+paisa|gpay|phonepe|paytm)\b", re.IGNORECASE),
        "upi payment fraud transaction id utr freeze account 1930",
    ),
    (
        re.compile(r"\b(otp\s+(de\s+diya|bata\s+diya|le\s+liya|share\s+kar\s+diya))\b", re.IGNORECASE),
        "otp phishing unauthorized bank transaction block card netbanking",
    ),
    # Deepfake and morphed media
    (
        re.compile(r"\b(deepfake|deep\s*fake|face\s*swap|nakli\s*(video|photo)|ai\s*(video|photo|voice))\b", re.IGNORECASE),
        "deepfake morphed media ai face swap non consensual intimate video preservation stopncii 66E",
    ),
    (
        re.compile(r"\b(morph(ed)?\s*(photo|image|video|pic)|aslil\s*(photo|video)|gandi\s*(photo|video))\b", re.IGNORECASE),
        "morphed photo image manipulation blackmail extortion cybercrime.gov.in evidence",
    ),
    # Sextortion and video call blackmail
    (
        re.compile(r"\b(sextortion|video\s*call\s*(recording|blackmail)|nude\s*(call|video)|blakmail)\b", re.IGNORECASE),
        "sextortion video call blackmail extortion do not pay private profile 8764874306",
    ),
    # Digital arrest and impersonation
    (
        re.compile(r"\b(digital\s*arrest|cbi\s*(call|officer)|police\s*threat|skype\s*arrest|fake\s*warrant)\b", re.IGNORECASE),
        "digital arrest scam police cbi impersonation fake arrest video call no provision in law",
    ),
    # Harassment and stalking
    (
        re.compile(r"\b(dhamki|threat(en)?|harass(ment)?|gali|badtameezi|piche\s*pada|stalk(ing)?)\b", re.IGNORECASE),
        "cyber harassment stalking abusive messages mahila thana 1090 8764874306",
    ),
    # Fake profile and social media impersonation
    (
        re.compile(r"\b(fake\s*profile|nakli\s*(id|account)|mera\s*naam\s*se\s*account|फेक\s*(प्रोफाइल|अकाउंट|आईडी)|नकली\s*(प्रोफाइल|आईडी))\b", re.IGNORECASE),
        "fake profile social media impersonation report account identity theft",
    ),
    # Account hacking
    (
        re.compile(r"\b(account\s*hack|insta(gram)?\s*hack|whatsapp\s*hack|facebook\s*hack|login\s*nahi\s*ho\s*raha|हैक|अकाउंट\s*हैक)\b", re.IGNORECASE),
        "account hacking unauthorized access recovery two factor authentication reset password",
    ),
    # Telecom and SIM
    (
        re.compile(r"\b(sim\s*swap|sim\s*(band|block|deactivate)\s*ho\s*gayi)\b", re.IGNORECASE),
        "sim swap fraud duplicate sim telecom operator unauthorized debit",
    ),
    # Lost phone and IMEI
    (
        re.compile(r"\b(phone\s*(chori|kho|gum)|mobile\s*(chori|kho|lost|stolen)|imei\s*block)\b", re.IGNORECASE),
        "lost stolen mobile phone ceir sanchar saathi block imei police receipt",
    ),
    # Suspicious call / Chakshu
    (
        re.compile(r"\b(farzi\s*call|fake\s*call|spam\s*(call|sms)|suspicious\s*(call|message))\b", re.IGNORECASE),
        "chakshu portal report suspicious call sms sanchar saathi before loss",
    ),
    # Malicious APK
    (
        re.compile(r"\b(apk|anydesk|teamviewer|bill\s*update\s*apk|challan\s*apk|remote\s*access)\b", re.IGNORECASE),
        "malicious apk anydesk remote access malware airplane mode uninstall",
    ),
    # Local police station / Alwar
    (
        re.compile(r"\b(alwar|thana|police\s*station|kotwali|neb|aravali\s*vihar|shivaji\s*park|mahila\s*thana|sho)\b", re.IGNORECASE),
        "alwar police station local jurisdiction thana control room 8764874306 zero fir",
    ),
]


def normalize_text(text: str) -> str:
    """Normalize unicode, strip accents, strip digit commas, lower-case, and normalize whitespace."""
    norm = unicodedata.normalize("NFKD", text)
    # Remove commas between digits e.g. 50,000 -> 50000
    norm = re.sub(r"(?<=\d),(?=\d)", "", norm)
    return " ".join(norm.split())


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokenization supporting English, Devanagari, and numbers."""
    cleaned = normalize_text(text).lower()
    tokens = UNICODE_WORD.findall(cleaned)
    out: list[str] = []
    for t in tokens:
        if t in PROTECTED_TERMS or (t not in STOPWORDS and len(t) > 1):
            out.append(t)
    return out


def classify_domains(query: str) -> list[str]:
    """Classify user query into one or more controlled domain tags.
    Supports multi-label classification.
    """
    q = query.lower()
    domains: set[str] = set()

    # Casual greeting / conversation check
    greeting_patterns = {
        "hi", "hello", "hey", "namaste", "namaskar", "kese", "kaise", "ho", "a", "lot", "very", "much",
        "thanks", "thank", "you", "dhanyawad", "shukriya", "bye", "goodbye", "good", "morning", "evening", "night"
    }
    words = set(re.findall(r"[\w]+", q))
    if words and words.issubset(greeting_patterns):
        return ["conversation"]

    # 1. Deepfake
    if any(k in q for k in ("deepfake", "deep fake", "morphed", "face swap", "ai video", "ai photo", "ai voice", "nakli video", "डीपफेक", "मॉर्फ")):
        domains.add("deepfake")
        domains.add("social_media")

    # 2. Sextortion
    if any(k in q for k in ("sextortion", "nude", "video call recording", "blackmail", "blakmail", "private photo", "intimate", "ब्लैकमेल", "न्यूड", "अश्लील")):
        domains.add("sextortion")
        domains.add("harassment")

    # 3. Harassment & Stalking & Fake Profile
    if any(k in q for k in ("harass", "stalk", "dhamki", "threat", "abuse", "gali", "badtameezi", "fake profile", "impersonat", "धमकी", "गाली", "फेक प्रोफाइल", "नकली प्रोफाइल", "इंस्टाग्राम", "व्हाट्सएप", "फेसबुक")):
        domains.add("harassment")
        domains.add("social_media")

    # 4. Financial fraud (Only if money loss / transaction occurred or discussed)
    has_money_terms = any(k in q for k in (
        "paise", "paisa", "rupee", "rs", "inr", "upi", "gpay", "phonepe", "paytm", "bank", "debit", "card",
        "atm", "account se", "kat gaye", "chori ho gaye", "transferred", "fraud", "loot", "पैसे", "रुपए",
        "कट गए", "खाते", "बैंक", "digital arrest", "job scam", "task scam", "investment", "trading scam",
        "loan app", "deposit"
    ))
    # Distinguish extortion demand ("asking for money") from actual financial loss
    is_demanding_only = any(k in q for k in ("demanding money", "asking for money", "paisa mang raha", "paise mang", "पैसे मांग")) and not any(k in q for k in ("kat gaye", "bhej diye", "transferred", "paid", "de diye", "कट गए", "दे दिए"))
    
    if has_money_terms and not is_demanding_only:
        domains.add("financial_fraud")

    # 5. Account Hacking
    if any(k in q for k in ("hack", "hacked", "login nahi", "access kho", "password change", "account takeover", "हैक")):
        domains.add("account_hacking")
        domains.add("social_media")

    # 6. Phishing
    if any(k in q for k in ("phishing", "fake link", "lottery link", "suspicious link", "sms link", "pan update link")):
        domains.add("phishing")

    # 7. Malware
    if any(k in q for k in ("apk", "anydesk", "teamviewer", "quicksupport", "malware", "virus", "screen share")):
        domains.add("malware")

    # 8. Telecom
    if any(k in q for k in ("sim swap", "sim block", "sim band", "chakshu", "tafcop", "farzi call", "fake call", "spam call", "mobile connection", "connection", "in my name", "registered in my name")):
        domains.add("telecom")

    # 9. Device
    if any(k in q for k in ("lost phone", "stolen phone", "phone kho", "mobile chori", "ceir", "imei")):
        domains.add("device")
        domains.add("telecom")

    # 10. Local Police
    if any(k in q for k in ("alwar", "thana", "police station", "sho", "kotwali", "neb", "aravali vihar", "shivaji park", "mahila thana", "bhiwadi", "tijara", "behror", "neemrana", "sadar", "control room", "sp office", "jurisdiction", "ilaka", "area")):
        domains.add("local_police")

    # 11. Reporting & Evidence
    if any(k in q for k in ("1930", "cybercrime.gov.in", "fir", "zero fir", "report", "complaint", "shikayat", "evidence", "saboot", "proof", "documents", "screenshot")):
        domains.add("reporting")
        domains.add("evidence")

    # 12. Legal Procedure
    if any(k in q for k in ("law", "section", "act", "it act", "bnss", "bns", "ipc", "rbi", "zero fir", "police refuse", "liability")):
        domains.add("legal_procedure")

    # 13. Cyber Safety
    if any(k in q for k in ("safety", "safe", "secure", "strong password", "2fa", "two factor", "tips", "bachav")):
        domains.add("cyber_safety")

    if not domains:
        domains.add("general")

    return sorted(domains)


def expand_query(query: str) -> str:
    """Expand user query with English domain keywords for Hinglish/Hindi queries."""
    expanded_parts = [query]
    for pattern, expansion in EXPANSION_RULES:
        if pattern.search(query):
            expanded_parts.append(expansion)
    return " ".join(expanded_parts)
