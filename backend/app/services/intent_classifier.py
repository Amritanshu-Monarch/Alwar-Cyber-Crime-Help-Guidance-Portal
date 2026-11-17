"""Deterministic, high-precision intent classifier with three outcomes:
1. 'in_scope': Verified cybercrime, digital fraud, safety, or police procedure topics.
2. 'out_of_scope': General knowledge, cooking, sports, entertainment, coding, weather, etc.
3. 'needs_clarification': Ambiguous or borderline queries where cyber context is possible.

Never rejects silently: borderline queries return targeted clarifying questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.agent.facts import CONTROL_ROOM

IntentOutcome = Literal["in_scope", "out_of_scope", "needs_clarification"]

# 1. Definitive in-scope cyber/police/telecom/fraud keywords & regexes
IN_SCOPE_PATTERNS = [
    # Helplines & Numbers
    re.compile(r"\b(1930|112|1090|8764874306|2337456)\b"),
    # Financial Fraud & Payment Scams
    re.compile(
        r"\b(upi|otp|atm|credit\s*card|debit\s*card|net\s*banking|gpay|google\s*pay|phonepe|paytm|"
        r"bank\s*transfer|unauthorized\s*(debit|transaction)|wrong\s*(transfer|upi)|galat\s*(transfer|upi|account)|"
        r"aeps|biometric\s*fraud|kyc\s*update|qr\s*code|olx\s*fraud|lottery\s*fraud|part\s*time\s*job|"
        r"work\s*from\s*home|investment\s*scam|trading\s*scam|telegram\s*(task|scam|fraud|group)|"
        r"money\s*lost|paise\s*(kat|cut|nikal|loot|chale)|khate\s*se\s*paise|rupaye\s*(kat|chale|nikal)|"
        r"paisa\s*wapas|electricity\s*(bill|cut|update)|bijli\s*bill|scammed\s*me)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:पैसे\s*कट|खाते\s*से\s*पैसे|ओटीपी|यूपीआई|रुपये\s*कट|बैंक\s*फ्रॉड|अनधिकृत\s*लेनदेन|पैसे\s*वापस|बायोमेट्रिक|"
        r"बिजली\s*बिल|बिजली\s*काट|पार्ट\s*टाइम\s*जॉब)",
    ),
    # Account Hacking, Impersonation & Social Media
    re.compile(
        r"\b(hack|hacked|hacker|hackers|takeover|taken\s*over|unauthorized\s*access|compromised|whatsapp\s*hack|"
        r"instagram\s*hack|facebook\s*hack|telegram\s*hack|gmail\s*hack|google\s*account\s*hack|"
        r"fake\s*profile|duplicate\s*profile|impersonat|cloned\s*(my\s*)?(profile|account)|nakli\s*profile|"
        r"fake\s*account|fake\s*id|character\s*assassination)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:हैक|खाता\s*कब्जा|कब्जा\s*कर|फेक\s*प्रोफाइल|नकली\s*आईडी|व्हाट्सएप|इंस्टाग्राम|फेसबुक|टेलीग्राम|प्रोफाइल\s*की\s*कॉपी|फेक\s*अकाउंट)"),
    # Deepfake, Extortion, Harassment, Sextortion
    re.compile(
        r"\b(deepfake|morphed|morphing|nude\s*video|private\s*photos?|intimate\s*image|stopncii|"
        r"sextortion|video\s*call\s*blackmail|blackmail|ransom|extortion|threaten|dhamki|cyber\s*bullying|"
        r"cyber\s*harassment|cyber\s*stalking|loan\s*app|chinese\s*loan|doxxing)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:डीपफेक|मॉर्फ|अश्लील|ब्लैकमेल|धमकी|सेक्सटॉर्शन|लोन\s*ऐप|स्टॉपएनसीआईआई|साइबर\s*स्टॉकिंग|गाली-गलौज)"),
    # Device, Malware & Telecom
    re.compile(
        r"\b(lost\s*phone|stolen\s*phone|phone\s*chori|mobile\s*kho\s*gaya|ceir|imei|block\s*imei|"
        r"apk|malicious\s*apk|malware|virus|trojan|call\s*forwarding|\*401\*|\*21\*|##002#|\*#62#|"
        r"sim\s*swap|duplicate\s*sim|chakshu|sanchar\s*saathi|remote\s*access|anydesk|teamviewer|"
        r"quicksupport|rustdesk)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:फोन\s*चोरी|मोबाइल\s*खो|आईएमईआई|कॉल\s*फॉरवर्डिंग|एपीके|सिम\s*स्वैप)"),
    # Police, Evidence, Procedures & Alwar Stations
    re.compile(
        r"\b(alwar\s*(police|cyber|thana|control|station)|kotwali|neb\s*thana|kala\s*kuan|shivaji\s*park|"
        r"aravali\s*vihar|mahila\s*thana|cyber\s*thana|cyber\s*crime\s*police|cyber\s*cell|cyber\s*control\s*room|"
        r"ncrp|cybercrime\.gov\.in|digital\s*arrest|mrm\s*portal|grm\s*portal|freeze\s*account|unfreeze|lien|"
        r"tenant\s*verification|servant\s*verification|kirayedar\s*verif|digital\s*evidence|police\s*station|"
        r"evidence\s*(to\s*keep|checklist)|cyber\s*complaint)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:अलवर\s*साइबर|अलवर\s*में\s*(?:किरायेदार|नौकर)|साइबर\s*थाना|साइबर\s*सेल|पुलिस\s*थाने|"
        r"एनसीआरपी|एफआईआर|डिजिटल\s*अरेस्ट|किरायेदार\s*सत्यापन|घरेलू\s*नौकर|पुलिस\s*सत्यापन|"
        r"डिजिटल\s*साक्ष्य|स्क्रीनशॉट\s*तैयार)",
    ),
]

# 2. Definitive out-of-scope non-cyber domains (checked with priority if clear non-cyber query)
OUT_OF_SCOPE_PATTERNS = [
    # Cooking / Food Recipes
    re.compile(
        r"\b(recipe|cook|cooking|bake|cake|brownie|pizza|burger|pasta|biryani|paneer|curry|tea|coffee|"
        r"sweets|gulab\s*jamun|roti|dal|sabzi|masala|rasoi|khana\s*kaise\s*banaye|dish)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:रेसिपी|खाना\s*बनाना|गुलाब\s*जामुन|सब्जी|रोटी|पिज्जा|केक|मिठाई)"),
    # Sports & Cricket
    re.compile(
        r"\b(cricket|match|ipl|world\s*cup|virat\s*kohli|rohit\s*sharma|dhoni|football|fifa|messi|ronaldo|"
        r"goal|wicket|run\s*score|scorecard|stadium|olympics|badminton|tennis|hockey)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:क्रिकेट|मैच|स्कोर|विराट\s*कोहली|आईपीएल|फुटबॉल|ओलंपिक|विश्व\s*कप)"),
    # General Coding & Academic Mathematics/Physics
    re.compile(
        r"\b(binary\s*search|sorting\s*algorithm|sort\s*a\s*list|write\s*a\s*python|write\s*a\s*code|"
        r"javascript\s*code|react\s*hook|react\s*js|html\s*css|c\+\+|java\s*program|algebraic\s*equation|"
        r"trigonometry|calculus|derivative|integral|solve\s*equation|solve\s*this|3x\s*plus|pythagoras|"
        r"special\s*relativity|quantum\s*mechanics|black\s*hole|gravitational\s*wave|speed\s*of\s*light|"
        r"planets|solar\s*system|astronomy|for\s*loop)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:त्रिकोणमिति|समीकरण|पाइथन\s*प्रोग्रामिंग|सौरमंडल|ब्लैक\s*होल|भौतिक\s*विज्ञान)"),
    # Politics, Entertainment, Movies, Jokes
    re.compile(
        r"\b(prime\s*minister\s*of|president\s*of|election\s*in|parliament|congress|bjp|movie\s*plot|"
        r"movie\s*release|film\s*release|box\s*office|actor|actress|cinema|theatre|hollywood|bollywood|"
        r"inception|pushpa|sholay|tell\s*me\s*a\s*joke|funny\s*joke|chutkula|comedy)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:फिल्म|मूवी|जोक|चुटकुला|अभिनेता|निर्देशक|राष्ट्रपति|प्रधानमंत्री|चुनाव)"),
    # Tourism, Weather, Automotive, Agriculture, Health/Medical, Stock Market
    re.compile(
        r"\b(weather\s*forecast|barish|rain\s*tomorrow|dhoop|mausam|temperature\s*today|"
        r"tourist\s*places|visit\s*in|sariska|siliserh|tiger\s*reserve|hotel\s*booking|car\s*engine|engine\s*oil|"
        r"car\s*mileage|petrol\s*mileage|puncture|car\s*ac|farming|agriculture|fertilizer|kheti|fasal|"
        r"beej\s*kaunsa|sarson|wheat\s*crop|blood\s*pressure|fever|cough|cold|headache|treatment|medicine\s*for|"
        r"share\s*price|nse|bse|mutual\s*fund|bitcoin\s*me\s*invest|crypto\s*bitcoin)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?:मौसम|बारिश|घूमने\s*की\s*जगह|होटल|गाड़ी\s*का\s*माइलेज|पेट्रोल|खेती|फसल|खाद|बुखार|दवा|म्यूचुअल\s*फंड|शेयर\s*बाजार)"),
]

# 3. Borderline ambiguous queries where cyber context is plausible but details are missing
AMBIGUOUS_PATTERNS = [
    # Help / Emergency without context
    re.compile(r"\b(urgent\s*help|please\s*help|help\s*me|madad\s*chahiye|madad\s*karein|emergency)\b", re.IGNORECASE),
    re.compile(r"(?:मदद\s*कीजिए|मदद\s*चाहिए|सहायता\s*चाहिए)"),
    # Phone / Calling calls without fraud details
    re.compile(r"\b(calling\s*(me\s*)?repeatedly|bar\s*bar\s*phone|call\s*karke|ajeeb\s*call|unknown\s*call)\b", re.IGNORECASE),
    re.compile(r"(?:बार-बार\s*फोन|फोन\s*करके\s*परेशान|कॉल\s*आया\s*था)"),
    # Links & Screen taps
    re.compile(r"\b(clicked\s*(on\s*)?a\s*link|strange\s*link|link\s*pe\s*touch|link\s*khol\s*diya)\b", re.IGNORECASE),
    re.compile(r"(?:लिंक\s*आया\s*था|लिंक\s*खोल\s*दिया|लिंक\s*पर\s*क्लिक)"),
    # Device trouble
    re.compile(r"\b(phone\s*stopped\s*working|mobile\s*switch\s*off|phone\s*chal\s*nahi\s*raha)\b", re.IGNORECASE),
    re.compile(r"(?:मोबाइल\s*अचानक\s*से\s*बंद|फोन\s*चल\s*नहीं\s*रहा)"),
    # Generic complaints / police contact
    re.compile(r"\b(file\s*a\s*(general\s*)?complaint|how\s*to\s*complain|complaint\s*karne\s*ka\s*tareeka)\b", re.IGNORECASE),
    re.compile(r"(?:शिकायत\s*पत्र\s*कैसे\s*लिखना|शिकायत\s*कैसे\s*करें)"),
    re.compile(r"\b(police\s*station\s*contact|thana\s*number|police\s*ka\s*number|talk\s*to\s*(an\s*)?officer|sho\s*saheb)\b", re.IGNORECASE),
    re.compile(r"(?:थाना\s*अधिकारी\s*का\s*फोन|पुलिस\s*से\s*बात)"),
    # Unspecified message / loss
    re.compile(r"\b(unexpected\s*message|message\s*aaya\s*tha|message\s*safe|sahi\s*hai\s*ya\s*fake|scam\s*hai\s*kya)\b", re.IGNORECASE),
    re.compile(r"(?:संदेश\s*आया\s*था|मैसेज\s*सही\s*है\s*या\s*फर्जी)"),
    re.compile(r"\b(took\s*something|nuksaan\s*ho\s*gaya|loss\s*ho\s*gaya)\b", re.IGNORECASE),
    re.compile(r"(?:नुकसान\s*कर\s*दिया|नुकसान\s*हो\s*गया)"),
]


@dataclass
class IntentResult:
    status: IntentOutcome
    confidence: float
    domain: str
    clarification_question: str = ""
    reason: str = ""


def get_clarification_prompt(lang: str = "en") -> str:
    """Return an empathetic, targeted clarifying question for borderline cases."""
    if lang == "hi":
        return (
            "क्या आप बता सकते हैं कि यह मामला किसी ऑनलाइन फ्रॉड, बैंक लेनदेन, संदिग्ध कॉल/मैसेज या सोशल मीडिया से जुड़ा है? "
            f"यदि वित्तीय नुकसान हुआ है, तो कृपया तुरंत 1930 या अलवर साइबर कंट्रोल रूम ({CONTROL_ROOM}) पर संपर्क करें।"
        )
    if lang == "hinglish":
        return (
            "Kya aap bata sakte hain ki ye kisi online fraud, unauthorized transaction, suspicious call/link ya social media harassment se juda hai? "
            f"Agar paise ka nuksan hua hai, to sabse pehle turant 1930 ya Alwar Cyber Control Room ({CONTROL_ROOM}) par call karein."
        )
    return (
        "Could you please clarify if this incident involves online fraud, an unauthorized transaction, a suspicious link/call, or social media harassment? "
        f"If you suffered financial loss, please call 1930 or the Alwar Cyber Control Room at {CONTROL_ROOM} immediately."
    )


def classify_intent(query: str, lang: str = "en") -> IntentResult:
    """Classify the user inquiry into in_scope, out_of_scope, or needs_clarification."""
    text = (query or "").strip()
    if not text:
        return IntentResult(
            status="needs_clarification",
            confidence=1.0,
            domain="ambiguous",
            clarification_question=get_clarification_prompt(lang),
            reason="empty_query",
        )

    # 1. Check explicit out-of-scope patterns FIRST (weather, recipes, cricket, movies, code)
    # This prevents general words like "Alwar weather" or "Jaipur hotel" from being misrouted as police queries.
    for pat in OUT_OF_SCOPE_PATTERNS:
        if pat.search(text):
            return IntentResult(
                status="out_of_scope",
                confidence=0.98,
                domain="unrelated",
                reason="matched_out_of_scope_category",
            )

    # 2. Check ambiguous borderline cases
    for pat in AMBIGUOUS_PATTERNS:
        if pat.search(text):
            return IntentResult(
                status="needs_clarification",
                confidence=0.90,
                domain="ambiguous",
                clarification_question=get_clarification_prompt(lang),
                reason="ambiguous_requires_clarification",
            )

    # 3. Check explicit in-scope cyber indicators
    for pat in IN_SCOPE_PATTERNS:
        if pat.search(text):
            return IntentResult(
                status="in_scope",
                confidence=0.95,
                domain="cybercrime",
                reason="matched_cyber_indicator",
            )

    # 4. Short queries without cyber keywords are ambiguous
    if len(text.split()) <= 4:
        return IntentResult(
            status="needs_clarification",
            confidence=0.75,
            domain="ambiguous",
            clarification_question=get_clarification_prompt(lang),
            reason="short_unspecified_query",
        )

    # 5. Default fallback: out_of_scope
    return IntentResult(
        status="out_of_scope",
        confidence=0.70,
        domain="general_out_of_domain",
        reason="no_cyber_indicators",
    )
