"""Evidence grounding, source attribution, and citation management for the Alwar Cyber Crime Agent.

Distinguishes:
1. Verified official facts grounded in authoritative knowledge base documents or tools.
2. General cyber-safety guidance.
3. Information that cannot be verified from available sources (communicating limitations clearly).
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.abstention import (
    VERIFIED_DOMAINS,
    VERIFIED_PHONE_NUMBERS,
    is_abstention_expressed,
)
from app.agent.facts import CONTROL_ROOM, OFFICIAL_FACTS
from app.services.query_expander import tokenize


def is_unverified_limitation_expressed(reply: str) -> bool:
    """Check whether the reply explicitly communicates that specific details could not be verified."""
    return is_abstention_expressed(reply)

STOPWORDS_DISCARD = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "you", "your",
    "have", "has", "had", "can", "will", "would", "should", "not", "but", "about",
    "into", "over", "after", "before", "such", "than", "then", "them", "these",
    "what", "when", "where", "which", "while", "also", "some", "other", "their",
    "call", "help", "please", "karein", "karna", "hoga", "apne", "kisi", "sath",
})

# Generic institutional/boilerplate terms that appear in portal headers, footers, and contact disclaimers.
# These must NOT be used to claim that a specific document was utilized.
INSTITUTIONAL_GENERIC_TERMS = frozenset({
    "alwar", "police", "cyber", "control", "room", "national", "helpline",
    "official", "records", "available", "detail", "details", "information",
    "confirm", "contact", "directory", "station", "portal", "action", "actions",
    "sampark", "rajasthan", "8764874306", "1930", "112", "1090",
})

DETAILED_REQUEST_PATTERN = re.compile(
    r"\b(?:full|complete|detailed?|procedure|process|steps?|checklist|"
    r"step\s+by\s+step|how\s+to|pura|poora|पूरी|पूरा|विस्तार|प्रक्रिया|कदम)\b",
    re.IGNORECASE,
)


def requests_detailed_answer(query_text: str) -> bool:
    """Detect explicit requests that should receive the complete verified procedure."""
    return bool(DETAILED_REQUEST_PATTERN.search(query_text or ""))


def format_evidence_context(hits: list[dict[str, Any]]) -> str:
    """Format retrieved knowledge chunks into an authoritative, structured evidence block."""
    if not hits:
        return ""

    blocks: list[str] = ["[VERIFIED OFFICIAL KNOWLEDGE CONTEXT (Ground Truth)]:"]
    for i, h in enumerate(hits, 1):
        src = h.get("source", "official_kb")
        sec = h.get("section", "General")
        auth = h.get("authority", "official_government")
        juris = h.get("jurisdiction", "national")
        text = h.get("text", "").strip()

        blocks.append(
            f"--- Document [{i}]: {src} ---\n"
            f"Authority: {auth} | Jurisdiction: {juris} | Section: {sec}\n"
            f"{text}"
        )

    return "\n\n".join(blocks)


def format_grounding_directive(has_hits: bool, query_text: str, lang: str = "en") -> str:
    """Construct evidence-aware prompt instructions enforcing the three-way distinction and abstention rules."""
    if has_hits:
        length_rule = (
            "3. RESPONSE LENGTH: This is a detailed procedure request. Provide the complete verified "
            "numbered steps, keeping each step concise. Do not omit requested recovery or reporting steps.\n\n"
            if requests_detailed_answer(query_text)
            else "3. RESPONSE LENGTH: Max 2-3 short, clear sentences. Plain words, no markdown headings, no bullet lists.\n\n"
        )
        return (
            "[EVIDENCE GROUNDING & FACTUAL ATTRIBUTION RULES]:\n"
            "1. THREE-TIER INFORMATION DISTINCTION:\n"
            "   a. VERIFIED OFFICIAL FACTS: Answer factual questions using strictly the verified official documents above.\n"
            "   b. GENERAL CYBER-SAFETY GUIDANCE: If providing preventive cyber hygiene tips (e.g. passwords, 2FA, suspicious links), frame them clearly as general safety guidance rather than official statutory filing rules.\n"
            "   c. PARTIALLY SUPPORTED / MIXED QUESTIONS: If the user asks for multiple facts where some are in the verified context and others are NOT (e.g. asking for 1930 helpline AND personal mobile of SHO, or complaint procedure AND guaranteed 24-hour refund), answer the supported portion using verified evidence, and explicitly state that the unsupported detail (officer personal phone, fixed deadline, refund guarantee, filing fee) could not be verified from available official records.\n"
            "   d. UNVERIFIED / INSUFFICIENT EVIDENCE: If the user asks for a specific official fact that is NOT present in the verified context, state clearly that you cannot verify that specific detail from available official information. Direct them to confirm with Alwar Cyber Control Room at "
            f"{CONTROL_ROOM} or National Cyber Helpline 1930.\n"
            "2. ANTI-HALLUCINATION & HIGH-RISK SAFETY SAFEGUARDS:\n"
            "   - Cyber crime complaint filing on cybercrime.gov.in and at police stations has NO FEE (it is 100% free by law; never invent a fee).\n"
            "   - There is NO fixed time guarantee or 100% refund guarantee; recovery depends strictly on freezing funds during the golden hour.\n"
            "   - Never invent police contacts, personal officer numbers, legal provisions, deadlines, government schemes, or refund guarantees.\n"
            "   - Do not invent citations or cite documents you did not use.\n"
            "   - Use only the relevant document or documents for this incident. Do not mix steps from unrelated platforms or incident types.\n"
            "   - If the evidence contains an applicable step, include it; do not replace verified steps with generic advice.\n"
            f"{length_rule}"
            f"User Message:\n{query_text}"
        )

    return (
        "[OFFICIAL KNOWLEDGE SEARCH RESULT: No matching official documents found]\n"
        "1. NO MATCHING OFFICIAL RECORDS: You do NOT have verified knowledge base documents for this inquiry.\n"
        "2. ANTI-HALLUCINATION GUARD: Do NOT guess, fabricate, or hallucinate police names, police stations, legal sections, filing fees, deadlines, or refund guarantees.\n"
        "3. CYBER INCIDENT REPORTING: If this is an online fraud, hacking, harassment, or cyber emergency, communicate clearly that specific official records were not found and immediately direct the citizen to call 1930 or Alwar Cyber Control Room at "
        f"{CONTROL_ROOM}.\n"
        "4. GENERAL GUIDANCE: If the user asks for general safety advice, provide brief general cyber-safety guidance while clarifying that official case verification requires contacting 1930 or local police.\n"
        "5. OUT-OF-DOMAIN / OFF-TOPIC: If completely off-topic (cooking, jokes, general coding, sports, movies), politely state: 'I only assist with cyber crime reporting, online fraud, digital safety, and police procedures in Alwar.' Do NOT invent answers.\n\n"
        f"User Message:\n{query_text}"
    )


def verify_factual_claims_entailment(
    reply: str,
    candidate_hits: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Entailment-style factual verification:
    For each factual claim (phone, portal, deadline, fee), verify it exists
    in retrieved chunks or in the verified contacts table; otherwise drop or abstain.

    Returns:
        tuple[bool, list[str]]: (all_claims_entailed, list_of_unverified_claims)
    """
    unverified: list[str] = []
    reply_lower = reply.lower()

    # Aggregate all text from retrieved chunks
    chunks_text = " ".join(h.get("text", "") for h in candidate_hits).lower()

    # 1. Verify phone numbers
    for m in re.finditer(r"\b(?:\+91[\-\s]?)?([6-9]\d{9})\b", reply):
        num = m.group(1)
        # Skip transaction IDs/UTRs
        prefix = reply[max(0, m.start() - 25) : m.start()].lower()
        if any(tag in prefix for tag in ("utr", "txn", "transaction", "ack", "complaint id")):
            continue
        if num not in VERIFIED_PHONE_NUMBERS and num not in chunks_text:
            unverified.append(f"phone:{num}")

    # 2. Verify web domains & portals
    found_domains = re.findall(r"\b([a-z0-9\-\.]+\.(?:gov\.in|org\.in|org|in|com))\b", reply_lower)
    for dom in found_domains:
        if dom not in VERIFIED_DOMAINS and dom not in chunks_text:
            unverified.append(f"portal:{dom}")

    # 3. Verify fee claims (by law, cyber reporting is free)
    if re.search(r"\b(?:fee\s+of\s+rs|\d+\s*rupees?\s*fee|registration\s*charge)\b", reply_lower):
        if not re.search(r"\b(?:no\s+fee|free\s+of\s+cost|कोई\s+शुल्क\s+नहीं|निःशुल्क)\b", reply_lower):
            unverified.append("unverified_fee")

    # 4. Verify false refund guarantees
    if re.search(r"\b(?:100%\s*refund|guaranteed\s*refund)\b", reply_lower):
        if not re.search(r"\b(?:cannot\s+be\s+guaranteed|no\s+guarantee|गारंटी\s+नहीं)\b", reply_lower):
            unverified.append("unverified_refund_guarantee")

    return (len(unverified) == 0, unverified)


def extract_distinctive_terms(source: str, text: str, section: str) -> set[str]:
    """Extract distinguishing tokens and entities from a chunk to test for grounding in the response."""
    terms: set[str] = set()

    # Stem tokens from source filename (e.g. "mrm_portal_recovery.yaml" -> "mrm", "portal", "recovery")
    stem = re.sub(r"\.(?:yaml|yml|md)$", "", source.lower())
    for part in re.split(r"[-_]", stem):
        if len(part) >= 3 and part not in STOPWORDS_DISCARD and part not in INSTITUTIONAL_GENERIC_TERMS:
            terms.add(part)

    # Section title tokens
    for sec_token in tokenize(section):
        if len(sec_token) >= 3 and sec_token not in STOPWORDS_DISCARD and sec_token not in INSTITUTIONAL_GENERIC_TERMS:
            terms.add(sec_token)

    # Distinctive keywords and identifiers from text
    # 1. Domain-specific codes, portal names, and identifiers (excluding generic emergency helplines)
    special_identifiers = re.findall(
        r"\b(?:\*401\*|ceir|chakshu|stopncii|stopncii\.org|cybercrime\.gov\.in|sancharsaathi|anydesk|teamviewer|quicksupport|aeps|biometric|zero fir|bns|bnss|173|golden hour|kotwali|neemrana|behror|tijara|kishangarh)\b",
        text.lower(),
    )
    terms.update(special_identifiers)

    # 2. Substantive content tokens
    for token in tokenize(text):
        if len(token) >= 4 and token not in STOPWORDS_DISCARD and token not in INSTITUTIONAL_GENERIC_TERMS:
            terms.add(token)

    return terms


def is_chunk_utilized(reply: str, hit: dict[str, Any]) -> bool:
    """Check whether distinctive terms or concepts from the hit appear in the reply."""
    reply_lower = reply.lower()
    source = hit.get("source", "")
    section = hit.get("section", "")
    text = hit.get("text", "")

    # 1. Direct source stem match (e.g. "stopncii" in reply for stopncii source)
    stem_name = re.sub(r"\.(?:yaml|yml|md)$", "", source.lower())
    stem_parts = [
        p for p in re.split(r"[-_]", stem_name)
        if len(p) >= 3 and p not in STOPWORDS_DISCARD and p not in INSTITUTIONAL_GENERIC_TERMS
    ]

    # Special case handling for known unique topics:
    if "stopncii" in stem_name and ("stopncii" in reply_lower or "intimate" in reply_lower or "morphed" in reply_lower):
        return True
    if "call_forwarding" in stem_name and ("401" in reply_lower or "forwarding" in reply_lower):
        return True
    if "mrm" in stem_name and ("mrm" in reply_lower or "lien" in reply_lower or ("freeze" in reply_lower and "unfreeze" in reply_lower)):
        return True
    if "digital_arrest" in stem_name and ("digital arrest" in reply_lower or "skype" in reply_lower or "cbi" in reply_lower):
        return True
    if "apk" in stem_name and ("apk" in reply_lower or "malware" in reply_lower or "airplane" in reply_lower):
        return True
    if "aeps" in stem_name and ("aeps" in reply_lower or "biometric" in reply_lower or "fingerprint" in reply_lower or "aadhaar" in reply_lower):
        return True
    if "loan_app" in stem_name and ("loan" in reply_lower or "blackmail" in reply_lower or "recovery" in reply_lower):
        return True
    if "upi" in stem_name and ("upi" in reply_lower or "गोल्डन" in reply_lower or "golden" in reply_lower or "होल्ड" in reply_lower or "freeze" in reply_lower or "फ्रीज" in reply_lower or "beneficiary" in reply_lower):
        return True
    platform_aliases = {
        "facebook": ("facebook", "fb"),
        "instagram": ("instagram", "insta"),
        "telegram": ("telegram", "टेलीग्राम"),
        "whatsapp": ("whatsapp", "व्हाट्सएप"),
    }
    for platform, aliases in platform_aliases.items():
        if platform in stem_name and not any(alias in reply_lower for alias in aliases):
            return False

    distinctive_terms = extract_distinctive_terms(source, text, section)
    reply_tokens = set(tokenize(reply))

    # Check for direct overlap with distinctive terms
    overlap = distinctive_terms & reply_tokens
    # High-signal entities (portal domains, legal sections, scam markers) need only 1 match
    # Note: Generic emergency numbers (1930, 112, 8764874306) are intentionally excluded
    # because they appear in standard disclaimers and do not prove source utilization.
    high_signal = {"*401*", "ceir", "chakshu", "stopncii", "cybercrime", "aeps", "biometric", "utr"}
    if any(k in overlap for k in high_signal):
        return True

    # Check if multiple distinctive tokens overlap
    if len(overlap) >= 2:
        return True

    # If the reply mentions the stem part and has at least one content token
    if any(sp in reply_lower for sp in stem_parts) and len(overlap) >= 1:
        return True

    return False


def attribute_evidence(
    reply: str,
    candidate_hits: list[dict[str, Any]],
    domains: list[str],
    lang: str = "en",
    tools_used: list[str] | None = None,
) -> tuple[list[str], str]:
    """Determine verified source attribution and generate a safe citation note.

    Returns:
        tuple[list[str], str]: (sorted_sources, citation_note)
    """
    clean_reply = (reply or "").strip()
    tools = tools_used or []

    # 1. Fallback or empty reply
    if not clean_reply or clean_reply.startswith("Sorry, I could not answer that"):
        return [], ""

    # 2. Pure conversation
    if domains == ["conversation"] and not candidate_hits and not tools:
        return [], ""

    # 3. Check candidate hits for actual utilization in reply
    utilized_sources: set[str] = set()
    for hit in candidate_hits:
        if is_chunk_utilized(clean_reply, hit):
            src = hit.get("source", "").strip()
            if src:
                utilized_sources.add(src)

    # 4. Attributed verified sources from KB (retaining source attribution even if a secondary limitation was noted)
    if utilized_sources:
        sorted_sources = sorted(utilized_sources)
        sources_str = ", ".join(sorted_sources)
        if is_unverified_limitation_expressed(clean_reply):
            note = (
                f"आधिकारिक ज्ञानकोष से सत्यापित: {sources_str}। कुछ विशिष्ट विवरण उपलब्ध आधिकारिक रिकॉर्ड से सत्यापित नहीं हो सके।"
                if lang in ("hi", "hinglish")
                else f"Verified from official knowledge base: {sources_str}. Additional specific details could not be verified from available official records."
            )
        else:
            note = (
                f"आधिकारिक ज्ञानकोष से सत्यापित: {sources_str}."
                if lang in ("hi", "hinglish")
                else f"Verified from official knowledge base: {sources_str}."
            )
        return sorted_sources, note

    # 5. Explicit limitation: specific detail could not be verified and no verified sources were utilized
    if is_unverified_limitation_expressed(clean_reply):
        note = (
            "उपलब्ध आधिकारिक रिकॉर्ड से विशिष्ट विवरण सत्यापित नहीं हो सके। कृपया अलवर साइबर कंट्रोल रूम (8764874306) या 1930 से पुष्टि करें।"
            if lang in ("hi", "hinglish")
            else f"Specific details could not be verified from available official records. Please confirm with Alwar Cyber Control Room ({CONTROL_ROOM}) or 1930."
        )
        return [], note

    # 6. Check tool-based official verification (requiring tool invocation AND reply content utilization)
    if "get_complaint_guide" in tools and any(
        k in clean_reply.lower() for k in ("guide", "checklist", "step", "document", "process", "प्रक्रिया", "कदम", "गाइड", "चेकलिस्ट")
    ):
        note = (
            "चेकलिस्ट आधिकारिक राजस्थान पुलिस एवं I4C शिकायत प्रक्रिया से सत्यापित है।"
            if lang in ("hi", "hinglish")
            else "Checklist verified from official Rajasthan Police and I4C complaint procedures."
        )
        return ["official_complaint_guide"], note

    if "find_police_station" in tools and any(
        k in clean_reply.lower() for k in ("thana", "police station", "station", "थाना", "sho", "circle", "kotwali", "aravali", "neemrana", "behror", "kishangarh")
    ):
        note = (
            "पुलिस थाना विवरण आधिकारिक अलवर जिला पुलिस निर्देशिका से सत्यापित है।"
            if lang in ("hi", "hinglish")
            else "Police station details verified from official Alwar District Police directory."
        )
        return ["alwar_police_stations_registry"], note

    if "draft_complaint_letter" in tools and any(
        k in clean_reply.lower() for k in ("letter", "application", "complaint", "draft", "आवेदन", "प्रार्थना", "पत्र", "शिकायत")
    ):
        note = (
            "शिकायत पत्र आधिकारिक राजस्थान पुलिस प्रारूप में तैयार किया गया है।"
            if lang in ("hi", "hinglish")
            else "Formal complaint letter drafted in official Rajasthan Police format."
        )
        return ["rajasthan_police_complaint_format"], note

    # 7. General cyber-safety guidance
    is_general_safety = (
        "cyber_safety" in domains
        or any(k in clean_reply.lower() for k in ("password", "2fa", "two-factor", "otp", "phishing", "suspicious link", "pin", "पासवर्ड", "ओटीपी"))
    )
    if is_general_safety:
        note = (
            "सामान्य साइबर सुरक्षा दिशानिर्देश (विशिष्ट आधिकारिक रिकॉर्ड से असत्यापित)।"
            if lang in ("hi", "hinglish")
            else "General cyber-safety guidance (unverified against specific statutory/official filing records)."
        )
        return [], note

    # 8. No matching official records found in KB
    if not candidate_hits:
        note = (
            "आधिकारिक ज्ञानकोष में कोई संबंधित रिकॉर्ड नहीं मिला।"
            if lang == "hi"
            else "No matching official records found in knowledge base."
        )
        return [], note

    # 9. Default unverified fallback
    note = (
        "उपलब्ध आधिकारिक स्रोतों से विशिष्ट तथ्य असत्यापित हैं।"
        if lang == "hi"
        else "Information not verified against specific official knowledge base records."
    )
    return [], note
