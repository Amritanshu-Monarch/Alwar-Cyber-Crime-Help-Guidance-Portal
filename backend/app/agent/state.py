"""Per-session case state management and triage logic.

Tracks:
- case_type (financial_fraud, deepfake, sextortion, account_hacking, lost_phone, loan_harassment, etc.)
- money_lost (bool)
- amount (str | None)
- time_since_incident (str | None)
- payment_method (upi, card, net_banking, aeps)
- platform (whatsapp, instagram, telegram, olx, skype, call, sms)
- language (en, hi, hinglish)
- police_station_area (str | None)
- steps_already_done (list of completed actions e.g. 'called_1930', 'blocked_card')
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.agent.facts import CONTROL_ROOM


@dataclass
class SessionCaseState:
    session_id: str
    case_type: str | None = None
    money_lost: bool | None = None
    amount: str | None = None
    time_since_incident: str | None = None
    payment_method: str | None = None
    platform: str | None = None
    language: str = "en"
    police_station_area: str | None = None
    steps_already_done: list[str] = field(default_factory=list)
    triage_question_asked: bool = False

    def to_context_summary(self) -> str:
        """Format state into an informative context directive for the model."""
        parts = []
        if self.case_type:
            parts.append(f"Case Type: {self.case_type}")
        if self.money_lost is not None:
            parts.append(f"Money Lost: {'Yes' if self.money_lost else 'No'}")
        if self.amount:
            parts.append(f"Amount: {self.amount}")
        if self.payment_method:
            parts.append(f"Payment Method: {self.payment_method}")
        if self.time_since_incident:
            parts.append(f"Time: {self.time_since_incident}")
        if self.platform:
            parts.append(f"Platform: {self.platform}")
        if self.steps_already_done:
            done_str = ", ".join(self.steps_already_done)
            parts.append(f"Steps Already Completed by Citizen: [{done_str}] (CRITICAL: Do NOT instruct them to repeat these steps)")
        if not parts:
            return ""
        return "[ACTIVE CITIZEN CASE STATE]:\n" + "\n".join(parts)


AMOUNT_PATTERN = re.compile(
    r"\b(?:rs\.?|rupees?|inr|रुपये|रूपए|पैसे)?\s*([1-9]\d{2,6})\s*(?:rs\.?|rupees?|inr|रुपये|रूपए|पैसे)?\b",
    re.IGNORECASE,
)

DONE_1930_PATTERN = re.compile(
    r"\b(?:1930\s*(?:pe|par|per)?\s*(?:call|shikayat|complaint)?\s*(?:kar\s*diya|kar\s*di|kar\s*chuka|kiya\s*tha|called|reported)|already\s*called\s*1930)\b",
    re.IGNORECASE,
)
DONE_CARD_PATTERN = re.compile(
    r"\b(?:card|atm)\s*(?:block|band)\s*(?:karwa|kar|karwaya|blocked)\b",
    re.IGNORECASE,
)
DONE_SIM_PATTERN = re.compile(
    r"\b(?:sim)\s*(?:block|band)\s*(?:karwa|kar|karwaya|blocked)\b",
    re.IGNORECASE,
)
DONE_POLICE_PATTERN = re.compile(
    r"\b(?:thana|police\s*station)\s*(?:gaya|gaye|report|complaint\s*de\s*di)\b",
    re.IGNORECASE,
)


def extract_and_update_state(text: str, state: SessionCaseState, effective_lang: str = "en") -> SessionCaseState:
    """Analyze citizen input and incrementally update session state."""
    state.language = effective_lang
    q_lower = text.lower()

    # 1. Detect completed steps to avoid redundant advice
    if DONE_1930_PATTERN.search(text):
        if "called_1930" not in state.steps_already_done:
            state.steps_already_done.append("called_1930")

    if DONE_CARD_PATTERN.search(text):
        if "blocked_card" not in state.steps_already_done:
            state.steps_already_done.append("blocked_card")

    if DONE_SIM_PATTERN.search(text):
        if "blocked_sim" not in state.steps_already_done:
            state.steps_already_done.append("blocked_sim")

    if DONE_POLICE_PATTERN.search(text):
        if "visited_police_station" not in state.steps_already_done:
            state.steps_already_done.append("visited_police_station")

    # 2. Detect platforms
    for plat in ("whatsapp", "telegram", "instagram", "facebook", "skype", "olx", "gmail"):
        if plat in q_lower:
            state.platform = plat.capitalize()
            break

    # 3. Detect payment method
    if "upi" in q_lower or "gpay" in q_lower or "phonepe" in q_lower or "paytm" in q_lower:
        state.payment_method = "UPI"
    elif "card" in q_lower or "atm" in q_lower or "debit" in q_lower or "credit" in q_lower:
        state.payment_method = "Card"
    elif "aeps" in q_lower or "biometric" in q_lower or "बायोमेट्रिक" in text:
        state.payment_method = "AePS Biometric"
    elif "net banking" in q_lower or "bank transfer" in q_lower or "khate" in q_lower:
        state.payment_method = "Net Banking"

    # 4. Detect financial loss & amount
    m_amt = AMOUNT_PATTERN.search(text)
    if m_amt:
        state.amount = f"Rs {m_amt.group(1)}"
        state.money_lost = True

    if any(k in q_lower for k in ("paise kat", "paise cut", "money lost", "rupaye kat", "fraud ho gaya", "debited", "पैसे कट", "रुपये कट")):
        state.money_lost = True

    # 5. Detect case type
    if any(k in q_lower for k in ("deepfake", "morph", "डीपफेक", "मॉर्फ")):
        state.case_type = "deepfake"
    elif any(k in q_lower for k in ("sextortion", "video call blackmail", "सेक्सटॉर्शन")):
        state.case_type = "sextortion"
    elif any(k in q_lower for k in ("hack", "takeover", "हैक", "कब्जा")):
        state.case_type = "account_hacking"
    elif any(k in q_lower for k in ("loan app", "लोन ऐप")):
        state.case_type = "loan_app_harassment"
    elif any(k in q_lower for k in ("chori", "lost phone", "stolen phone", "ceir", "फोन चोरी")):
        state.case_type = "lost_phone"
    elif any(k in q_lower for k in ("digital arrest", "डिजिटल अरेस्ट")):
        state.case_type = "digital_arrest"
    elif state.money_lost or any(k in q_lower for k in ("upi", "otp", "bank", "paise", "कट गए")):
        state.case_type = "financial_fraud"

    # 6. Detect time since incident
    if any(k in q_lower for k in ("abhi", "just now", "5 minute", "10 minute", "aaj", "today", "yesterday", "कल", "आज", "अभी")):
        state.time_since_incident = "within_golden_hour" if any(k in q_lower for k in ("abhi", "just now", "minute", "अभी")) else "recent"

    return state


def needs_triage_question(state: SessionCaseState) -> bool:
    """Return True if this is a fresh financial fraud incident missing critical triage details."""
    if state.triage_question_asked:
        return False
    # If money was lost but both time and payment method are completely unknown
    if state.money_lost and not state.payment_method and not state.time_since_incident:
        return True
    return False


def get_triage_reply(state: SessionCaseState, lang: str = "en") -> str:
    """Generate the immediate urgent safe action alongside the ONE essential triage question."""
    state.triage_question_asked = True
    if lang == "hi":
        return (
            "सबसे पहले तुरंत 1930 पर कॉल करें ताकि ठग के खाते में रकम होल्ड कराई जा सके। "
            "पैसे कितनी देर पहले कटे, और यह लेनदेन यूपीआई से हुआ या कार्ड से?"
        )
    if lang == "hinglish":
        return (
            "Sabse pehle turant National Cyber Helpline 1930 par call karein taaki fraudster ka account freeze karwaya ja sake. "
            "Paise kitne der pehle kate, aur transaction UPI se hua ya card se?"
        )
    return (
        "First, call the National Cyber Helpline 1930 immediately to freeze the fraudster's beneficiary account. "
        "How long ago was the money debited, and was the transaction via UPI or card?"
    )
