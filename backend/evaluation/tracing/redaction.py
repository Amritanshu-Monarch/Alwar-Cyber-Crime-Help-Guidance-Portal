"""Data Privacy & Security Redaction Engine.

Sanitizes citizen personal information, banking credentials, Aadhaar, OTPs,
and passwords before traces or evaluation logs are persisted.
Preserves official emergency helplines and verified police numbers.
"""

import re
from typing import Any

from app.agent.facts import CONTROL_ROOM

OFFICIAL_VERIFIED_NUMBERS = frozenset({
    "1930",
    "112",
    "1090",
    CONTROL_ROOM,  # "8764874306"
    "0144-2337456",
    "01442337456",
    "2337456",
})

# Patterns for sensitive citizen information
CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
OTP_PATTERN = re.compile(
    r"\b(?:otp|one[- ]time[- ]password|pin|code)(?:\s+(?:is|was|hai))?\s*[:=-]?\s*([0-9]{4,8})\b",
    re.IGNORECASE,
)
PASSWORD_PATTERN = re.compile(
    r"\b(?:password|passwd|pwd)(?:\s+(?:is|was|hai))?\s*[:=-]?\s*([^\s,;]+)\b",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"\b(?:\+91[\-\s]?)?([6-9]\d{9})\b")
API_KEY_PATTERN = re.compile(r"\b(?:AIza[0-9A-Za-z-_]{35}|ghp_[0-9A-Za-z]{36}|sk-[0-9A-Za-z]{24,})\b")


def redact_text(text: str) -> str:
    """Mask sensitive PII and financial credentials in text."""
    if not text:
        return ""

    s = text

    # 1. API Keys & tokens
    s = API_KEY_PATTERN.sub("[REDACTED_API_KEY]", s)

    # 2. Credit/Debit Card Numbers
    s = CARD_PATTERN.sub("[REDACTED_CARD]", s)

    # 3. Aadhaar Numbers (12-digit)
    s = AADHAAR_PATTERN.sub("[REDACTED_AADHAAR]", s)

    # 4. OTPs & PINs
    s = OTP_PATTERN.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED_OTP]"), s)

    # 5. Passwords
    s = PASSWORD_PATTERN.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED_PASSWORD]"), s)

    # 6. Citizen Mobile Numbers (excluding verified government helplines)
    def _replace_phone(m: re.Match) -> str:
        num = m.group(1)
        if num in OFFICIAL_VERIFIED_NUMBERS:
            return m.group(0)
        # Also check if preceded by UTR or transaction ID
        start = m.start()
        prefix = s[max(0, start - 15) : start].lower()
        if any(tag in prefix for tag in ("utr", "txn", "transaction", "ack")):
            return m.group(0)
        return "[REDACTED_PHONE]"

    s = PHONE_PATTERN.sub(_replace_phone, s)

    return s


def redact_record_data(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact dictionary objects containing trace or dataset attributes."""
    cleaned: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            cleaned[k] = redact_text(v)
        elif isinstance(v, dict):
            cleaned[k] = redact_record_data(v)
        elif isinstance(v, list):
            cleaned[k] = [
                redact_record_data(item) if isinstance(item, dict)
                else (redact_text(item) if isinstance(item, str) else item)
                for item in v
            ]
        else:
            cleaned[k] = v
    return cleaned

