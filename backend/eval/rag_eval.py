"""Live retrieval eval: `python eval/rag_eval.py` (needs GOOGLE_API_KEY).

In-scope questions must retrieve the expected section in the top results.
Off-topic questions must return nothing, so the agent refuses instead of improvising."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag import get_kb  # noqa: E402

# (query as the agent would write it, expected source filename, substring of expected section title)
IN_SCOPE = [
    ("what should I do immediately after UPI fraud", "upi-fraud.md", "Immediate"),
    ("steps to register complaint on cybercrime.gov.in portal", "cybercrime-portal.md", "Registration"),
    ("will I get my money back from bank RBI liability", "verified-current-laws.md", "RBI"),
    ("what is zero FIR", "police-reporting.md", "Zero FIR"),
    ("police not registering my complaint", "police-reporting.md", "Refuse"),
    ("documents needed for cyber crime complaint", "evidence-preservation.md", "Evidence"),
    ("what to tell when calling 1930", "national-helplines.md", "1930"),
    ("digital arrest video call police threat", "bank-transfer-fraud.md", "Digital Arrest"),
    ("someone asked me to install AnyDesk", "malicious-apk.md", "Remote"),
    ("loan app blackmail morphed photos contacts", "morphed-media.md", "Photo Morphing"),
    ("sextortion video call blackmail", "sextortion.md", "Video Call"),
    ("instagram account hacked", "instagram-hacking.md", "Compromised"),
    ("part time job task scam deposit", "job-fraud.md", "Task Scam"),
    ("how to block lost mobile phone IMEI", "lost-mobile.md", "CEIR"),
    ("check mobile connections registered in my name", "suspicious-calls-sms.md", "TAFCOP"),
    ("report suspicious SMS WhatsApp number Chakshu", "suspicious-calls-sms.md", "Chakshu"),
    ("how to make strong password two factor authentication", "safety-essentials.md", "Password"),
    ("how to keep children safe online", "safety-essentials.md", "Children"),
    ("is it safe to delete the scam chat", "evidence-preservation.md", "Evidence"),
    ("where is the alwar cyber police station located", "alwar-cyber-unit.md", "Police Station"),
    ("which police station covers kala kuan", "locality-jurisdiction.md", "Locality"),
]

OFF_TOPIC = [
    "who won the cricket world cup",
    "write a python function to sort a list",
    "what is the capital of France",
    "give me a recipe for paneer butter masala",
    "explain quantum computing",
    "best movies of 2025",
    "how to lose weight fast",
    "tell me a joke",
]


def main() -> int:
    kb = get_kb()
    kb.ensure_fresh()
    hits_ok = 0
    print("== in scope (expect expected section in results)")
    for q, src, sec in IN_SCOPE:
        res = kb.search(q)
        ok = any(r["source"] == src and sec.lower() in r["section"].lower() for r in res)
        hits_ok += ok
        top = f"{res[0]['source']} > {res[0]['section']}" if res else "NO RESULTS"
        print(f"{'PASS' if ok else 'FAIL'}  {q!r:70} top: {top}")
    print("\n== off topic (expect no results)")
    refused = 0
    for q in OFF_TOPIC:
        res = kb.search(q)
        ok = not res
        refused += ok
        print(f"{'PASS' if ok else 'FAIL'}  {q!r:50} got {len(res)} results" + ("" if ok else f" top: {res[0]['section']}"))
    print(f"\nretrieval hit rate: {hits_ok}/{len(IN_SCOPE)}   off-topic rejected: {refused}/{len(OFF_TOPIC)}")
    return 0 if hits_ok == len(IN_SCOPE) and refused == len(OFF_TOPIC) else 1


if __name__ == "__main__":
    raise SystemExit(main())
