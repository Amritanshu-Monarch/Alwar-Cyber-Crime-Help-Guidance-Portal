"""Live retrieval eval: `uv run python eval/rag_eval.py` (needs GOOGLE_API_KEY).

In-scope questions must retrieve the expected section in the top results.
Off-topic questions must return nothing, so the agent refuses instead of improvising."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag import get_kb  # noqa: E402

# (query as the agent would write it, expected source, substring of expected section title)
IN_SCOPE = [
    ("what should I do immediately after UPI fraud", "reporting-process.md", "First 10 minutes"),
    ("steps to register complaint on cybercrime.gov.in portal", "reporting-process.md", "National Cyber Crime Reporting Portal"),
    ("will I get my money back from bank RBI liability", "reporting-process.md", "bank"),
    ("what is zero FIR", "reporting-process.md", "Zero FIR"),
    ("police not registering my complaint", "complaint-details.md", "do not register"),
    ("documents needed for cyber crime complaint", "complaint-details.md", "Documents"),
    ("what to tell when calling 1930", "complaint-details.md", "1930"),
    ("how to write complaint application to SHO", "complaint-details.md", "write a complaint"),
    ("digital arrest video call police threat", "fraud-types.md", "Digital arrest"),
    ("someone asked me to install AnyDesk", "fraud-types.md", "Remote access"),
    ("loan app blackmail morphed photos contacts", "fraud-types.md", "Loan app"),
    ("sextortion video call blackmail", "fraud-types.md", "Sextortion"),
    ("instagram account hacked", "fraud-types.md", "hacking"),
    ("part time job task scam deposit", "fraud-types.md", "Job"),
    ("how to block lost mobile phone IMEI", "applications-and-services.md", "lost or stolen"),
    ("check mobile connections registered in my name", "applications-and-services.md", "mobile connections"),
    ("report suspicious SMS WhatsApp number Chakshu", "applications-and-services.md", "Chakshu"),
    ("tenant verification documents Rajasthan", "applications-and-services.md", "Tenant"),
    ("website hacked ransomware report CERT-In", "applications-and-services.md", "organisation"),
    ("how to make strong password two factor authentication", "cyber-safety.md", "password"),
    ("how to keep children safe online", "cyber-safety.md", "children"),
    ("is it safe to delete the scam chat", "reporting-process.md", "evidence"),
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
