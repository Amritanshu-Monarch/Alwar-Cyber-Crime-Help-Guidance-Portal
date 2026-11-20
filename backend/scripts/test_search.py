import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.rag import get_kb

def main():
    kb = get_kb()
    print(f"KB count: {kb.count()}")
    queries = [
        "mere paise freeze ho gaye mrm portal se kaise milenge",
        "Someone dressed as police on video call digital arrest",
        "how to stop call forwarding *401*",
        "StopNCII remove morphed photos intimate image",
        "Instagram hacked recovery video selfie",
        "social media grievance officer india reporting links",
        "instant loan app blackmailing contacts",
        "AePS biometric fingerprint money withdrawal fraud",
        "sent money to wrong upi id by mistake reversal",
        "bijli bill katne ka fake message aaya",
    ]
    for q in queries:
        hits = kb.search(q, k=2)
        if hits:
            top = f"{hits[0]['source']} | {hits[0]['section']} ({hits[0]['domain']})"
        else:
            top = "NO HIT"
        print(f"Q: {q[:50]:50} -> {top}")

if __name__ == "__main__":
    main()
