"""Runtime Acceptance Test Runner for Sub-phase B3.1.
Executes all functional scenarios through the live proxy at http://localhost:3000/api/chat
and tests /api/voice and /api/tts.
"""

import io
import json
import time
import urllib.request
import urllib.error
import uuid
import wave

BASE_URL = "http://localhost:3000"

def send_chat(message: str, session_id: str = None, lang: str = "en") -> dict:
    if not session_id:
        session_id = str(uuid.uuid4())
    url = f"{BASE_URL}/api/chat"
    payload = json.dumps({"session_id": session_id, "message": message, "lang": lang}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode("utf-8"))
            data["_status"] = resp.status
            data["_elapsed"] = round(elapsed, 2)
            return data
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        body = e.read().decode("utf-8")
        return {"_status": e.code, "_error": body, "_elapsed": round(elapsed, 2)}
    except Exception as e:
        elapsed = time.time() - start
        return {"_status": 500, "_error": str(e), "_elapsed": round(elapsed, 2)}

def test_tts(text: str) -> dict:
    url = f"{BASE_URL}/api/tts"
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            elapsed = time.time() - start
            data = resp.read()
            return {
                "_status": resp.status,
                "content_type": resp.headers.get("Content-Type"),
                "size_bytes": len(data),
                "is_wav": data[:4] == b"RIFF",
                "_elapsed": round(elapsed, 2),
            }
    except Exception as e:
        return {"_status": 500, "_error": str(e), "_elapsed": round(time.time() - start, 2)}

def test_voice_endpoint() -> dict:
    url = f"{BASE_URL}/api/voice"
    # Create a 1-second silence WAV at 16000Hz, 16-bit mono
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    wav_bytes = buf.getvalue()

    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = io.BytesIO()
    
    # audio field
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="audio"; filename="speech.wav"\r\n')
    body.write(b'Content-Type: audio/wav\r\n\r\n')
    body.write(wav_bytes)
    body.write(b"\r\n")

    # session_id field
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="session_id"\r\n\r\n')
    body.write(str(uuid.uuid4()).encode("utf-8"))
    body.write(b"\r\n")

    # lang field
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="lang"\r\n\r\n')
    body.write(b"en\r\n")

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    payload = body.getvalue()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            elapsed = time.time() - start
            data = json.loads(resp.read().decode("utf-8"))
            data["_status"] = resp.status
            data["_elapsed"] = round(elapsed, 2)
            return data
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        body_text = e.read().decode("utf-8")
        return {"_status": e.code, "_error": body_text, "_elapsed": round(elapsed, 2)}
    except Exception as e:
        elapsed = time.time() - start
        return {"_status": 500, "_error": str(e), "_elapsed": round(elapsed, 2)}

TEST_SCENARIOS = {
    "A_basic_supported": [
        ("How can I report an online financial fraud?", "en"),
        ("What should I do if someone creates a fake social media profile?", "en"),
        ("How can I report cyber harassment?", "en"),
        ("साइबर फ्रॉड होने पर मुझे क्या करना चाहिए?", "hi"),
        ("अगर किसी ने मेरे नाम से फेक इंस्टाग्राम अकाउंट बनाया है तो क्या करूँ?", "hi"),
    ],
    "B_deterministic_abstention": [
        ("What is the personal mobile number of SHO Alwar Kotwali?", "en"),
        ("Within how many hours is the police required by law to resolve my cyber complaint?", "en"),
        ("Will the accused person definitely be convicted and denied bail for sure?", "en"),
        ("Is my 50000 rupees 100% guaranteed to be recovered within 24 hours?", "en"),
        ("What is the official registration fee to file an FIR at the cyber police station?", "en"),
    ],
    "C_mixed_support": [
        ("What is the 1930 helpline procedure and can you give me the personal cell phone of the Alwar SP?", "en"),
        ("How do I report a cyber crime on cybercrime.gov.in and what is the strict 24 hour deadline for police to finish the case?", "en"),
        ("How can I report wrong UPI transfer within golden hour and do you give 100% money back guarantee?", "en"),
    ],
    "D_out_of_domain": [
        ("How do I write a Python function to invert a binary tree?", "en"),
        ("Can you recommend a good Italian restaurant in Alwar?", "en"),
        ("Who won the Best Picture award at the Oscars this year?", "en"),
    ],
    "E_language_handling": [
        ("Report fake profile on facebook", "en"),
        ("फेक फेसबुक प्रोफाइल की रिपोर्ट कैसे करें?", "hi"),
        ("Meri fake profile banayi hai facebook par kaise report karu?", "hi"),
    ],
}

if __name__ == "__main__":
    results = {}
    print("=== STARTING RUNTIME ACCEPTANCE TESTS ===", flush=True)
    
    for category, queries in TEST_SCENARIOS.items():
        print(f"\n--- Running Category: {category} ---", flush=True)
        results[category] = []
        for query, lang in queries:
            print(f"\nQuery [{lang}]: {query}", flush=True)
            res = send_chat(query, lang=lang)
            print(f"Status: {res.get('_status')} ({res.get('_elapsed')}s)", flush=True)
            if res.get("reply"):
                print(f"Reply: {res.get('reply')[:140]}...", flush=True)
            else:
                print(f"Error: {res.get('_error')}", flush=True)
            print(f"Sources: {res.get('sources')}", flush=True)
            print(f"Citation Note: {res.get('citation_note')}", flush=True)
            results[category].append({
                "query": query,
                "lang": lang,
                "result": res
            })

    print("\n--- Running Voice TTS Test ---", flush=True)
    tts_res = test_tts("Aap sabse pehle 1930 par call karein aur Alwar Cyber Control Room se sampark karein.")
    print("TTS Result:", tts_res, flush=True)
    results["tts"] = tts_res

    print("\n--- Running Voice STT/Audio Upload Test ---", flush=True)
    voice_res = test_voice_endpoint()
    print("Voice Endpoint Result:", voice_res, flush=True)
    results["voice_upload"] = voice_res

    with open("reports/runtime_acceptance_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== ALL RESULTS SAVED TO reports/runtime_acceptance_results.json ===", flush=True)
