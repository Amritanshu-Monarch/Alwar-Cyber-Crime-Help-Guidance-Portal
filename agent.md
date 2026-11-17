# 🛡️ Alwar Cyber Crime Guidance Agent — Architecture & Specification (`agent.md`)

This document provides a comprehensive technical overview of the AI Voice and Chat Agent powering the **Alwar Cyber Crime Help & Guidance Portal** (Rajasthan, India). It documents the multi-provider LLM/speech architecture, conversational state & triage mechanics, calibrated scope guardrails, domain-aware hybrid RAG pipeline, context-aware anti-hallucination sanitizers, and golden dataset evaluation.

---

## 1. System Overview & Core Architecture

The agent is an empathetic, safety-first, bilingual voice-and-text guidance assistant built with **FastAPI**, **Google Agent Development Kit (ADK)**, and a modular **Multi-Provider Architecture** supporting **Google Gemini**, **Sarvam AI** (Indian Indic LLM/voice), **Groq/OpenRouter** (OpenAI-compatible), and **Local Ollama**. It provides citizens of Alwar with instant cyber emergency guidance, actionable complaint checklists, police station mapping, and formal complaint letter drafting.

```
 Citizen (Voice / Text)
       │
       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Frontend UI (index.html / Vanilla JS + Web Audio API)                  │
 │ • Voice Recorder (16kHz PCM / WAV recording & streaming playback)      │
 │ • Real-time Multilingual Support (English / Hindi / Hinglish)          │
 │ • Interactive Cards (Complaint Checklists, Letter Drafter, Thana Maps) │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ HTTP POST (/api/chat, /api/voice)
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ FastAPI Gateway (backend/app/main.py)                                  │
 │ • CORS, Rate Limiting (30 req/min), Request Validation (Pydantic)      │
 │ • Provider Health Endpoint (/api/health)                               │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Agent Service Orchestrator (backend/app/agent/runner.py)               │
 │                                                                        │
 │  1. Multilingual Detection (English, Devanagari Hindi, Roman Hinglish) │
 │  2. 3-Way Intent Classifier (in_scope, out_of_scope, clarify)          │
 │  3. Session Case State Tracker & Triage Step (state.py)                │
 │  4. Controlled Hybrid RAG (rag.py: BM25 + ChromaDB + RRF + Reranker)   │
 │  5. Provider Registry & Automatic Cascading Fallback (providers/)      │
 │     • Order: Gemini -> Sarvam AI -> Groq / OpenRouter -> Local Ollama  │
 │  6. Structured Answer Ordering (Urgent Action -> 2-3 Steps -> Checklist)│
 │  7. Context-Aware High-Risk Sanitizer (abstention.py)                  │
 │  8. Entailment-Style Grounding & Citation Validation (grounding.py)    │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │  Speech Pipeline  │               │ RAG Vector Store  │
        │ • STT: Sarvam/Gem │               │ • ChromaDB        │
        │ • TTS: Sarvam/Gem │               │ • BM25 + RRF (k=4)│
        └───────────────────┘               └───────────────────┘
```

---

## 2. Multi-Provider Architecture (`backend/app/providers/`)

To guarantee high availability and sovereign Indian AI integration, all LLM, speech-to-text (STT), and text-to-speech (TTS) services are abstracted behind unified provider interfaces:

```
                  ┌─────────────────────────────────────┐
                  │       ProviderRegistry               │
                  │ (Automatic Fallback on 429/5xx/T.O.)│
                  └──────────────────┬──────────────────┘
                                     │
         ┌──────────────┬────────────┴───┬──────────────┬──────────────┐
         ▼              ▼                ▼              ▼              ▼
   GeminiProvider  SarvamProvider   GroqProvider   OpenRouterProvider  OllamaProvider
   (Primary ADK)   (Indic Chat,     (Llama-3.3     (DeepSeek / Mistral  (Local Offline,
                   Saaras STT,       Ultra-Fast)    Free Tier)           Deterministic)
                   Bulbul TTS)
```

### Provider Matrix

| Provider | Type | Implementation Class | Key Capabilities & Notes |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | LLM, STT, TTS | `GeminiLLMProvider`, `GeminiSTTProvider`, `GeminiTTSProvider` | `gemini-2.5-flash` primary model; ADK tool execution; `gemini-2.5-flash-lite` fallback. |
| **Sarvam AI** | LLM, STT, TTS | `SarvamLLMProvider`, `SarvamSTTProvider`, `SarvamTTSProvider` | Specialized Indic voice & language models (`saaras:v2` STT, `bulbul:v2` TTS, `sarvam-2b/8b` chat). |
| **Groq** | LLM | `GroqProvider` | Ultra-fast OpenAI-compatible inference (`llama-3.3-70b-versatile`). |
| **OpenRouter** | LLM | `OpenRouterProvider` | Multi-model free-tier routing (`mistralai/mistral-7b-instruct:free`). |
| **Local Ollama** | LLM | `OllamaProvider` | 100% offline local inference (`http://localhost:11434`, model from `OLLAMA_MODEL`). **Deterministic code routing**: does NOT rely on small-model tool calling; routes KB chunks and contacts in code. |

### Configuration via `.env`
```ini
LLM_PROVIDER_ORDER=gemini,sarvam,groq,ollama
STT_PROVIDER=sarvam
TTS_PROVIDER=sarvam
OLLAMA_MODEL=llama3.2:3b
SARVAM_API_KEY=your_sarvam_key_here
GROQ_API_KEY=your_groq_key_here
```

### Reachability Health Check (`/api/health`)
Returns live health checks for each configured provider:
```json
{
  "status": "ok",
  "providers": {
    "gemini": {"status": "reachable", "details": "Gemini 2.5 Flash operational"},
    "sarvam": {"status": "reachable", "details": "Sarvam AI API reachable"},
    "groq": {"status": "reachable", "details": "Groq reachable"},
    "ollama": {"status": "unreachable", "details": "Ollama connection refused at localhost:11434"}
  }
}
```

---

## 3. Calibrated Scope Guard & 3-Way Intent Classification

To eliminate false rejections of legitimate citizen complaints in Hindi and Hinglish while preventing out-of-domain drift, the agent uses a **calibrated 3-way intent classifier**:

```
                         Citizen Query
                               │
                               ▼
                ┌──────────────────────────────┐
                │ classify_intent(text, lang)  │
                └──────────────┬───────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
   [in_scope]         [needs_clarification]   [out_of_scope]
Proceed to RAG &      Ask ONE clarifying     Polite deterministic
Answer Generation     question (never reject   boundary abstention
                      silently)
```

1. **`in_scope`**: Legitimate cyber crime queries (financial fraud, blackmail, hacked accounts, lost phones, thana finder).
2. **`needs_clarification`**: Borderline or ambiguous inputs (e.g. *"Mera phone check karo"*, *"Police help chahiye"*). Instead of rejecting, the agent immediately asks a clarifying question:
   > *"Kya aapke saath koi online fraud, blackmailing ya cyber apradh hua hai? Kripya thoda vistar se batayein taaki sahi sahayata mil sake."*
3. **`out_of_scope`**: Clear off-topic queries (recipes, sports, weather, jokes, general python coding). Emits the official abstention wording:
   > *"I only assist with cyber crime reporting, online fraud, digital safety, and police procedures in Alwar."*

### Calibrated Thresholds
Calibrated and validated against **150 labelled test cases** (`backend/tests/data/scope_eval.jsonl`) achieving **98.00% precision/recall** across English, Hindi (Devanagari), and Hinglish:
- `OOS_DISTANCE = 0.42`
- `BM25_MIN_SCORE = 7.0`
- Retrieval context capped to **top 3–4 chunks** (`RERANK_K = 4`) to prevent off-topic clutter.

---

## 4. User-Oriented Conversation & Session State (`backend/app/agent/state.py`)

Victims of cyber crime are frequently stressed and provide incomplete details. The agent maintains **conversational state** across turns to structure guidance efficiently:

### State Fields Tracked
- `case_type`: `upi_fraud`, `sextortion`, `deepfake`, `account_hack`, `lost_phone`, `digital_arrest`, `loan_app`, `thana_finder`
- `money_lost`: boolean
- `amount`: float (e.g. Rs 25,000)
- `time_since_incident`: e.g. "30 minutes ago", "yesterday"
- `payment_method`: `upi`, `card`, `net_banking`, `crypto`
- `platform`: `whatsapp`, `instagram`, `telegram`, `facebook`, `phonepe`, `gpay`
- `steps_already_done`: list of actions the citizen has already taken (e.g. `["called_1930", "blocked_sim"]`)

### Proactive Triage & Answer Ordering
1. **Triage Question**: If essential fraud details are missing, the agent delivers the single immediate safe action (**call 1930**) and asks **ONE short question first**:
   > *"Paise kitne der pehle kate, aur transaction UPI se hua ya card se?"*
2. **Strict Answer Ordering**:
   1. **Single most urgent action** (e.g. Golden Hour call to 1930 or SIM block).
   2. **Next 2–3 actionable steps** specific to THIS case type.
   3. **Offer detailed procedure**: *"Kya aapko iska poora step-by-step procedure chahiye?"*
3. **Step Deduplication**: The agent inspects `steps_already_done`. If the citizen says *"I already called 1930"*, the agent never repeats that instruction.
4. **Token Budget & Sentence Boundary Trimming**:
   - `max_output_tokens=1000` allows complete Hindi replies (which require ~2x tokens of English).
   - `trim_to_sentence_boundary()` ensures responses never get cut off mid-sentence.

---

## 5. Defense-in-Depth Anti-Hallucination & Safer Sanitizer

Legal accuracy is paramount. The agent enforces strict guardrails before, during, and after generation:

```
 Citizen Message
       │
       ▼
 [1. Calibrated Scope Guard] ──► Abstain if out of scope / Clarify if borderline
       │
       ▼
 [2. Hybrid Retrieval Cap]  ──► Top 3-4 chunks only (BM25 + ChromaDB + RRF)
       │
       ▼
 [3. Low-Entropy Generation] ──► Temperature = 0.2, Strict Grounding Directives
       │
       ▼
 [4. Context-Aware Phone Sanitizer] ──► Overwrites ONLY unverified numbers in official
       │                                 contact context; PRESERVES user suspect numbers,
       │                                 retrieved numbers, and UTRs/Txn IDs
       ▼
 [5. Tricky Phrasing Neutralization] ──► Catches Hindi/Hinglish fee, refund, and arrest claims
       │
       ▼
 [6. Entailment-Based Citation Check]──► Validates factual claims (phones, portals, deadlines,
                                         fees) against retrieved chunks; drops unverified cites
```

### Context-Aware Phone Sanitization
- **Preserved**:
  - Verified police numbers: `1930`, `112`, `1090`, `8764874306` (Alwar Control Room), `0144-2337456`.
  - Suspect numbers typed by the user (e.g. *"scammer called from 9876543210"*).
  - Valid numbers present in retrieved knowledge base chunks.
  - Numbers preceded by transaction identifiers (UTR, Txn ID, Ack ID, Complaint No).
- **Sanitized**:
  - Any unverified 10-digit number appearing in an official contact context (*"call SHO at 9876543210"*, *"थानाधिकारी का नंबर 9876543210"*) is replaced with the official **Alwar Cyber Control Room (`8764874306`)**.

### Tricky Hindi/Hinglish Phrasing Neutralization
- **Filing Fee Claims**: Catches *"shikayat darj karane ka kharcha aayega"*, *"registration fee of rs 200"*, *"पंजीकरण शुल्क लगेगा"*. Appends statutory reminder that cyber reporting is **100% free by law**.
- **100% Refund Guarantees**: Catches *"24 ghante me refund pakka ho jayega"*, *"guaranteed paisa wapas milenge"*, *"24 घंटे में पैसे वापस की गारंटी"*. Clarifies that fund recovery depends strictly on freezing funds during the golden hour.
- **Judicial Outcome Promises**: Catches *"accused ko turant arrest kar liya jayega pakka"*, *"पक्की गिरफ्तारी होगी और जमानत नहीं मिलेगी"*. Appends statutory note that arrests and bail are subject to police investigation and court discretion.

### Entailment-Style Grounding & Citation Check
Replaces naive token overlap with factual claim entailment:
- A retrieved chunk is cited **only** if the specific factual entity (e.g. `*401*`, `StopNCII`, `CEIR`, `MRM lien`, `AEPS biometric lock`) or procedure is asserted in the reply.
- All factual claims (phone, portal, fee, deadline) are validated via `verify_factual_claims_entailment()`. If ungrounded facts are found, citations are dropped and limitation disclaimers are communicated.

---

## 6. Official Verified Directory & Helplines

| Agency / Service | Contact / Portal | Jurisdiction / Purpose |
| :--- | :--- | :--- |
| **Alwar Cyber Control Room** | **`8764874306`** | District Cyber Emergency Response, Alwar |
| **Alwar Police Control Room (Landline)** | **`0144-2337456`** | General Police Control Room, Alwar District |
| **National Cyber Helpline** | **`1930`** | Golden-hour financial fraud reporting (I4C) |
| **Emergency Services** | **`112`** | Immediate police, fire, or ambulance dispatch |
| **Women Helpline** | **`1090`** | Stalking, harassment, and female safety |
| **National Cyber Portal** | `cybercrime.gov.in` | Formal complaint lodging & status tracking |
| **Lost/Stolen Phone Portal (CEIR)** | `sancharsaathi.gov.in` / `ceir.gov.in` | IMEI blocking and duplicate SIM tracking |
| **Morphed Media Takedown** | `stopncii.org` | Non-consensual intimate image hash generation |
| **RBI Sachet Portal** | `sachet.rbi.org.in` | Verification & reporting of illegal loan apps |

---

## 7. Golden Evaluation Dataset & Results (`eval_report.md`)

The agent is continuously benchmarked against a **112-case golden evaluation dataset** (`backend/tests/eval/golden_dataset.json`) covering 8 cyber crime categories + out-of-scope controls in English, Hindi, and Hinglish.

### Latest Benchmark Summary
- **Out-of-Scope False-Reject Rate**: **0.0%** (Zero legitimate cyber complaints falsely rejected).
- **Hallucinated-Number Rate**: **0.0%** (Zero unverified phone numbers in official contexts).
- **Forbidden Facts Neutralization Rate**: **0.0%** (Zero unneutralized fee or refund promises).
- **Groundedness Rate**: **65.66%** across all localized sub-questions.
- **Average Offline Pipeline Latency**: **~0.01s - 0.04s** per turn.

Run the evaluation suite anytime via:
```powershell
python backend/evaluation/eval_runner.py --mock
```
Results are saved to [`eval_report.md`](file:///C:/claude_work/Alwar-Cyber-Crime-Help-Guidance-Portal/eval_report.md).

---

## 8. Quick Commands Reference

### Run the FastAPI Backend
```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Complete Test Suite
```powershell
.\backend\.venv\Scripts\pytest.exe -c backend\pyproject.toml -p no:cacheprovider -m "not integration" backend\tests
```

### Run Provider & Model Evaluations
```powershell
# Calibrated evaluation across all providers
.\backend\.venv\Scripts\python.exe backend/evaluation/eval_runner.py --mock

# Live evaluation on active Gemini provider
.\backend\.venv\Scripts\python.exe backend/evaluation/eval_runner.py --providers gemini
```
