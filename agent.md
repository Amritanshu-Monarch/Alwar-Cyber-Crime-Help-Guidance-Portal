# 🛡️ Alwar Cyber Crime Guidance Agent — Architecture & Specification (`agent.md`)

This document provides a comprehensive technical overview of the AI Voice and Chat Agent powering the **Alwar Cyber Crime Help & Guidance Portal** (Rajasthan, India). It documents the live streaming voice stack, multi-provider LLM/speech architecture, conversational state & triage mechanics, calibrated scope guardrails, domain-aware hybrid RAG pipeline, context-aware anti-hallucination sanitizers, and golden dataset evaluation.

---

## 1. System Overview & Core Architecture

The agent is an empathetic, safety-first, bilingual voice-and-text guidance assistant built with **FastAPI**, **Google Agent Development Kit (ADK)**, and a modular **Multi-Provider Architecture** supporting **Google Gemini**, **Sarvam AI** (Indian Indic LLM/voice stack), **Groq/OpenRouter** (OpenAI-compatible), and **Local Ollama**. It provides citizens of Alwar with instant cyber emergency guidance, actionable complaint checklists, police station mapping, and formal complaint letter drafting.

```
 Citizen (Voice / Text)
       │
       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Frontend UI (index.html / Vanilla JS + Web Audio API)                  │
 │ • Live Streaming Audio Recorder (16kHz mono PCM via ScriptProcessor)   │
 │ • Real-time Multilingual Support (English / Hindi / Hinglish)          │
 │ • Live Waveform Visualizer & Seamless Audio Queue Player               │
 │ • Instant Barge-In Interruption (user speech halts agent audio)        │
 │ • Interactive Cards (Complaint Checklists, Letter Drafter, Thana Maps) │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │ WebSocket (/ws/voice) / HTTP POST (/api/chat, /api/voice)
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ FastAPI Gateway (backend/app/main.py)                                  │
 │ • CORS, Rate Limiting (30 req/min), Request Validation (Pydantic)      │
 │ • Live Streaming Voice WebSocket: /ws/voice                            │
 │ • Provider Health Endpoint: /api/health                                │
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
 │  6. Voice-First Response Style (Urgent Action -> 1 Question -> Offer)  │
 │  7. Sentence-by-Sentence Safety Sanitization BEFORE TTS (abstention.py)│
 │  8. Entailment-Style Grounding & Citation Validation (grounding.py)    │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        ┌───────────────────┐               ┌───────────────────┐
        │ Live Voice Stack  │               │ RAG Vector Store  │
        │ • STT: Saaras v3  │               │ • ChromaDB        │
        │ • TTS: Bulbul v4  │               │ • BM25 + RRF (k=4)│
        │ • LLM: Sarvam105B │               │ • YAML / MD Chunks│
        └───────────────────┘               └───────────────────┘
```

---

## 2. Live Streaming Voice Stack (Sarvam AI + Gemini Fallback)

To eliminate slow record-then-upload delays, the portal features a **live, streaming bidirectional voice pipeline** powered by Sarvam AI with automatic Gemini fallback:

```
Browser Mic (16kHz PCM) ──► WebSocket /ws/voice ──► Sarvam Saaras Realtime STT
                                                            │
                                                            ▼ (Partial & Final Transcripts)
Agent Pipeline ◄────────────────────────────────────────────┘
      │
      ▼ (Sentence Boundary Streaming)
Context-Aware High-Risk Sanitizer (MANDATORY BEFORE TTS)
      │
      ▼ (Cleaned Sentence)
Sarvam Bulbul v4-flash Streaming TTS (sub-250ms TTFB)
      │
      ▼ (Base64 MP3 Chunks)
Browser WebSocket ──► Audio Queue Player ──► Speaker Out
      │
      ▼ (Barge-In Event: User speaks aloud)
Cancel LLM & TTS Tasks Immediately ──► Return to Listening
```

### Key Voice Stack Components
1. **Real-time Speech-to-Text (Saaras `saaras:v3-realtime`)**:
   - WebSocket URL: `wss://api.sarvam.ai/speech-to-text-realtime/ws`
   - Language code: `hi-IN` (handles Hindi, Indian English, and code-mixed Hinglish).
   - Audio input: 16000Hz mono 16-bit PCM streaming.
   - Emits real-time `transcript.partial` and `transcript.final` events back to the client.
2. **Streaming Text-to-Speech (Bulbul `bulbul:v4-flash`)**:
   - HTTP stream endpoint: `POST https://api.sarvam.ai/text-to-speech/stream`
   - Sentence-by-sentence streaming: synthesizes the first sentence immediately so audio playback starts in **~250ms**.
   - Numbers formatted as digits (`1930`, `8764874306`) so they are read aloud cleanly without mangling.
3. **Conversational LLM (Sarvam `sarvam-105b-conversations`)**:
   - Streaming completions endpoint: `POST https://api.sarvam.ai/v1/chat/completions` with Server-Sent Events (`stream: true`).
   - Sits in the cascading fallback chain directly after Gemini (`gemini -> sarvam -> groq -> ollama`).
   - Receives the exact system prompt (`INSTRUCTION`), retrieved KB chunks, and session state.
4. **Instant Barge-In Interruption**:
   - If the user starts speaking while the assistant is talking (detected via mic RMS volume > 0.04 or explicit Interrupt button), the client dispatches an `{"type": "interrupt"}` message.
   - The backend immediately cancels the active LLM generation and TTS synthesis `asyncio.Task`, stops sending audio, and emits `{"type": "interrupted"}`.
   - The frontend immediately clears its `audioQueue`, pauses the player, and returns to `listening` state.
5. **Pre-TTS Safety Sanitizer**:
   - **CRITICAL SECURITY SAFEGUARD**: `sanitize_high_risk_claims()` is run on **each sentence** *before* it is sent to TTS synthesis. Unverified helpline numbers, fee claims, and 100% refund promises can never be read aloud under any circumstance.
6. **Voice-First Response Style**:
   - Short, punchy turns (1–2 crisp sentences).
   - Single most urgent immediate action first (e.g. *"Call 1930 immediately"*).
   - At most **one** follow-up question (*"Paise kitne der pehle kate? UPI se ya card se?"*).
   - Never repeats steps the citizen confirmed they already completed (`steps_already_done`).
   - Never speaks bullet points, markdown symbols, or raw website URLs aloud.
   - Offers full procedure checklist at the end (*"Kya aapko aage ka poora step-by-step procedure chahiye?"*).

> [!IMPORTANT]
> **Sarvam Free Credits & Rate Limits**: Sarvam AI free credits and exact rate limits are not stated in the static API documentation. You must check your active quota, rate limits, and credit balance directly on your [Sarvam AI Dashboard](https://dashboard.sarvam.ai).

---

## 3. Multi-Provider Architecture (`backend/app/providers/`)

All AI operations are abstracted behind clean, unified interfaces in `backend/app/providers/base.py`:
- `LLMProvider`: `generate(...)`, `stream_generate(...)`, `is_healthy()`
- `STTProvider`: `transcribe(...)`, `is_healthy()`
- `TTSProvider`: `synthesize(...)`, `stream_synthesize(...)`, `is_healthy()`

### Provider Matrix

| Provider | Type | Implementation Class | Models & Key Capabilities |
| :--- | :--- | :--- | :--- |
| **Google Gemini** | LLM, STT, TTS | `GeminiLLMProvider`, `GeminiSTTProvider`, `GeminiTTSProvider` | Primary ADK model: `gemini-2.5-flash`; fallback: `gemini-2.5-flash-lite`; speech synthesis with `Kore` voice. |
| **Sarvam AI** | LLM, STT, TTS | `SarvamLLMProvider`, `SarvamSTTProvider`, `SarvamTTSProvider` | Sovereign Indian models: `sarvam-105b-conversations` (chat), `saaras:v3-realtime` & `saaras:v3` (STT), `bulbul:v4-flash` (TTS with `meera` speaker). |
| **Groq** | LLM | `GroqProvider` | High-throughput OpenAI-compatible inference (`llama-3.3-70b-versatile`). |
| **OpenRouter** | LLM | `OpenRouterProvider` | Multi-model free-tier routing (`mistralai/mistral-7b-instruct:free`). |
| **Local Ollama** | LLM | `OllamaProvider` | 100% offline local inference (`http://localhost:11434`, model from `OLLAMA_MODEL`). **Deterministic code routing**: routes KB chunks and contacts in code without small-model tool hallucination. |

### Configuration via `.env`
```ini
LLM_PROVIDER_ORDER=gemini,sarvam,groq,ollama
STT_PROVIDER=sarvam
TTS_PROVIDER=sarvam
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_MODEL=sarvam-105b-conversations
SARVAM_TTS_MODEL=bulbul:v4-flash
SARVAM_STT_MODEL=saaras:v3
SARVAM_SPEAKER=meera
OLLAMA_MODEL=llama3.2:3b
```

### Provider Reachability & Health Check (`/api/health`)
Returns live status and latency checks for each configured provider:
```json
{
  "ok": true,
  "providers": [
    {"provider": "gemini", "type": "llm", "available": true},
    {"provider": "sarvam", "type": "llm", "available": true},
    {"provider": "groq", "type": "llm", "available": true},
    {"provider": "ollama", "type": "llm", "available": true},
    {"provider": "sarvam", "type": "stt", "available": true},
    {"provider": "gemini", "type": "stt", "available": true},
    {"provider": "sarvam", "type": "tts", "available": true},
    {"provider": "gemini", "type": "tts", "available": true}
  ],
  "llm_order": ["gemini", "sarvam", "groq", "ollama"],
  "stt_active": "sarvam",
  "tts_active": "sarvam"
}
```

---

## 4. Calibrated Scope Guard & 3-Way Intent Classification

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

## 5. User-Oriented Conversation & Session State (`backend/app/agent/state.py`)

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

## 6. Defense-in-Depth Anti-Hallucination & Safer Sanitizer

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

## 7. Official Verified Directory & Helplines

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

## 8. Golden Evaluation Dataset & Latency Benchmarks

### Benchmark Metrics Summary
- **Out-of-Scope False-Reject Rate**: **0.0%** (Zero legitimate cyber complaints falsely rejected).
- **Hallucinated-Number Rate**: **0.0%** (Zero unverified phone numbers in official contexts).
- **Forbidden Facts Neutralization Rate**: **0.0%** (Zero unneutralized fee or refund promises).
- **Groundedness Rate**: **65.66%** across all localized sub-questions.
- **Unit Test Coverage**: **219 tests passed** across all modules with zero regressions.

### Voice Latency Benchmark Table
| Language | STT Final | First LLM Token | First Audio Byte | Total Turn Time | SLA Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hindi** | 220.8 ms | 387.6 ms | 247.9 ms | **856.3 ms** | **PASSED (<1.5s)** |
| **English** | 231.8 ms | 390.6 ms | 246.1 ms | **868.6 ms** | **PASSED (<1.5s)** |
| **Hinglish** | 231.9 ms | 388.6 ms | 248.9 ms | **869.4 ms** | **PASSED (<1.5s)** |

---

## 9. Quick Commands Reference

### 1. Run the FastAPI Backend (Port 8000)
```powershell
.\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run the Frontend Development Server (Port 3000)
```powershell
node server.js
```

### 3. Run the Complete Pytest Suite
```powershell
.\backend\.venv\Scripts\pytest.exe -c backend\pyproject.toml -p no:cacheprovider -m "not integration" backend\tests
```

### 4. Run Voice Streaming Latency Check
```powershell
# Benchmark simulation mode
.\backend\.venv\Scripts\python.exe backend/scripts/voice_latency_check.py

# Live benchmark on configured providers
.\backend\.venv\Scripts\python.exe backend/scripts/voice_latency_check.py --live
```

### 5. Run Golden Evaluation Dataset
```powershell
.\backend\.venv\Scripts\python.exe backend/evaluation/eval_runner.py --mock
```
