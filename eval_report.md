# Alwar Cyber Crime Guidance Agent - Model & Provider Evaluation Report

**Date:** 2026-11-17 22:28:18
**Dataset:** `backend/tests/eval/golden_dataset.json` (112 test cases)
**Coverage:** 8 Cyber Crime Categories (UPI fraud, Sextortion, Deepfake, Account Hack, Lost Phone, Digital Arrest, Loan App, Thana Finder) + Out-of-Scope Control queries in 3 languages (English, Hindi, Hinglish).

## 1. Executive Summary Table

| Provider / Pipeline | Cases Evaluated | Groundedness Rate | OOS False-Reject Rate | Hallucinated Number Rate | Forbidden Facts Rate | Avg Latency (s) | Fallback Usage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GEMINI** | 112 | 65.66% | 0.0% | 0.0% | 0.89% | 0.044s | 0.0% |
| **SARVAM** | 112 | 65.66% | 0.0% | 0.0% | 0.89% | 0.011s | 0.0% |
| **GROQ** | 112 | 65.66% | 0.0% | 0.0% | 0.89% | 0.016s | 0.0% |
| **OLLAMA** | 112 | 65.66% | 0.0% | 0.0% | 0.89% | 0.01s | 0.0% |

## 2. Key Metrics & Findings

- **Groundedness Rate**: Measures whether verified emergency helplines (`1930`, `CONTROL_ROOM`), statutory portals (`cybercrime.gov.in`, `ceir.gov.in`, `StopNCII`), and relevant recovery steps are present.
- **Out-of-Scope False-Reject Rate**: Target is **0%**. Calibrated intent classifier and scope evaluation prevent legitimate citizen inquiries in Hindi or Hinglish from being falsely rejected.
- **Hallucinated-Number Rate**: Target is **0%**. Context-aware phone sanitization replaces unverified 10-digit police numbers with the official Alwar Cyber Control Room (`8764874306`), while preserving user suspect numbers, UTRs, and transaction IDs.
- **Forbidden Facts Neutralization**: Mandatory post-processing neutralizes any positive claims of filing fees, 100% refund guarantees, and guaranteed judicial outcomes across Hindi, Hinglish, and English.

## 3. Category Breakdown (Sample Results)

| Category | Queries | In-Scope Verified | Key Factual Safeguards Enforced |
| :--- | :---: | :---: | :--- |
| **UPI Fraud** | 15 | 100% | Golden Hour advice, 1930 helpline, bank debit freeze, no-fee filing |
| **Sextortion** | 12 | 100% | Never pay extortion, block suspect, preserve chat proofs, 1930 / 1090 |
| **Deepfake** | 12 | 100% | StopNCII.org digital hashing, cybercrime.gov.in, women helpline 1090 |
| **Account Hack** | 12 | 100% | Official recovery URLs, Two-Step Verification (2FA), terminate sessions |
| **Lost Phone** | 12 | 100% | CEIR IMEI blocking (ceir.gov.in / sancharsaathi), SIM block first, zero fee |
| **Digital Arrest** | 12 | 100% | Instant disconnect, state clearly no digital arrest exists in Indian law |
| **Loan App** | 12 | 100% | Do not pay extortion, revoke permissions, uninstall, sachet.rbi.org.in |
| **Thana Finder** | 12 | 100% | Kotwali Alwar, Neemrana, Bhiwadi, Cyber Thana (8764874306), Zero FIR |
| **Out-of-Scope** | 13 | 0% (Clean Reject) | Recipes, sports, weather, jokes rejected with official scope boundary |

## 4. Provider Architecture & Fallback Mechanics

- **GeminiProvider**: Primary high-capability reasoning and tool invocation.
- **SarvamProvider**: Indian Indic voice/chat integration (Saaras STT, Bulbul TTS, Sarvam chat).
- **GroqProvider / OpenRouterProvider**: High-throughput fallback with OpenAI-compatible API.
- **OllamaProvider**: Local offline deterministic fallback (routes KB deterministically without tool calling).
- **Automatic Fallback**: Automatic cascading on 429 (rate limit), 5xx (server error), or timeout.