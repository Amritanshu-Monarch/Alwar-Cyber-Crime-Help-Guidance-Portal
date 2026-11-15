
# 🛡️ Alwar Cyber Crime Help & Guidance Portal

A lightweight, mobile-first, and senior-citizen-accessible web application designed to empower citizens of Alwar, Rajasthan with instant access to cybercrime reporting, emergency helplines, automated complaint drafting, and digital safety guidance. Built by interns Amritanshu, Rajveer, Aayush & Rishabh ✌️

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![Mobile First](https://img.shields.io/badge/Design-Mobile--First-success)

---

## 🌟 Key Features

* 🌐 **Bilingual (English / Hindi) Toggle**: Instant, seamless translation across every single text element, menu, modal, and generated letter—making it accessible to users of all literacy levels.
* 📱 **Mobile-First 2-Column Grid**: Optimized for mobile viewing (~90% of site traffic) taking UI inspiration from official RajCop Citizen app layouts while maintaining full structural integrity on tablets and desktop monitors.
* 📝 **Automated Complaint Letter Generator**: Generates a formally structured legal complaint draft to the Station House Officer (SHO) in less than a minute. Includes one-click options to **Copy Text** or **Print / Save as PDF**.
* 🚨 **Pinned Emergency Speed-Dial Bar**: Fixed bottom bar featuring high-visibility emergency calling buttons for the **Alwar Police Cyber Control Room (`8764874306`)** and **National Cyber Helpline (`1930`)**.
* 💬 **Direct WhatsApp Reporting**: Direct single-tap redirection to chat with the Alwar Police Cyber Crime Desk for reporting online frauds in real time.
* ⚠️ **Digital Evidence Preservation Rules**: Prominently displays vital "Golden Hour" guidance warning citizens **NOT** to delete APKs, WhatsApp chats, screenshots, or call logs after facing a scam.
* 📍 **Police Station Finder & Verification Guides**: Quick links for mapping local police stations across Alwar and tenant/domestic servant verification procedures.

---

## 🤖 AI Voice Agent (v2)

`backend/` + `frontend/` add a Hindi/English voice and chat assistant built on Google Gemini:

* **Agent**: Google ADK agent with tools: `search_knowledge` (RAG), `get_contacts`, `find_police_station`, `draft_complaint_letter`.
* **RAG**: Gemini embeddings (`gemini-embedding-001`) + ChromaDB over `backend/kb/*.md`.
* **Voice**: browser mic → WAV → Gemini speech-to-text → agent → Gemini TTS (browser TTS fallback).
* **Stack**: FastAPI, uv, React + TypeScript + shadcn/ui. The original single-file site (`index.html`) is kept as the static fallback.

### Run

```powershell
# 1. backend (needs a key from https://aistudio.google.com/app/apikey)
cd backend
copy .env.example .env          # set GOOGLE_API_KEY
uv sync
uv run python -m app.services.rag   # embed kb/*.md (also runs automatically on first start)
uv run uvicorn app.main:app --port 8000
uv run pytest                   # 23 tests, no API key needed

# 2. frontend (proxies /api to :8000)
cd frontend
npm install
npm run dev
```

If the repo lives in OneDrive, create the venv outside it: `$env:UV_PROJECT_ENVIRONMENT="C:\venvs\alwar-cyber-agent"; $env:UV_LINK_MODE="copy"`.

Add knowledge by dropping markdown files into `backend/kb/` and re-running the ingest command.

---

## 🛠️ Tech Stack

* **Frontend**: HTML5, Vanilla JavaScript (ES6)
* **Styling**: [Tailwind CSS CDN](https://tailwindcss.com/)
* **Icons**: [FontAwesome 6.4.0](https://fontawesome.com/)
* **Typography & UI**: Responsive Flexbox/Grid, Glassmorphism backdrop filters, custom CSS pulse keyframe animations.

---

## 📂 Project Structure

```text
.
├── index.html          # Single-file lightweight core web app
├── README.md           # Project documentation
└── favicon.png         # Official Rajasthan Police emblem favicon
