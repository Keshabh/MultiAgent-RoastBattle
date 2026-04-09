<div align="center">

```
██████╗  ██████╗  █████╗ ███████╗████████╗    ██████╗  █████╗ ████████╗████████╗██╗     ███████╗
██╔══██╗██╔═══██╗██╔══██╗██╔════╝╚══██╔══╝    ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██║     ██╔════╝
██████╔╝██║   ██║███████║███████╗   ██║       ██████╔╝███████║   ██║      ██║   ██║     █████╗  
██╔══██╗██║   ██║██╔══██║╚════██║   ██║       ██╔══██╗██╔══██║   ██║      ██║   ██║     ██╔══╝  
██║  ██║╚██████╔╝██║  ██║███████║   ██║       ██████╔╝██║  ██║   ██║      ██║   ███████╗███████╗
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚══════╝
```

### 🎤 Multi-Agent AI Roast Battle Arena

**Two celebrities. Real research. 3 escalating rounds. One winner.**

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-Click%20Here-FFD700?style=for-the-badge&logo=google-chrome&logoColor=black)](https://keshabh.github.io/MultiAgent-RoastBattle)
[![Backend](https://img.shields.io/badge/BACKEND-Railway-purple?style=for-the-badge&logo=railway)](https://multiagent-roastbattle-production.up.railway.app/health)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-00C853?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=for-the-badge)](https://groq.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-1868F2?style=for-the-badge)](https://smith.langchain.com)

---

> *"Type two names. Watch AI agents research, roast, and battle it out — live, on beat, bar by bar."*

</div>

---

## 🎬 Demo

> 📌 **[▶ Click here to try it live](https://keshabh.github.io/MultiAgent-RoastBattle)**

```
Enter: Elon Musk  vs  Mark Zuckerberg
       ↓
🗄️ Two-level cache check — full battle or per-celeb DDG data
       ↓
🔍 4 parallel DDG searches via Send API (web + news per celeb)
       ↓
🧠 LLM Call 1 — 8B model distills research into roast ammo
       ↓
🎤 LLM Call 2 — 70B model generates full 3-round battle as JSON
       ↓
🎬 Roast engine plays it out bar by bar, on the beat
       ↓
👑 Round verdicts → Final winner revealed
```

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      USER INPUT                         │
│               Celebrity 1  vs  Celebrity 2              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│             LEVEL 1 — BATTLE CACHE CHECK                │
│                                                         │
│  key = normalized(celeb_a + celeb_b)                    │
│  HIT  → return full battle JSON instantly ⚡            │
│  MISS → proceed to Level 2                              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│             LEVEL 2 — PER-CELEB DDG CACHE               │
│                                                         │
│  Check 4 keys independently:                            │
│  celeb1:general  celeb1:news                            │
│  celeb2:general  celeb2:news                            │
│                                                         │
│  Only cache-missed keys → Send API fan-out              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│          SEND API — PARALLEL DDG SEARCHES               │
│                                                         │
│  ddg_search_node(celeb1, general) ──┐                   │
│  ddg_search_node(celeb1, news)    ──┤                   │
│  ddg_search_node(celeb2, general) ──┤──► gather_results │
│  ddg_search_node(celeb2, news)    ──┘                   │
│                                                         │
│  All 4 run in parallel. Results written to cache.       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           LLM CALL 1 — RESEARCH SYNTHESIZER             │
│                    llama-3.1-8b-instant                 │
│                                                         │
│  Input:  all 4 DDG dumps (both celebs, single prompt)   │
│  Output: 12 roast ammo bullets per celebrity            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           LLM CALL 2 — BATTLE GENERATOR                 │
│                  llama-3.3-70b-versatile                │
│                                                         │
│  Input:  roast ammo for both celebs                     │
│  Output: full 3-round battle JSON                       │
│          (all roasts + verdicts + winner pre-generated) │
│                                                         │
│  Round 1 → sharp opener                                 │
│  Round 2 → personal, counter Round 1                    │
│  Round 3 → nuclear, mic drop                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                ROAST ENGINE (Frontend)                  │
│                                                         │
│  Zero LLM calls. Pure display logic.                    │
│  Lines animate bar by bar, synced to BPM                │
│  Punchline lands in gold — crowd reacts                 │
│  Judge verdict card after each round                    │
│  Winner revealed with final verdict                     │
└─────────────────────────────────────────────────────────┘
```
## StateGraph
<img width="1206" height="553" alt="image" src="https://github.com/user-attachments/assets/7dd54a17-b6f3-49b8-b404-518cfab6e593" />

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph — StateGraph, Send API, conditional edges |
| **LLM — Research** | LLaMA 3.1 8B Instant via Groq |
| **LLM — Battle** | LLaMA 3.3 70B Versatile via Groq |
| **Web Search** | DuckDuckGo Search (web + news, no API key) |
| **Caching** | diskcache — two-level, TTL 7 days |
| **Observability** | LangSmith — tracing, token counts, latency |
| **Backend** | FastAPI + asyncio |
| **Frontend** | Vanilla HTML/CSS/JS — zero frameworks |
| **Beat Engine** | Web Audio API — procedural, no audio files |
| **Deployment** | Railway (backend) + GitHub Pages (frontend) |

---

## 🔑 Key Features

**Agentic Patterns**
- LangGraph StateGraph with typed state schema
- Send API parallel fan-out / fan-in
- Conditional edge routing at every decision point

**Caching Strategy**
- Two-level diskcache — full battle + per-celebrity DDG
- Order-normalized keys (`elon_vs_trump` == `trump_vs_elon`)
- TTL 7 days — auto-expiring, zero manual cleanup

**LLM Pipeline**
- Right model per task — 8B for summarization, 70B for creativity
- Two LLM calls total — all work front-loaded before playback
- Structured JSON output — roast engine is pure display, zero LLM

**Production Patterns**
- `asyncio.to_thread` — non-blocking event loop
- LangSmith tracing — opt-in via env variable
- Pydantic request validation + structured logging
- Full env-based config — zero hardcoded values

---

## 🚀 Run Locally

### Prerequisites
- Python 3.13+
- Groq API key — free at [console.groq.com](https://console.groq.com)
- LangSmith API key — free at [smith.langchain.com](https://smith.langchain.com) *(optional)*

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Keshabh/MultiAgent-RoastBattle.git
cd MultiAgent-RoastBattle

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY at minimum
```

```bash
# 5. Run the backend
uvicorn main:app --reload --port 8000

# 6. Open the frontend
# Open index.html directly in Chrome
# Make sure BACKEND_URL = "http://127.0.0.1:8000" in index.html
```

---

## 🔧 Environment Variables

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Models (optional — defaults shown)
GROQ_MODEL_PROFILE=llama-3.1-8b-instant
GROQ_MODEL_ROAST=llama-3.3-70b-versatile
GROQ_TEMPERATURE_PROFILE=0.9
GROQ_TEMPERATURE_ROAST=1.1

# Cache (optional — defaults shown)
CACHE_DIR=./cache
CACHE_TTL_SECONDS=604800

# LangSmith tracing (optional — remove to disable)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=roast-battle

# CORS (optional)
ALLOWED_ORIGINS=*
```

---

## 📁 Project Structure

```
MultiAgent-RoastBattle/
├── main.py              # FastAPI app + LangGraph pipeline
├── requirements.txt     # Python dependencies
├── Procfile             # Railway deployment config
├── .env.example         # Environment variables template
├── index.html           # Complete frontend (single file)
└── README.md
```

---

## 🌐 Deployment

| Service | URL |
|---|---|
| Frontend | [keshabh.github.io/MultiAgent-RoastBattle](https://keshabh.github.io/MultiAgent-RoastBattle) |
| Backend | [multiagent-roastbattle-production.up.railway.app](https://multiagent-roastbattle-production.up.railway.app/health) |

---

<div align="center">

Built with 🔥 by [Keshabh](https://github.com/Keshabh)

*Part of a multi-agent AI portfolio — also check out the RAG Document Q&A system*

</div>
