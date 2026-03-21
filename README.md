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
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-00C853?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=for-the-badge)](https://groq.com)

---

> *"Type two names. Watch AI agents research, roast, and battle it out — live, on beat, bar by bar."*

</div>

---

## 🎬 Demo

> 📌 **[▶ Click here to try it live](https://keshabh.github.io/MultiAgent-RoastBattle)**

```
Enter: Elon Musk  vs  Mark Zuckerberg
       ↓
🔍 Agents research both celebs across web + news
       ↓
🎤 Round 1 — Light jabs land on screen, bar by bar, on the beat
       ↓
🔥 Round 2 — Gets personal. Clap backs. Crowd getting loud.
       ↓
💀 Round 3 — NUCLEAR. Mic drop. One winner.
```

---

## 🧠 How It Works — Multi-Agent Architecture

This is not a chatbot. It's a **LangGraph state machine** with specialized agents passing a shared state through a directed graph.

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│              Celebrity 1  vs  Celebrity 2               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 RESEARCH AGENT                          │
│                                                         │
│  DuckDuckGo Web Search ──► Celebrity 1 profile          │
│  DuckDuckGo News       ──► scandals, drama, failures    │
│  Biography Search      ──► Celebrity 2 profile          │
│                                                         │
│  Profile Builder distills raw data into                 │
│  "roast ammunition" bullet points per celeb             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              BATTLE MASTER (LangGraph Loop)             │
│                                                         │
│   Round 1 ──► Round 2 ──► Round 3 ──► END               │
│      │           │           │                          │
│   [Roast]    [Escalate]  [Nuclear]                      │
│   Writer 1   Writer 1    Writer 1                       │
│   Writer 2   Writer 2    Writer 2                       │
│                                                         │
│  Each round reads ALL previous rounds before writing    │
│  Conditional edge: continue if round < 3, else END      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FRONTEND                              │
│                                                         │
│  Bars appear line by line synced to BPM                 │
│  Punchline hits in gold — holds longer on screen        │
│  Procedural rap beat via Web Audio API                  │
│  Round banners punch in between rounds                  │
│  Winner revealed after Round 3                          │
└─────────────────────────────────────────────────────────┘
```

### Agent Breakdown

| Agent | Role | Tools Used |
|---|---|---|
| **Research Agent** | Scrapes real info about both celebs | DuckDuckGo Web, DuckDuckGo News |
| **Profile Builder** | Distills research into roast ammunition | LLaMA 3.3 70B via Groq |
| **Battle Master** | Orchestrates 3 rounds via LangGraph loop | LangGraph StateGraph |
| **Roast Writer ×2** | Writes each celeb's bars with escalation | LLaMA 3.3 70B via Groq |

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph (StateGraph with conditional edges) |
| **LLM** | LLaMA 3.3 70B Versatile via Groq API |
| **Web Search** | DuckDuckGo Search (web + news) |
| **Backend** | FastAPI + Python |
| **Frontend** | Vanilla HTML/CSS/JS — zero frameworks |
| **Beat Engine** | Web Audio API (procedural, no audio files) |
| **Deployment** | Railway (backend) + GitHub Pages (frontend) |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Keshabh/MultiAgent-RoastBattle.git
cd MultiAgent-RoastBattle

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# edit .env → add your GROQ_API_KEY

# 4. Run the backend
uvicorn main:app --reload --port 8000
```

```bash
# 5. Open the frontend
# Just open frontend/index.html in Chrome
# Make sure BACKEND_URL = "http://localhost:8000" in index.html
```

### Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📁 Project Structure

```
MultiAgent-RoastBattle/
│   ├── main.py              # FastAPI app + all agents
│   ├── requirements.txt     # Python dependencies
│   ├── Procfile             # Railway deployment config
│   |── .env.example         # Environment variables template
│   |── index.html           # Complete frontend (single file)
└   |── README.md
```
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
