# 🎤 Roast Battle Arena — Multi-Agent AI System

Two celebrities. 5 rounds. One winner.
Multi-agent system that researches real celebrities and generates an escalating roast battle with two different voices and a live rap beat.

---

## Architecture

```
User Input (2 celebrity names)
        ↓
[Research Agent] — DuckDuckGo web + news search for both celebs (parallel)
        ↓
[Profile Builder Agent] — distills raw research into roast ammunition
        ↓
[Battle Master — LangGraph loop]
  Round 1 → Round 2 → Round 3 → Round 4 → Round 5
  Each round escalates. Each roast references the previous.
        ↓
Frontend — displays round by round with:
  - Two different TTS voices (one per celeb)
  - Procedural rap beat via Web Audio API
  - Live round-by-round reveal
```

**Agents used:** Research Agent × 2, Profile Builder × 2, Roast Writer × 2, Battle Master (orchestrator)
**Tools used:** DuckDuckGo Search, DuckDuckGo News, Groq LLM, Browser TTS, Web Audio API

---

## Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Add your Groq API key
```bash
cp .env.example .env
# edit .env and add your GROQ_API_KEY
```

### 3. Run the backend
```bash
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend
Open `frontend/index.html` in Chrome. That's it.

---

## How to use

1. Enter two celebrity names (e.g. "Elon Musk" vs "Mark Zuckerberg")
2. Click "LET THE BATTLE BEGIN"
3. Wait ~30 seconds while agents research both celebrities
4. Watch and listen as the battle unfolds round by round
5. Hit PLAY on the beat bar for background music

---

## What makes this impressive technically

- **LangGraph state machine** with conditional looping (5 rounds)
- **Parallel research** — both celebrities researched simultaneously  
- **Escalating context** — each round reads all previous rounds before writing
- **Real web search** — actual facts, not hallucinated ones
- **Multi-agent debate** — 7 specialized agents with distinct roles
- **Web Audio API** — procedural beat generated in browser, no audio files needed

---

## Resume bullet point

> Built a multi-agent roast battle system using LangGraph with 7 specialized agents (Research, Profile Builder, Battle Master, Roast Writer) that autonomously researches celebrities via web and news APIs, then orchestrates an escalating 5-round debate with real-time voice synthesis and procedural audio generation.
