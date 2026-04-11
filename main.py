# ============================================================
# main.py — Celebrity Roast Battle API (Production Upgraded)
# ============================================================

# ── Standard library ────────────────────────────────────────
import asyncio
import logging
import os
import re
from typing import Annotated, TypedDict, List

# ── Third-party ──────────────────────────────────────────────
import diskcache as dc
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ── LangSmith — opt-in via env ───────────────────────────────
_tracing_raw = os.getenv("LANGCHAIN_TRACING_V2", "false").lower().strip()
_tracing_on  = _tracing_raw in ("true", "1", "yes")
if _tracing_on:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "roast-battle")

from langsmith import traceable  # noqa: E402  (must come after env setup)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("roast_battle")

# ── Groq client ──────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Models & temperatures (env-configurable, no redeploy needed) ─
GROQ_MODEL_PROFILE     = os.getenv("GROQ_MODEL_PROFILE",     "llama-3.1-8b-instant")
GROQ_MODEL_ROAST       = os.getenv("GROQ_MODEL_ROAST",       "llama-3.3-70b-versatile")
GROQ_TEMPERATURE_PROFILE = float(os.getenv("GROQ_TEMPERATURE_PROFILE", "0.9"))
GROQ_TEMPERATURE_ROAST   = float(os.getenv("GROQ_TEMPERATURE_ROAST",   "1.1"))

# ── diskcache — two-level cache ──────────────────────────────
#    Level 1: full battle result  key="battle:{celeb_a}:{celeb_b}"
#    Level 2: per-celeb DDG data  key="ddg:{celeb}:{search_type}"
#    TTL: 24 hours = 86400 seconds
CACHE_DIR = os.getenv("CACHE_DIR", "/app/cache")
cache     = dc.Cache(CACHE_DIR)
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", 2_592_000)) #30 days

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="Roast Battle Arena")

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helpers
# ============================================================

def normalize_celeb(name: str) -> str:
    """Lowercase + underscore — used as cache key segment."""
    return name.lower().strip().replace(" ", "_")


def normalize_battle_key(celeb_a: str, celeb_b: str) -> str:
    """
    Order-independent battle cache key.
    'Trump vs Elon' == 'Elon vs Trump' → same cache entry.
    """
    pair = sorted([normalize_celeb(celeb_a), normalize_celeb(celeb_b)])
    return f"battle:{pair[0]}:{pair[1]}"


def clean_llm_output(raw: str, max_lines: int | None = None) -> str:
    """Strip <think> blocks and optionally cap line count."""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    if max_lines:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        raw   = "\n".join(lines[:max_lines])
    return raw


def llm(
    system: str,
    user: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int = 300,
    max_lines: int | None = None,
) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw = response.choices[0].message.content.strip()
    return clean_llm_output(raw, max_lines)


# ============================================================
# DDG Search (called inside Send-API fan-out node)
# ============================================================

def _run_ddg_search(celeb: str, search_type: str) -> str:
    """
    Runs ONE DDG search task for a given celeb + search_type.
    search_type ∈ {"general", "news"}
    Returns raw text (up to 4000 chars) or fallback string.
    """
    ddgs    = DDGS()
    results = []

    try:
        if search_type == "general":
            # Two queries bundled: controversies/scandals + biography/career
            web = ddgs.text(
                f"{celeb} biography controversies career scandals failures embarrassing moments net worth personal life",
                max_results=5,
            )
            for r in web:
                results.append(r.get("body", ""))

        elif search_type == "news":
            news = ddgs.news(
                f"{celeb} latest controversy drama scandal affair",
                max_results=4,
            )
            for r in news:
                results.append(r.get("body", ""))

    except Exception:
        logger.exception("DDG search failed | celeb=%s type=%s", celeb, search_type)

    return "\n".join([r for r in results if r])[:4_000] or f"Basic info about {celeb}"


# ============================================================
# State Schema
# ============================================================

class SearchPayload(TypedDict):
    """Payload sent to ddg_search_node via Send API."""
    celeb:       str   # display name e.g. "Elon Musk"
    search_type: str   # "general" | "news"
    state_key:   str   # which BattleState key to write result into


class BattleState(TypedDict):
    celeb1: str
    celeb2: str

    # ── Level-2 cache / DDG results (all 4 slots) ────────────
    celeb1_general: str
    celeb1_news:    str
    celeb2_general: str
    celeb2_news:    str

    # ── LLM Call 1 output ────────────────────────────────────
    research_brief: str   # structured roast ammo, both celebs

    # ── LLM Call 2 output ────────────────────────────────────
    #    List[dict] with all 3 rounds pre-generated
    rounds: List[dict]

    # ── internal bookkeeping ─────────────────────────────────
    missing_keys:  List[str]   # which DDG keys need fresh fetching
    status:        str


# ============================================================
# LangGraph Nodes
# ============================================================

# ── Node 1: check_battle_cache ───────────────────────────────
def check_battle_cache(state: BattleState) -> BattleState:
    """
    Level-1 cache check.
    If the full battle was already generated today, return it immediately
    and set status='cached' so the graph routes straight to END.
    """
    key    = normalize_battle_key(state["celeb1"], state["celeb2"])
    cached = cache.get(key)

    if cached is not None:
        logger.info("✅ Level-1 cache HIT | key=%s", key)
        return {**state, "rounds": cached, "status": "cached"}

    logger.info("❌ Level-1 cache MISS | key=%s", key)
    return {**state, "status": "researching"}


# ── Node 2: check_ddg_cache ──────────────────────────────────
def check_ddg_cache(state: BattleState) -> BattleState:
    """
    Level-2 cache check — per celeb, per search_type.
    Populates whatever is already cached and records which keys are missing.
    """
    slots = {
        "celeb1_general": (state["celeb1"], "general"),
        "celeb1_news":    (state["celeb1"], "news"),
        "celeb2_general": (state["celeb2"], "general"),
        "celeb2_news":    (state["celeb2"], "news"),
    }

    updates      = {}
    missing_keys = []

    for state_key, (celeb, search_type) in slots.items():
        cache_key = f"ddg:{normalize_celeb(celeb)}:{search_type}"
        hit       = cache.get(cache_key)
        if hit is not None:
            logger.info("✅ DDG cache HIT  | %s", cache_key)
            updates[state_key] = hit
        else:
            logger.info("❌ DDG cache MISS | %s", cache_key)
            missing_keys.append(state_key)

    return {**state, **updates, "missing_keys": missing_keys}


# ── Conditional Edge(fans out send api)────────────────────────────────
def dispatch_searches(state: BattleState):
    """
    Conditional edge function after check_ddg_cache.
    - All cached → route straight to research_synthesizer
    - Any missing → fan-out via Send API
    """
    if not state["missing_keys"]:
        return "research_synthesizer"

    slots = {
        "celeb1_general": (state["celeb1"], "general"),
        "celeb1_news":    (state["celeb1"], "news"),
        "celeb2_general": (state["celeb2"], "general"),
        "celeb2_news":    (state["celeb2"], "news"),
    }

    sends = []
    for state_key in state["missing_keys"]:
        celeb, search_type = slots[state_key]
        logger.info("📡 Dispatching DDG search | celeb=%s type=%s", celeb, search_type)
        sends.append(
            Send(
                "ddg_search_node",
                SearchPayload(
                    celeb=celeb,
                    search_type=search_type,
                    state_key=state_key,
                ),
            )
        )
    return sends


# ── Node 4: ddg_search_node ──────────────────────────────────
def ddg_search_node(payload: SearchPayload) -> BattleState:
    """
    Runs one DDG search, writes result to diskcache, returns partial state.
    Runs in parallel for all missing keys (Send API fan-out).
    """
    celeb       = payload["celeb"]
    search_type = payload["search_type"]
    state_key   = payload["state_key"]

    result    = _run_ddg_search(celeb, search_type)
    cache_key = f"ddg:{normalize_celeb(celeb)}:{search_type}"
    cache.set(cache_key, result, expire=CACHE_TTL)
    logger.info("💾 DDG cache written | %s", cache_key)

    # Return only the relevant slice of BattleState
    return {state_key: result}


# ── Node 5: research_synthesizer (LLM Call 1) ────────────────
RESEARCH_SYSTEM = """You are the most savage roast writer in Hollywood.
Given raw internet research for TWO celebrities, extract the most ruthlessly roastable facts for EACH.

For each celebrity output:
- 8 punchy bullet points
- Each bullet = one ready-to-use roast setup
- Prioritize: physical appearance, career flops with SPECIFIC names/amounts,
  public humiliations, hypocrisy receipts, failed relationships, insecurities

Format EXACTLY as:
=== CELEB_A ===
• bullet
• bullet
...

=== CELEB_B ===
• bullet
• bullet
...

Be SAVAGE, SPECIFIC, and FACTUAL. No fluff."""


@traceable
def research_synthesizer(state: BattleState) -> BattleState:
    """
    LLM Call 1 — 8B model, summarizes all DDG data into roast ammo.
    Both celebs in a single prompt so the LLM sees cross-roast angles.
    """
    logger.info("🧠 LLM Call 1 — Research Synthesizer | model=%s", GROQ_MODEL_PROFILE)
    logger.info(
    "📊 DDG raw sizes | %s general=%d news=%d | %s general=%d news=%d",
    state['celeb1'], len(state['celeb1_general']), len(state['celeb1_news']),
    state['celeb2'], len(state['celeb2_general']), len(state['celeb2_news']),
)

    user_prompt = f"""
=== {state['celeb1'].upper()} — GENERAL RESEARCH ===
{state['celeb1_general'][:1500]}

=== {state['celeb1'].upper()} — NEWS ===
{state['celeb1_news'][:1500]}

=== {state['celeb2'].upper()} — GENERAL RESEARCH ===
{state['celeb2_general'][:1500]}

=== {state['celeb2'].upper()} — NEWS ===
{state['celeb2_news'][:1500]}

Extract savage roast ammo for both {state['celeb1']} and {state['celeb2']}.
""".strip()

    brief = llm(
        RESEARCH_SYSTEM,
        user_prompt,
        model=GROQ_MODEL_PROFILE,
        temperature=GROQ_TEMPERATURE_PROFILE,
        max_tokens=800,
    )

    logger.info("✅ Research brief generated | chars=%d", len(brief))
    return {**state, "research_brief": brief, "status": "generating_battle"}


# ── Node 6: battle_generator (LLM Call 2) ────────────────────
BATTLE_SYSTEM = """You are the most savage rap battle writer alive.
Generate a FULL 3-round roast battle between two celebrities.

THIS IS A REAL BACK-AND-FORTH BATTLE — not 6 independent roasts.
Each roast MUST react to and reference what was just said before it.
Think of it like a rap battle where every bar is a response.

THE ORDER:
- celeb1_round1 → Opening shot. Use facts. Set the tone.
- celeb2_round1 → Directly clap back at what celeb1 just said. Then use facts to hit back harder.
- celeb1_round2 → Counter celeb2's round1. Show anger or mock their response. Use new facts.
- celeb2_round2 → Respond to celeb1's round2 counter. Use new facts. Escalate.
- celeb1_round3 → NUCLEAR. Reference the whole exchange. Use the most devastating facts. End with a mic drop.
- celeb2_round3 → NUCLEAR clap back at everything. Counter every point. Last line = coldest line of the battle.

Rules per roast (NO EXCEPTIONS):
- EXACTLY 4 LINES. Not 3. Not 5. FOUR.
- Line 1 + Line 2 rhyme (AA)
- Line 3 + Line 4 rhyme (BB)
- Line 4 = the punchline — short, cold, specific, devastating
- Max 10 words per line. Simple words. Sound like lunch-table clowning.
- Use REAL facts, REAL names, REAL numbers from the research.
- DO NOT repeat the same fact across rounds — each round must bring NEW ammo.

Round escalation:
- Round 1: Sharp opener. Famous embarrassments. Cocky and fast.
- Round 2: Get personal. Hit insecurities. Reference and counter Round 1.
- Round 3: NUCLEAR. Most devastating. Build to one mic-drop closing line so cold the beat stops.

Output MUST be valid JSON in this exact structure — nothing else:
{
  "rounds": [
    {
      "round": 1,
      "celeb1_roast": "line1\\nline2\\nline3\\nline4",
      "celeb2_roast": "line1\\nline2\\nline3\\nline4",
      "round_winner": "celeb1 or celeb2",
      "judge_verdict": "one sentence judge commentary"
    },
    { "round": 2, ... },
    { "round": 3, ... }
  ],
  "final_winner": "celeb1 or celeb2",
  "final_verdict": "one sentence final verdict"
}

NO markdown. NO backticks. RAW JSON ONLY."""


@traceable
def battle_generator(state: BattleState) -> BattleState:
    """
    LLM Call 2 — 70B model, generates the entire battle in one shot.
    Output is a fully structured JSON blob consumed directly by the frontend.
    """
    logger.info("🎤 LLM Call 2 — Battle Generator | model=%s", GROQ_MODEL_ROAST)

    # Split research brief into per-celeb sections
    brief = state["research_brief"]

    user_prompt = f"""
CELEBRITY 1: {state['celeb1']}
CELEBRITY 2: {state['celeb2']}

ROAST AMMO (use this — real facts only):
{brief}

Generate the full 3-round roast battle as JSON.
In the JSON, refer to celebrities by their actual names, not "celeb1" or "celeb2".
""".strip()

    raw = llm(
        BATTLE_SYSTEM,
        user_prompt,
        model=GROQ_MODEL_ROAST,
        temperature=GROQ_TEMPERATURE_ROAST,
        max_tokens=800,
    )

    # Parse JSON — strip any accidental markdown fences
    raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        import json
        battle_data = json.loads(raw_clean)
        rounds      = battle_data.get("rounds", [])
        logger.info("✅ Battle JSON parsed | rounds=%d", len(rounds))
    except Exception:
        logger.exception("❌ Failed to parse battle JSON | raw=%s", raw_clean[:300])
        raise HTTPException(status_code=500, detail="Battle generation failed — invalid JSON from LLM.")

    # Write Level-1 battle cache
    cache_key = normalize_battle_key(state["celeb1"], state["celeb2"])
    cache.set(cache_key, battle_data, expire=CACHE_TTL)
    logger.info("💾 Level-1 battle cache written | key=%s", cache_key)

    return {**state, "rounds": battle_data, "status": "done"}


# ============================================================
# Graph Routing
# ============================================================

def route_after_battle_cache(state: BattleState) -> str:
    """Skip everything if full battle is already cached."""
    return "end" if state["status"] == "cached" else "check_ddg_cache"

# ============================================================
# Build Graph
# ============================================================

def build_graph():
    graph = StateGraph(BattleState)

    # Nodes
    graph.add_node("check_battle_cache",   check_battle_cache)
    graph.add_node("check_ddg_cache",      check_ddg_cache)
    graph.add_node("ddg_search_node",      ddg_search_node)
    graph.add_node("research_synthesizer", research_synthesizer)
    graph.add_node("battle_generator",     battle_generator)

    # Entry
    graph.set_entry_point("check_battle_cache")

    # Level-1 cache → hit: END, miss: check DDG cache
    graph.add_conditional_edges(
        "check_battle_cache",
        route_after_battle_cache,
        {"end": END, "check_ddg_cache": "check_ddg_cache"},
    )

    # Level-2 cache → all hit: research, any miss: Send API fan-out
    graph.add_conditional_edges(
        "check_ddg_cache",
        dispatch_searches,   # ← moved here as edge function, not a node
    )

    # Fan-in → after all ddg_search_node runs complete → research
    graph.add_edge("ddg_search_node",      "research_synthesizer")

    # LLM Call 1 → LLM Call 2 → END
    graph.add_edge("research_synthesizer", "battle_generator")
    graph.add_edge("battle_generator",     END)

    return graph.compile()

battle_graph = build_graph()


# ============================================================
# API
# ============================================================

class BattleRequest(BaseModel):
    celeb1: str = Field(min_length=1, max_length=128)
    celeb2: str = Field(min_length=1, max_length=128)

    @field_validator("celeb1", "celeb2")
    @classmethod
    def strip_names(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Celebrity name cannot be blank.")
        return stripped


@app.post("/battle")
async def start_battle(request: BattleRequest):
    logger.info("🥊 BATTLE REQUEST | %s vs %s", request.celeb1, request.celeb2)

    initial_state: BattleState = {
        "celeb1":         request.celeb1,
        "celeb2":         request.celeb2,
        "celeb1_general": "",
        "celeb1_news":    "",
        "celeb2_general": "",
        "celeb2_news":    "",
        "research_brief": "",
        "rounds":         [],
        "missing_keys":   [],
        "status":         "starting",
    }

    # Run sync LangGraph graph in a background thread so the
    # async event loop stays free for other requests
    final_state = await asyncio.to_thread(battle_graph.invoke, initial_state)

    result = final_state["rounds"]  # full battle_data dict or cached dict

    return {
        "celeb1":        request.celeb1,
        "celeb2":        request.celeb2,
        "from_cache":    final_state["status"] == "cached",
        "battle":        result,
    }


@app.get("/health")
async def health():
    return {"status": "Roast Battle Arena online 🎤"}
