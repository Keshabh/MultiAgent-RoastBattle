from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
import re
from dotenv import load_dotenv

load_dotenv()

from groq import Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class BattleState(TypedDict):
    celeb1: str
    celeb2: str
    research1: str
    research2: str
    rounds: List[dict]
    current_round: int
    status: str


def search_celeb(name: str) -> str:
    results = []
    ddgs = DDGS()
    try:
        web = ddgs.text(f"{name} controversies scandals failures embarrassing moments", max_results=5)
        for r in web: results.append(r.get("body", ""))
    except: pass
    try:
        news = ddgs.news(f"{name} latest controversy drama", max_results=4)
        for r in news: results.append(r.get("body", ""))
    except: pass
    try:
        facts = ddgs.text(f"{name} biography career net worth personal life", max_results=3)
        for r in facts: results.append(r.get("body", ""))
    except: pass
    return "\n".join([r for r in results if r])[:4000] or f"Basic info about {name}"


def llm(system: str, user: str, max_tokens: int = 300, max_lines: int = None) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        temperature=1.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    if max_lines:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        raw = "\n".join(lines[:max_lines])
    return raw


def research_agent(state: BattleState) -> BattleState:
    print(f"🔍 Researching {state['celeb1']} and {state['celeb2']}...")
    raw1 = search_celeb(state["celeb1"])
    raw2 = search_celeb(state["celeb2"])

    profile_prompt = """You are the most savage roast writer in Hollywood — written for Oscars, Comedy Central, and the Met Gala.
Extract ONLY the most ruthlessly roastable facts. Maximum ammunition, minimum fluff.

Prioritize:
- Physical appearance jokes (hair, weight, face, fashion disasters)
- Career flops with SPECIFIC names and dollar amounts
- Public humiliations and viral embarrassments  
- Hypocrisy receipts (what they preach vs what they do)
- Failed relationships and personal drama
- Things they are visibly insecure about

Output: 8 punchy bullet points. Each bullet = one ready-to-use roast setup. Be SAVAGE and SPECIFIC."""

    profile1 = llm(profile_prompt, f"Celebrity: {state['celeb1']}\n\nResearch:\n{raw1}", max_tokens=400)
    profile2 = llm(profile_prompt, f"Celebrity: {state['celeb2']}\n\nResearch:\n{raw2}", max_tokens=400)

    return {**state, "research1": profile1, "research2": profile2, "status": "battling"}


def battle_agent(state: BattleState) -> BattleState:
    round_num = state["current_round"]
    rounds_so_far = state["rounds"]
    print(f"🎤 Writing round {round_num}...")

    escalation = {
        1: "ROUND 1 — Sharp opener. Two quick jabs about their most famous embarrassments. Cocky, fast, clean.",
        2: "ROUND 2 — Get personal. Hit something they're clearly insecure about. Reference Round 1 and twist it harder.",
        3: "ROUND 3 — NUCLEAR. This is the final round. Most devastating, funniest, most specific roast possible. Build to ONE mic drop closing line so cold the beat should stop. Make the crowd lose their minds."
    }

    prev_context = ""
    if rounds_so_far:
        prev_context = "=== PREVIOUS ROUNDS — REFERENCE AND TOP THESE ===\n"
        for r in rounds_so_far:
            prev_context += f"Round {r['round']} | {r['celeb']}:\n{r['line']}\n\n"

    ROAST_SYSTEM = """4 LINES. That's it. Not 5. Not 6. FOUR.

- Line 1 + Line 2 rhyme
- Line 3 + Line 4 rhyme  
- Line 4 = the punchline — short, funny, specific, cold
- Simple words only. Max 10 words per line.
- Real facts. Real names. Real numbers.
- Sound like you're clowning someone at lunch, not writing a poem.

EXAMPLE OUTPUT:
You bought Twitter for $44B just to cry online,
Your rockets keep blowing up but sure king, you're doing fine.
Six kids, three baby mamas, what a hall of fame,
Even your own board fired you — twice — and you still came.

OUTPUT = 4 LINES ONLY. No titles. No explanations. No extra text."""

    # Celeb 1 attacks
    roast1 = llm(
        ROAST_SYSTEM,
        f"""{prev_context}
{escalation[round_num]}

YOU ARE: {state['celeb1']}. TARGET: {state['celeb2']}.
FACTS TO USE: {state['research2'][:600]}

WHAT WAS ALREADY SAID (DO NOT REPEAT, TOP THESE):
{prev_context if prev_context else "Nothing yet - this is Round 1."}

4 LINES ONLY. AA BB rhyme. Line 4 = punchline. GO:""",
        max_tokens=120,
        max_lines=4
    )

    # Celeb 2 claps back
    roast2 = llm(
        ROAST_SYSTEM,
        f"""{prev_context}
{state['celeb1']} just said: "{roast1}"

{escalation[round_num]}

YOU ARE: {state['celeb2']}. TARGET: {state['celeb1']}.
FACTS TO USE: {state['research1'][:600]}

WHAT WAS ALREADY SAID (DO NOT REPEAT, TOP THESE):
{prev_context if prev_context else "Nothing yet - this is Round 1."}

4 LINES ONLY. AA BB rhyme. Clap back first. Line 4 = coldest line ever. GO:""",
        max_tokens=120,
        max_lines=4
    )

    new_rounds = rounds_so_far + [
        {"celeb": state["celeb1"], "line": roast1, "round": round_num},
        {"celeb": state["celeb2"], "line": roast2, "round": round_num}
    ]

    return {
        **state,
        "rounds": new_rounds,
        "current_round": round_num + 1,
        "status": "battling" if round_num < 3 else "done"
    }


def should_continue(state: BattleState) -> str:
    return "end" if state["current_round"] > 3 else "battle"


def build_graph():
    graph = StateGraph(BattleState)
    graph.add_node("research", research_agent)
    graph.add_node("battle", battle_agent)
    graph.set_entry_point("research")
    graph.add_edge("research", "battle")
    graph.add_conditional_edges("battle", should_continue, {
        "battle": "battle",
        "end": END
    })
    return graph.compile()

battle_graph = build_graph()


class BattleRequest(BaseModel):
    celeb1: str
    celeb2: str


@app.post("/battle")
async def start_battle(request: BattleRequest):
    print(f"\n🥊 BATTLE: {request.celeb1} vs {request.celeb2}\n")
    initial_state: BattleState = {
        "celeb1": request.celeb1,
        "celeb2": request.celeb2,
        "research1": "",
        "research2": "",
        "rounds": [],
        "current_round": 1,
        "status": "researching"
    }
    final_state = battle_graph.invoke(initial_state)
    return {
        "celeb1": request.celeb1,
        "celeb2": request.celeb2,
        "rounds": final_state["rounds"]
    }


@app.get("/health")
async def health():
    return {"status": "Roast Battle Arena online 🎤"}
