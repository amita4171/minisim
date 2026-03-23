"""
Generates N diverse agents with backgrounds, expertise, and personality traits.
Each agent is a dict with: name, background, expertise, personality, bias_tendency, initial_score, reasoning, memory_stream.
"""
from __future__ import annotations

import json
import time
from anthropic import Anthropic

client = Anthropic()

BACKGROUNDS = [
    "Macro Economist — PhD, focuses on monetary policy, inflation dynamics, and central bank behavior",
    "Quantitative Trader — builds systematic models, thinks in probabilities and base rates",
    "Political Analyst — tracks legislative dynamics, lobbying, and regulatory shifts",
    "Investigative Journalist — skeptical, hunts for hidden information and contrarian angles",
    "Venture Capitalist — pattern-matches across industries, optimistic bias toward innovation",
    "Retired Central Banker — deep institutional knowledge of Fed/ECB decision-making",
    "Behavioral Economist — studies cognitive biases, market sentiment, and herd behavior",
    "Geopolitical Strategist — analyzes international relations, trade wars, sanctions",
    "Data Scientist — relies on statistical models, historical patterns, and base rates",
    "Hedge Fund Portfolio Manager — risk-adjusted thinking, contrarian when consensus is strong",
    "Constitutional Lawyer — analyzes legal frameworks, precedent, and regulatory authority",
    "Supply Chain Analyst — tracks real-economy signals, shipping data, commodity flows",
    "Climate Scientist — understands long-term systemic risks and physical economy impacts",
    "Tech Industry Executive — Silicon Valley perspective, innovation-driven worldview",
    "Labor Economist — focuses on employment data, wage dynamics, union activity",
    "Emerging Markets Specialist — tracks capital flows, currency dynamics, contagion risks",
    "Military Intelligence Analyst — structured analytic techniques, red team thinking",
    "Academic Historian — draws parallels from historical episodes, skeptical of 'this time is different'",
    "Insurance Actuary — probabilistic thinker, focuses on tail risks and base rates",
    "Retail Investor / Market Enthusiast — follows social media sentiment, Reddit, fintwit",
]

PERSONALITY_TRAITS = [
    "cautious and evidence-driven",
    "bold and contrarian",
    "analytical and methodical",
    "intuitive and pattern-matching",
    "skeptical and devil's advocate",
    "consensus-seeking and diplomatic",
    "data-obsessed and quantitative",
    "narrative-driven and qualitative",
    "risk-averse and conservative",
    "risk-seeking and opportunistic",
]

AGENT_GEN_PROMPT = """You are creating a simulated analyst for a prediction market swarm.

Question being analyzed: {question}

World context:
{world_summary}

Create a unique analyst with this background: {background}
Personality: {personality}

The analyst should provide their INITIAL probability estimate for the question answering YES.

Return a JSON object:
{{
  "name": "A realistic full name",
  "background_detail": "2-3 sentence expansion of their specific expertise relevant to this question",
  "initial_score": <float between 0.0 and 1.0 — their P(YES) estimate>,
  "reasoning": "3-5 sentence explanation of their initial reasoning, drawing on their specific expertise",
  "confidence": <float between 0.0 and 1.0 — how confident they are in their estimate>,
  "key_factors": ["factor 1", "factor 2", "factor 3"]
}}

Make the reasoning specific to their background. A macro economist should cite monetary indicators.
A political analyst should cite political dynamics. Be authentic to the perspective.
Return ONLY valid JSON."""


def generate_population(question: str, world: dict, n_agents: int = 50) -> tuple[list[dict], int]:
    """Generate N diverse agents with initial opinions on the question."""
    start = time.time()

    # Prepare world summary for agents
    world_summary = _summarize_world(world)

    agents = []
    for i in range(n_agents):
        bg = BACKGROUNDS[i % len(BACKGROUNDS)]
        personality = PERSONALITY_TRAITS[i % len(PERSONALITY_TRAITS)]

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": AGENT_GEN_PROMPT.format(
                        question=question,
                        world_summary=world_summary,
                        background=bg,
                        personality=personality,
                    ),
                }
            ],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        agent = json.loads(raw)
        agent["id"] = i
        agent["background_category"] = bg.split("—")[0].strip()
        agent["personality"] = personality
        agent["memory_stream"] = [
            f"Initial assessment: P(YES) = {agent['initial_score']:.2f}. {agent['reasoning']}"
        ]
        agent["score_history"] = [agent["initial_score"]]
        agents.append(agent)

    elapsed_ms = int((time.time() - start) * 1000)
    return agents, elapsed_ms


def _summarize_world(world: dict) -> str:
    """Create a concise text summary of the world model for agent prompts."""
    parts = []

    if "pressures" in world:
        p = world["pressures"]
        parts.append("Pressures FOR YES: " + "; ".join(p.get("for_yes", [])))
        parts.append("Pressures FOR NO: " + "; ".join(p.get("for_no", [])))
        if p.get("uncertain"):
            parts.append("Uncertain factors: " + "; ".join(p["uncertain"]))

    if "key_uncertainties" in world:
        parts.append("Key uncertainties: " + "; ".join(world["key_uncertainties"]))

    if "base_rate_estimate" in world:
        parts.append(f"Base rate estimate: {world['base_rate_estimate']}")

    if "timeline" in world:
        events = [f"{t['date_or_period']}: {t['event']}" for t in world["timeline"][:5]]
        parts.append("Timeline: " + " | ".join(events))

    return "\n".join(parts)
