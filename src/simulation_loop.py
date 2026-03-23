"""
K rounds of opinion evolution with peer influence and memory.
Each round: agents see a sample of peer opinions, reflect, and update their score.
Park-style memory stream: observations + reflections accumulate.
"""
from __future__ import annotations

import json
import random
import time
from anthropic import Anthropic

client = Anthropic()

DELIBERATION_PROMPT = """You are {name}, a {background_detail}

Your personality: {personality}

You are participating in round {round_num} of a structured deliberation on this prediction question:
"{question}"

Your current estimate: P(YES) = {current_score:.2f}

Your memory of prior reasoning:
{memory}

Here are opinions from {n_peers} of your peers this round:
{peer_opinions}

Based on your expertise, personality, memory, and the peer opinions above:
1. Reflect on whether the peer opinions change your view
2. Update your probability estimate

Return a JSON object:
{{
  "updated_score": <float between 0.0 and 1.0>,
  "reflection": "2-3 sentences on how peer opinions influenced (or didn't influence) your thinking",
  "new_insight": "One new insight or consideration from this round",
  "confidence": <float between 0.0 and 1.0>
}}

Stay true to your background and personality. Don't just average — reason from your expertise.
Return ONLY valid JSON."""


def run_simulation(
    question: str,
    agents: list[dict],
    n_rounds: int = 3,
    peer_sample_size: int = 5,
) -> tuple[list[dict], int]:
    """Run K rounds of deliberation. Agents see peer samples and update opinions."""
    start = time.time()

    for round_num in range(1, n_rounds + 1):
        for agent in agents:
            # Sample peers (exclude self)
            peers = [a for a in agents if a["id"] != agent["id"]]
            sampled = random.sample(peers, min(peer_sample_size, len(peers)))

            peer_text = "\n".join(
                f"- {p['name']} ({p['background_category']}): P(YES) = {p['score_history'][-1]:.2f}"
                + (f" — \"{p['memory_stream'][-1][:150]}\"" if p['memory_stream'] else "")
                for p in sampled
            )

            memory_text = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(agent["memory_stream"][-5:]))

            current_score = agent["score_history"][-1]

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": DELIBERATION_PROMPT.format(
                            name=agent["name"],
                            background_detail=agent.get("background_detail", agent["background_category"]),
                            personality=agent["personality"],
                            round_num=round_num,
                            question=question,
                            current_score=current_score,
                            memory=memory_text,
                            n_peers=len(sampled),
                            peer_opinions=peer_text,
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

            result = json.loads(raw)

            new_score = max(0.0, min(1.0, float(result["updated_score"])))
            agent["score_history"].append(new_score)
            agent["confidence"] = result.get("confidence", agent.get("confidence", 0.5))

            reflection = result.get("reflection", "")
            new_insight = result.get("new_insight", "")
            agent["memory_stream"].append(
                f"Round {round_num}: Updated to P(YES) = {new_score:.2f}. {reflection} Insight: {new_insight}"
            )

    elapsed_ms = int((time.time() - start) * 1000)
    return agents, elapsed_ms
