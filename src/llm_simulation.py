"""
LLM-Powered Simulation — Real agent reasoning via local Ollama or API.

Replaces template-based reasoning in offline_engine.py with actual LLM calls.
Each agent gets a unique prompt reflecting their persona, the question context,
and the world model. Agents produce genuine diverse reasoning.

Usage:
    from src.llm_simulation import run_llm_simulation
    result = run_llm_simulation("Will the Fed cut rates?", n_agents=20, n_rounds=4)
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import time

from src.llm_engine import (
    LLMEngine,
    AGENT_SYSTEM_PROMPT,
    AGENT_INITIAL_PROMPT,
    AGENT_DELIBERATION_PROMPT,
)
from src.offline_engine import (
    BACKGROUNDS,
    PERSONALITIES,
    TEMP_TIERS,
    _detect_category,
    _generate_pressures,
    _generate_timeline,
    _make_name,
    build_world_offline,
)
from src.aggregator import aggregate


def run_llm_simulation(
    question: str,
    context: str = "",
    n_agents: int = 20,
    n_rounds: int = 4,
    market_price: float | None = None,
    peer_sample_size: int = 5,
    engine: LLMEngine | None = None,
) -> dict:
    """Run a full LLM-powered swarm simulation.

    Each agent gets real LLM calls for initial forecast and deliberation rounds.
    Falls back to offline engine if LLM is unavailable.
    """
    if engine is None:
        engine = LLMEngine()

    if not engine.is_available():
        print("  LLM not available, falling back to offline engine")
        from src.offline_engine import swarm_score_offline
        return swarm_score_offline(question, context, n_agents, n_rounds, market_price, peer_sample_size)

    print(f"  LLM engine: {engine.backend}/{engine.model}")

    # Build world model (still template-based — fast)
    world = build_world_offline(question, context)
    category = world.get("question_category", _detect_category(question))
    pressures = world.get("pressures", {})

    # Prepare context string
    context_str = context or "No additional context provided."
    pressures_yes = "; ".join(pressures.get("for_yes", [])[:4])
    pressures_no = "; ".join(pressures.get("for_no", [])[:4])
    pressures_unc = "; ".join(pressures.get("uncertain", [])[:3])

    seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # ── Phase 1: Generate agents with LLM-powered initial forecasts ──
    print(f"  Generating {n_agents} agents...")
    gen_start = time.time()
    agents = []

    for i in range(n_agents):
        bg = BACKGROUNDS[i % len(BACKGROUNDS)]
        pers = PERSONALITIES[i % len(PERSONALITIES)]
        tier = TEMP_TIERS[bg.get("temp_tier", "calibrator")]

        system = AGENT_SYSTEM_PROMPT.format(
            background=bg["detail"],
            personality=pers["label"],
            temp_tier=bg.get("temp_tier", "calibrator"),
        )

        prompt = AGENT_INITIAL_PROMPT.format(
            question=question,
            context=context_str,
            pressures_yes=pressures_yes,
            pressures_no=pressures_no,
            pressures_uncertain=pressures_unc,
        )

        parsed = engine.generate_json(prompt, system=system, temperature=tier["temperature"])

        if parsed and "initial_score" in parsed:
            score = max(0.02, min(0.98, float(parsed["initial_score"])))
            confidence = max(0.2, min(0.95, float(parsed.get("confidence", 0.5))))
            reasoning = parsed.get("reasoning", "")
            key_factors = parsed.get("key_factors", [])
        else:
            # Fallback: use offline anchor-based generation
            anchor = market_price or 0.40
            archetype_mean = 0.42
            deviation = (bg[category] - archetype_mean) * 1.5
            score = max(0.02, min(0.98, anchor + deviation + rng.gauss(0, tier["jitter_std"])))
            confidence = max(0.2, min(0.95, 0.5 + rng.gauss(0, 0.15)))
            reasoning = f"[Offline fallback] Background: {bg['label']}. Estimated P(YES) = {score:.2f}."
            key_factors = []

        agent = {
            "id": i,
            "name": _make_name(i),
            "background_category": bg["label"],
            "background_detail": bg["detail"],
            "personality": pers["label"],
            "temp_tier": bg.get("temp_tier", "calibrator"),
            "temperature": tier["temperature"],
            "initial_score": round(score, 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "key_factors": key_factors,
            "score_history": [round(score, 4)],
            "memory_stream": [f"Initial: P(YES) = {score:.2f}. {reasoning}"],
            "_convergence_rate": pers["convergence_rate"],
            "_contrarian_factor": pers["contrarian_factor"],
        }
        agents.append(agent)

        # Progress indicator
        if (i + 1) % 5 == 0 or i == n_agents - 1:
            print(f"    [{i+1}/{n_agents}] {bg['label']}: P(YES)={score:.2f}")

    agent_gen_ms = int((time.time() - gen_start) * 1000)
    print(f"  Agent generation: {agent_gen_ms}ms")

    # ── Phase 2: LLM-powered deliberation rounds ──
    print(f"  Running {n_rounds} deliberation rounds...")
    sim_start = time.time()

    for round_num in range(1, n_rounds + 1):
        print(f"    Round {round_num}...")
        for agent in agents:
            # Sample peers
            peers = [a for a in agents if a["id"] != agent["id"]]
            sampled = rng.sample(peers, min(peer_sample_size, len(peers)))

            peer_text = "\n".join(
                f"- {p['name']} ({p['background_category']}): P(YES)={p['score_history'][-1]:.2f}"
                for p in sampled
            )
            memory_text = "\n".join(agent["memory_stream"][-3:])

            system = AGENT_SYSTEM_PROMPT.format(
                background=agent["background_detail"],
                personality=agent["personality"],
                temp_tier=agent["temp_tier"],
            )

            prompt = AGENT_DELIBERATION_PROMPT.format(
                question=question,
                current_score=agent["score_history"][-1],
                background=agent["background_detail"],
                peer_opinions=peer_text,
                memory=memory_text,
            )

            tier = TEMP_TIERS[agent["temp_tier"]]
            parsed = engine.generate_json(prompt, system=system, temperature=tier["temperature"])

            if parsed and "updated_score" in parsed:
                new_score = max(0.02, min(0.98, float(parsed["updated_score"])))
                confidence = max(0.2, min(0.95, float(parsed.get("confidence", agent["confidence"]))))
                reflection = parsed.get("reflection", "")
                insight = parsed.get("new_insight", "")
            else:
                # Fallback: simple peer influence
                current = agent["score_history"][-1]
                peer_avg = statistics.mean([p["score_history"][-1] for p in sampled])
                delta = (peer_avg - current) * agent["_convergence_rate"]
                new_score = max(0.02, min(0.98, current + delta + rng.gauss(0, 0.02)))
                confidence = agent["confidence"]
                reflection = "[Offline fallback]"
                insight = ""

            agent["score_history"].append(round(new_score, 4))
            agent["confidence"] = round(confidence, 4)
            agent["memory_stream"].append(
                f"R{round_num}: P(YES)={new_score:.2f}. {reflection} {insight}"
            )

    sim_loop_ms = int((time.time() - sim_start) * 1000)
    print(f"  Deliberation: {sim_loop_ms}ms")

    # ── Phase 3: Aggregate ──
    result = aggregate(agents, market_price)

    result["timing"] = {
        "world_build_ms": world.get("_build_time_ms", 0),
        "agent_gen_ms": agent_gen_ms,
        "sim_loop_ms": sim_loop_ms,
        "total_ms": world.get("_build_time_ms", 0) + agent_gen_ms + sim_loop_ms,
    }

    result["agents"] = [
        {
            "id": a["id"],
            "name": a["name"],
            "background_category": a["background_category"],
            "background_detail": a["background_detail"],
            "personality": a["personality"],
            "temp_tier": a.get("temp_tier", "calibrator"),
            "temperature": a.get("temperature", 0.5),
            "initial_score": a["score_history"][0],
            "final_score": a["score_history"][-1],
            "score_history": a["score_history"],
            "confidence": a.get("confidence", 0.5),
            "reasoning": a.get("reasoning", ""),
            "key_factors": a.get("key_factors", []),
            "memory_stream": a["memory_stream"],
        }
        for a in agents
    ]

    result["world_model"] = {k: v for k, v in world.items() if not k.startswith("_")}
    result["question"] = question
    result["config"] = {
        "n_agents": n_agents,
        "rounds": n_rounds,
        "peer_sample_size": peer_sample_size,
        "mode": "llm",
        "backend": engine.backend,
        "model": engine.model,
    }
    result["llm_stats"] = engine.get_stats()

    return result
