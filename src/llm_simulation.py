"""
LLM-Powered Simulation — Real agent reasoning via local Ollama or API.

Features:
- Concurrent Ollama calls (5 at a time) for ~4x speedup
- Strong persona prompts to prevent mode collapse
- Per-archetype temperature and diversity instructions
- Graceful fallback to offline engine

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
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm_engine import (
    LLMEngine,
    AGENT_SYSTEM_PROMPT,
    AGENT_INITIAL_PROMPT,
    AGENT_DELIBERATION_PROMPT,
    ANCHOR_PROMPT,
    DIVERSITY_INSTRUCTIONS,
    PERSONA_NUDGES,
    DELIBERATION_NUDGES,
)
from src.offline_engine import (
    BACKGROUNDS,
    PERSONALITIES,
    TEMP_TIERS,
    _detect_category,
    _generate_pressures,
    _make_name,
    build_world_offline,
)
from src.aggregator import aggregate


CONCURRENCY = 2  # 2 is optimal for Apple Silicon GPU (Metal); 5+ causes contention


def _call_llm_initial(engine: LLMEngine, agent_info: dict) -> dict:
    """Generate initial forecast for one agent. Thread-safe."""
    system = AGENT_SYSTEM_PROMPT.format(**agent_info["system_vars"])
    prompt = AGENT_INITIAL_PROMPT.format(**agent_info["prompt_vars"])
    temp = agent_info["temperature"]

    parsed = engine.generate_json(prompt, system=system, temperature=temp)

    if parsed and "initial_score" in parsed:
        return {
            "score": max(0.02, min(0.98, float(parsed["initial_score"]))),
            "confidence": max(0.2, min(0.95, float(parsed.get("confidence", 0.5)))),
            "reasoning": parsed.get("reasoning", ""),
            "key_factors": parsed.get("key_factors", []),
            "from_llm": True,
        }

    return {"from_llm": False}


def _call_llm_deliberation(engine: LLMEngine, agent_info: dict) -> dict:
    """Generate deliberation update for one agent. Thread-safe."""
    system = AGENT_SYSTEM_PROMPT.format(**agent_info["system_vars"])
    prompt = AGENT_DELIBERATION_PROMPT.format(**agent_info["prompt_vars"])
    temp = agent_info["temperature"]

    parsed = engine.generate_json(prompt, system=system, temperature=temp, max_tokens=256)

    if parsed and "updated_score" in parsed:
        return {
            "score": max(0.02, min(0.98, float(parsed["updated_score"]))),
            "confidence": max(0.2, min(0.95, float(parsed.get("confidence", 0.5)))),
            "reflection": parsed.get("reflection", ""),
            "insight": parsed.get("new_insight", ""),
            "from_llm": True,
        }

    return {"from_llm": False}


def run_llm_simulation(
    question: str,
    context: str = "",
    n_agents: int = 20,
    n_rounds: int = 4,
    market_price: float | None = None,
    peer_sample_size: int = 5,
    engine: LLMEngine | None = None,
    concurrency: int = CONCURRENCY,
) -> dict:
    """Run a full LLM-powered swarm simulation with concurrent calls."""
    if engine is None:
        engine = LLMEngine()

    if not engine.is_available():
        print("  LLM not available, falling back to offline engine")
        from src.offline_engine import swarm_score_offline
        return swarm_score_offline(question, context, n_agents, n_rounds, market_price, peer_sample_size)

    print(f"  LLM engine: {engine.backend}/{engine.model} (concurrency={concurrency})")

    # Build world model
    world = build_world_offline(question, context)
    category = world.get("question_category", _detect_category(question))
    pressures = world.get("pressures", {})

    # Context-to-anchor: if no market price, LLM estimates base rate from context
    if market_price is None:
        print("  No market price — computing anchor from context via LLM...")
        anchor_result = engine.generate_json(
            ANCHOR_PROMPT.format(question=question, context=context or "No additional context."),
            temperature=0.3,
            max_tokens=128,
        )
        if anchor_result and "probability" in anchor_result:
            market_price = max(0.03, min(0.97, float(anchor_result["probability"])))
            print(f"  LLM anchor: {market_price:.2f} ({anchor_result.get('reasoning', '')[:60]})")
        else:
            market_price = 0.40
            print(f"  LLM anchor failed, using default: {market_price}")

    context_str = context or "No additional context provided."
    pressures_yes = "; ".join(pressures.get("for_yes", [])[:4])
    pressures_no = "; ".join(pressures.get("for_no", [])[:4])
    pressures_unc = "; ".join(pressures.get("uncertain", [])[:3])

    seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # ── Phase 1: Generate agents concurrently ──
    print(f"  Generating {n_agents} agents (parallel)...")
    gen_start = time.time()

    # Prepare all agent configs
    agent_configs = []
    for i in range(n_agents):
        bg = BACKGROUNDS[i % len(BACKGROUNDS)]
        pers = PERSONALITIES[i % len(PERSONALITIES)]
        tier = TEMP_TIERS[bg.get("temp_tier", "calibrator")]
        temp_tier = bg.get("temp_tier", "calibrator")

        agent_configs.append({
            "idx": i,
            "bg": bg,
            "pers": pers,
            "tier": tier,
            "temp_tier": temp_tier,
            "temperature": tier["temperature"],
            "system_vars": {
                "name": _make_name(i),
                "background": bg["detail"],
                "personality": pers["label"],
                "diversity_instruction": DIVERSITY_INSTRUCTIONS.get(temp_tier, ""),
            },
            "prompt_vars": {
                "background_short": bg["label"],
                "question": question,
                "context": context_str,
                "pressures_yes": pressures_yes,
                "pressures_no": pressures_no,
                "pressures_uncertain": pressures_unc,
                "persona_nudge": PERSONA_NUDGES.get(temp_tier, ""),
            },
        })

    # Run initial forecasts concurrently
    agents = [None] * n_agents
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(_call_llm_initial, engine, cfg): cfg
            for cfg in agent_configs
        }

        done_count = 0
        for future in as_completed(future_to_idx):
            cfg = future_to_idx[future]
            i = cfg["idx"]
            bg = cfg["bg"]
            pers = cfg["pers"]
            tier = cfg["tier"]
            temp_tier = cfg["temp_tier"]

            result = future.result()
            done_count += 1

            if result.get("from_llm"):
                score = result["score"]
                confidence = result["confidence"]
                reasoning = result["reasoning"]
                key_factors = result["key_factors"]
            else:
                # Fallback
                anchor = market_price or 0.40
                deviation = (bg[category] - 0.42) * 1.5
                score = max(0.02, min(0.98, anchor + deviation + rng.gauss(0, tier["jitter_std"])))
                confidence = max(0.2, min(0.95, 0.5 + rng.gauss(0, 0.15)))
                reasoning = f"[Offline fallback] {bg['label']}: P(YES) = {score:.2f}"
                key_factors = []

            agents[i] = {
                "id": i,
                "name": _make_name(i),
                "background_category": bg["label"],
                "background_detail": bg["detail"],
                "personality": pers["label"],
                "temp_tier": temp_tier,
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

            if done_count % 5 == 0 or done_count == n_agents:
                print(f"    [{done_count}/{n_agents}] {bg['label']}: P(YES)={score:.2f}")

    agent_gen_ms = int((time.time() - gen_start) * 1000)
    print(f"  Agent generation: {agent_gen_ms}ms ({agent_gen_ms/1000:.0f}s)")

    # ── Phase 2: Deliberation rounds (concurrent per round) ──
    print(f"  Running {n_rounds} deliberation rounds (parallel)...")
    sim_start = time.time()

    for round_num in range(1, n_rounds + 1):
        round_start = time.time()

        # Prepare all deliberation configs for this round
        delib_configs = []
        for agent in agents:
            peers = [a for a in agents if a["id"] != agent["id"]]
            sampled = rng.sample(peers, min(peer_sample_size, len(peers)))

            peer_text = "\n".join(
                f"- {p['name']} ({p['background_category']}): P(YES)={p['score_history'][-1]:.2f}"
                for p in sampled
            )
            memory_text = "\n".join(agent["memory_stream"][-3:])
            temp_tier = agent["temp_tier"]

            delib_configs.append({
                "agent": agent,
                "sampled": sampled,
                "temperature": TEMP_TIERS[temp_tier]["temperature"],
                "system_vars": {
                    "name": agent["name"],
                    "background": agent["background_detail"],
                    "personality": agent["personality"],
                    "diversity_instruction": DIVERSITY_INSTRUCTIONS.get(temp_tier, ""),
                },
                "prompt_vars": {
                    "name": agent["name"],
                    "background_short": agent["background_category"],
                    "personality": agent["personality"],
                    "question": question,
                    "current_score": agent["score_history"][-1],
                    "peer_opinions": peer_text,
                    "memory": memory_text,
                    "deliberation_nudge": DELIBERATION_NUDGES.get(temp_tier, ""),
                },
            })

        # Run deliberation concurrently
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_cfg = {
                pool.submit(_call_llm_deliberation, engine, cfg): cfg
                for cfg in delib_configs
            }

            for future in as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                agent = cfg["agent"]
                result = future.result()

                if result.get("from_llm"):
                    new_score = result["score"]
                    confidence = result["confidence"]
                    reflection = result.get("reflection", "")
                    insight = result.get("insight", "")
                else:
                    current = agent["score_history"][-1]
                    peer_avg = statistics.mean([p["score_history"][-1] for p in cfg["sampled"]])
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

        round_ms = int((time.time() - round_start) * 1000)
        scores = [a["score_history"][-1] for a in agents]
        std = statistics.stdev(scores) if len(scores) > 1 else 0
        print(f"    Round {round_num}: {round_ms/1000:.0f}s | mean={statistics.mean(scores):.2f} std={std:.3f}")

    sim_loop_ms = int((time.time() - sim_start) * 1000)
    print(f"  Deliberation: {sim_loop_ms}ms ({sim_loop_ms/1000:.0f}s)")

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
        "concurrency": concurrency,
    }
    result["llm_stats"] = engine.get_stats()

    # Auto-log to database for calibration tracking
    try:
        from src.database import Database
        db = Database()
        pred_id = db.log_prediction(
            question=question,
            swarm_probability=result["swarm_probability_yes"],
            market_price=market_price,
            category=category,
            n_agents=n_agents,
            n_rounds=n_rounds,
            mode=f"llm/{engine.model}",
            confidence_interval=result.get("confidence_interval"),
            diversity_score=result.get("diversity_score", 0),
            agents=result.get("agents"),
        )
        result["prediction_id"] = pred_id
        db.close()
    except Exception:
        pass

    return result
