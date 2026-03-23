"""
Bridge to Kalshi prediction markets.
swarm_score_kalshi_market() is a drop-in replacement for a single-LLM scorer.
"""
from __future__ import annotations

from src.world_builder import build_world
from src.agent_factory import generate_population
from src.simulation_loop import run_simulation
from src.aggregator import aggregate


def swarm_score_kalshi_market(
    question: str,
    context: str = "",
    n_agents: int = 50,
    rounds: int = 3,
    market_price: float | None = None,
    peer_sample_size: int = 5,
) -> dict:
    """
    Full pipeline: world build -> agent gen -> simulation -> aggregation.
    Returns the aggregated result dict with timing info.
    """
    # Phase 1: Build world model
    world = build_world(question, context)

    # Phase 2: Generate agent population
    agents, agent_gen_ms = generate_population(question, world, n_agents)

    # Phase 3: Run simulation rounds
    agents, sim_loop_ms = run_simulation(question, agents, rounds, peer_sample_size)

    # Phase 4: Aggregate
    result = aggregate(agents, market_price)

    # Attach timing
    result["timing"] = {
        "world_build_ms": world.get("_build_time_ms", 0),
        "agent_gen_ms": agent_gen_ms,
        "sim_loop_ms": sim_loop_ms,
        "total_ms": world.get("_build_time_ms", 0) + agent_gen_ms + sim_loop_ms,
    }

    # Attach full agent data for JSON export
    result["agents"] = [
        {
            "id": a["id"],
            "name": a["name"],
            "background_category": a["background_category"],
            "background_detail": a.get("background_detail", ""),
            "personality": a["personality"],
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
        "rounds": rounds,
        "peer_sample_size": peer_sample_size,
    }

    return result
