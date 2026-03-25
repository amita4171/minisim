"""
Prediction Router — Routes predictions based on question uncertainty.

The benchmark showed:
- Single LLM beats swarm on easy questions (stdev < 0.05)
- Swarm beats single LLM on uncertain questions (stdev > 0.10)
- Deliberation HURTS on clear-cut questions (dilutes correct extreme estimates)

The router:
1. Generates N agents with initial forecasts (no deliberation yet)
2. Measures initial variance (stdev of initial scores)
3. Routes:
   - Low variance (< 0.05): use single LLM anchor — agents agree, deliberation adds noise
   - Medium variance (0.05-0.10): run 1 round of deliberation — light refinement
   - High variance (> 0.10): run full 3-4 round deliberation — genuine disagreement, swarm adds value

This turns the benchmark weakness into a product feature:
"MiniSim knows when to think harder."

Usage:
    from src.router import routed_predict
    result = routed_predict("Will the Fed cut rates?", market_price=0.35)
"""
from __future__ import annotations

import statistics
import time

from src.core.llm_engine import LLMEngine, ANCHOR_PROMPT


# Routing thresholds (validated on 10-question benchmark)
LOW_VARIANCE_THRESHOLD = 0.05   # below: single LLM wins
HIGH_VARIANCE_THRESHOLD = 0.10  # above: full swarm wins


def routed_predict(
    question: str,
    context: str = "",
    n_agents: int = 15,
    market_price: float | None = None,
    peer_sample_size: int = 5,
    engine: LLMEngine | None = None,
    max_rounds: int = 3,
) -> dict:
    """Smart prediction routing based on initial agent variance.

    Returns the same result format as swarm_score_offline / run_llm_simulation,
    plus routing metadata.
    """
    if engine is None:
        engine = LLMEngine()

    start = time.time()

    # Step 1: Generate initial agent forecasts (no deliberation)
    if engine.is_available():
        result = _run_with_initial_only(question, context, n_agents, market_price,
                                         peer_sample_size, engine)
    else:
        from src.core.offline_engine import swarm_score_offline
        result = swarm_score_offline(question, context, n_agents, 1, market_price, peer_sample_size)

    initial_scores = [a["initial_score"] for a in result.get("agents", [])]
    initial_std = statistics.stdev(initial_scores) if len(initial_scores) > 1 else 0

    # Step 2: Route based on variance
    if initial_std < LOW_VARIANCE_THRESHOLD:
        route = "single_llm"
        route_reason = f"Low variance ({initial_std:.3f} < {LOW_VARIANCE_THRESHOLD}) — agents agree, using single LLM anchor"
        rounds_used = 0

        # Use the single LLM anchor as the prediction
        anchor_p = _get_single_llm_prediction(engine, question, context)
        if anchor_p is not None:
            result["swarm_probability_yes"] = anchor_p
            # Recompute edge
            if market_price is not None:
                result["edge"] = round(anchor_p - market_price, 4)
                result["swarm_vs_market_delta"] = result["edge"]

    elif initial_std < HIGH_VARIANCE_THRESHOLD:
        route = "light_deliberation"
        route_reason = f"Medium variance ({initial_std:.3f}) — 1 round of refinement"
        rounds_used = 1

        # Run 1 round of deliberation
        result = _run_deliberation(question, context, n_agents, 1, market_price,
                                    peer_sample_size, engine)

    else:
        route = "full_swarm"
        route_reason = f"High variance ({initial_std:.3f} > {HIGH_VARIANCE_THRESHOLD}) — full {max_rounds}-round deliberation"
        rounds_used = max_rounds

        # Run full deliberation
        result = _run_deliberation(question, context, n_agents, max_rounds, market_price,
                                    peer_sample_size, engine)

    total_ms = int((time.time() - start) * 1000)

    # Add routing metadata
    result["routing"] = {
        "route": route,
        "reason": route_reason,
        "initial_std": round(initial_std, 4),
        "rounds_used": rounds_used,
        "low_threshold": LOW_VARIANCE_THRESHOLD,
        "high_threshold": HIGH_VARIANCE_THRESHOLD,
    }

    if "timing" in result:
        result["timing"]["total_ms"] = total_ms
        result["timing"]["route"] = route
    else:
        result["timing"] = {"total_ms": total_ms, "route": route}

    print(f"  Router: {route} (initial_std={initial_std:.3f}, rounds={rounds_used})")

    return result


def _get_single_llm_prediction(engine: LLMEngine, question: str, context: str) -> float | None:
    """Get a single-shot LLM probability estimate."""
    result = engine.generate_json(
        ANCHOR_PROMPT.format(question=question, context=context or "No additional context."),
        temperature=0.3,
        max_tokens=128,
    )
    if result and "probability" in result:
        return max(0.02, min(0.98, float(result["probability"])))
    return None


def _run_with_initial_only(
    question: str, context: str, n_agents: int, market_price: float | None,
    peer_sample_size: int, engine: LLMEngine,
) -> dict:
    """Generate agents with initial forecasts only (no deliberation rounds)."""
    from src.core.llm_simulation import run_llm_simulation
    return run_llm_simulation(
        question=question,
        context=context,
        n_agents=n_agents,
        n_rounds=0,  # no deliberation
        market_price=market_price,
        peer_sample_size=peer_sample_size,
        engine=engine,
    )


def _run_deliberation(
    question: str, context: str, n_agents: int, n_rounds: int,
    market_price: float | None, peer_sample_size: int, engine: LLMEngine,
) -> dict:
    """Run full LLM simulation with deliberation."""
    from src.core.llm_simulation import run_llm_simulation
    return run_llm_simulation(
        question=question,
        context=context,
        n_agents=n_agents,
        n_rounds=n_rounds,
        market_price=market_price,
        peer_sample_size=peer_sample_size,
        engine=engine,
    )
