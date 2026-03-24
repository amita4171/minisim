"""
Offline simulation engine — no API calls required.
Generates diverse agents with background-specific probability distributions,
simulates peer influence rounds with personality-driven convergence,
and produces template-based reasoning strings.

This lets the full pipeline run locally (e.g., under Claude Code / Max subscription)
without needing a funded Anthropic API console account.

This module is the thin orchestrator; data and helpers live in:
  - src.archetypes   (BACKGROUNDS, TEMP_TIERS, PERSONALITIES, names)
  - src.world_templates (world building, reasoning, evidence, insights)
  - src.alpha        (question alpha, domain expertise)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time

# ── Re-exports from sub-modules (preserve backward compatibility) ──────────

from src.archetypes import (
    BACKGROUNDS,
    TEMP_TIERS,
    PERSONALITIES,
    FIRST_NAMES,
    LAST_NAMES,
    _make_name,
)

from src.world_templates import (
    _detect_category,
    _WORLD_TEMPLATES,
    build_world_offline,
    _generate_pressures,
    _generate_timeline,
    _generate_relationships,
    _REASONING_TEMPLATES,
    _FACTORS,
    _generate_reasoning,
    _EVIDENCE_TEMPLATES,
    _generate_evidence,
    _INSIGHT_BANK,
)

from src.alpha import (
    _compute_question_alpha,
    _compute_domain_expertise,
)


# ---------------------------------------------------------------------------
# Agent factory (offline)
# ---------------------------------------------------------------------------

def generate_population_offline(
    question: str,
    world: dict,
    n_agents: int = 50,
    seed: int | None = None,
    anchor: float | None = None,
) -> tuple[list[dict], int]:
    """Generate N diverse agents with initial scores anchored to market/base rate.

    The anchor (market price or base rate) sets the center of the distribution.
    Each archetype's category bias is treated as a *deviation* from the anchor.
    Question-specific alpha shifts the swarm independently of market price.
    Domain experts get confidence bonuses for higher weight in aggregation.
    """
    start = time.time()
    category = world.get("question_category", _detect_category(question))

    if anchor is None:
        anchor = world.get("base_rate_estimate", 0.40)

    # Question-specific alpha — this is how the swarm adds value over market
    alpha = _compute_question_alpha(question, category)
    adjusted_anchor = max(0.03, min(0.97, anchor + alpha))

    # Deterministic seed from question if not provided
    if seed is None:
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    agents = []
    for i in range(n_agents):
        bg = BACKGROUNDS[i % len(BACKGROUNDS)]
        pers = PERSONALITIES[i % len(PERSONALITIES)]

        # Temperature-stratified jitter (arxiv 2510.01218)
        tier = TEMP_TIERS[bg.get("temp_tier", "calibrator")]

        # Archetype deviation from mean
        archetype_mean = 0.42
        deviation = (bg[category] - archetype_mean) * 1.5

        center = adjusted_anchor + deviation
        jitter = rng.gauss(0, tier["jitter_std"])
        # Contrarians get pushed away from anchor
        if pers["contrarian_factor"] > 0.1:
            jitter += rng.choice([-1, 1]) * pers["contrarian_factor"] * 0.25
        initial_score = max(0.02, min(0.98, center + jitter))

        # Domain expertise bonus for confidence
        domain_bonus = _compute_domain_expertise(bg["label"], question, category)
        confidence = max(0.2, min(0.95, 0.5 + rng.gauss(0, 0.15) + domain_bonus))
        reasoning, key_factors = _generate_reasoning(bg["label"], initial_score, category, rng)

        agent = {
            "id": i,
            "name": _make_name(i),
            "background_category": bg["label"],
            "background_detail": bg["detail"],
            "personality": pers["label"],
            "temp_tier": bg.get("temp_tier", "calibrator"),
            "temperature": tier["temperature"],
            "initial_score": round(initial_score, 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "key_factors": key_factors,
            "score_history": [round(initial_score, 4)],
            "memory_stream": [
                f"Initial assessment: P(YES) = {initial_score:.2f}. {reasoning}"
            ],
            "_convergence_rate": pers["convergence_rate"],
            "_contrarian_factor": pers["contrarian_factor"],
        }
        agents.append(agent)

    elapsed_ms = int((time.time() - start) * 1000)
    return agents, elapsed_ms


# ---------------------------------------------------------------------------
# Opponent pairing
# ---------------------------------------------------------------------------

def _pair_opponents(agents: list[dict], rng: random.Random) -> list[tuple[dict, dict]]:
    """Pair YES-leaning with NO-leaning agents for critique round."""
    yes_agents = [a for a in agents if a["score_history"][-1] > 0.55]
    no_agents = [a for a in agents if a["score_history"][-1] < 0.45]
    uncertain = [a for a in agents if 0.45 <= a["score_history"][-1] <= 0.55]

    pairs = []
    # Pair YES with NO
    min_pairs = min(len(yes_agents), len(no_agents))
    if min_pairs > 0:
        rng.shuffle(yes_agents)
        rng.shuffle(no_agents)
        for i in range(min_pairs):
            pairs.append((yes_agents[i], no_agents[i]))

    # Remaining agents paired randomly with uncertain
    remaining = yes_agents[min_pairs:] + no_agents[min_pairs:] + uncertain
    rng.shuffle(remaining)
    for i in range(0, len(remaining) - 1, 2):
        pairs.append((remaining[i], remaining[i + 1]))

    return pairs


# ---------------------------------------------------------------------------
# 4-Round Structured Deliberation Protocol (offline)
# Research-backed: arxiv 2305.14325 (multi-agent debate), 2404.09127 (calibration)
#
# Round 1: Initial Forecast (already done in agent_factory — agents have scores)
# Round 2: Evidence Exchange — agents share evidence, see peer positions
# Round 3: Critique & Rebuttal — opposing agents challenge each other
# Round 4: Updated Forecast — final revision with reasoning shift summary
# ---------------------------------------------------------------------------

def run_simulation_offline(
    question: str,
    agents: list[dict],
    n_rounds: int = 4,
    peer_sample_size: int = 5,
    seed: int | None = None,
) -> tuple[list[dict], int]:
    """Run structured 4-round deliberation protocol without API calls.

    Round 1: Initial forecast (already in agent data — this round just logs it)
    Round 2: Evidence exchange — agents share evidence, see peer scores
    Round 3: Critique & rebuttal — opposing agents paired for debate
    Round 4: Updated forecast — final revision incorporating all information
    Extra rounds (if n_rounds > 4): additional peer-influence rounds
    """
    start = time.time()
    category = _detect_category(question)

    if seed is None:
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16) + 1000
    rng = random.Random(seed)

    insights = _INSIGHT_BANK.get(category, _INSIGHT_BANK["econ"])

    # --- ROUND 1: Initial Forecast (already done — just record evidence) ---
    for agent in agents:
        agent["evidence"] = _generate_evidence(category, agent, rng)
        agent["evidence_quality_received"] = []
        agent["critique_received"] = ""
        agent["disagreement_type"] = ""
        agent["mind_changed"] = False

    # --- ROUND 2: Evidence Exchange ---
    for agent in agents:
        peers = [a for a in agents if a["id"] != agent["id"]]
        sampled = rng.sample(peers, min(peer_sample_size, len(peers)))

        peer_scores = [p["score_history"][-1] for p in sampled]
        peer_avg = statistics.mean(peer_scores)

        # Agent evaluates peer evidence quality (simulated)
        evidence_ratings = []
        for p in sampled:
            for ev in p.get("evidence", []):
                rating = ev["quality"] + rng.gauss(0, 0.5)
                evidence_ratings.append(round(max(1, min(5, rating)), 1))
        agent["evidence_quality_received"] = evidence_ratings

        # Moderate update based on evidence quality and peer positions
        current = agent["score_history"][-1]
        conv_rate = agent["_convergence_rate"]
        noise = rng.gauss(0, 0.03)

        # Evidence exchange has smaller influence than critique
        delta = (peer_avg - current) * conv_rate * 0.6 + noise
        new_score = round(max(0.02, min(0.98, current + delta)), 4)
        agent["score_history"].append(new_score)

        agent["confidence"] = round(min(0.95, agent["confidence"] + rng.uniform(0.01, 0.03)), 4)
        peer_evidence_summary = "; ".join(
            f"{p['name']}: P(YES)={p['score_history'][-1]:.2f}"
            for p in sampled[:3]
        )
        agent["memory_stream"].append(
            f"Round 2 (Evidence Exchange): Reviewed peer evidence. "
            f"Peers: [{peer_evidence_summary}]. "
            f"Avg evidence quality: {statistics.mean(evidence_ratings):.1f}/5. "
            f"Updated to P(YES) = {new_score:.2f}. {rng.choice(insights)}"
        )

    # --- ROUND 3: Critique & Rebuttal ---
    pairs = _pair_opponents(agents, rng)
    critiqued_ids = set()

    for a1, a2 in pairs:
        critiqued_ids.add(a1["id"])
        critiqued_ids.add(a2["id"])

        score_gap = abs(a1["score_history"][-1] - a2["score_history"][-1])
        if score_gap > 0.20:
            disagreement = "empirical (different facts weighted)"
        elif score_gap > 0.10:
            disagreement = "methodological (different analytical frameworks)"
        else:
            disagreement = "minor (largely aligned with nuance differences)"

        # Each critiques the other — stronger effect than evidence exchange
        for agent, opponent in [(a1, a2), (a2, a1)]:
            current = agent["score_history"][-1]
            opp_score = opponent["score_history"][-1]
            conv_rate = agent["_convergence_rate"]
            contra = agent["_contrarian_factor"]

            # Critique effect: move toward opponent if they have strong evidence
            opp_evidence_quality = statistics.mean([e["quality"] for e in opponent.get("evidence", [{"quality": 3}])])
            critique_strength = (opp_evidence_quality - 3.0) / 5.0  # normalized -0.1 to +0.4

            if contra > 0.1:
                # Contrarians push back against critique
                delta = (current - opp_score) * contra * 0.3
            else:
                delta = (opp_score - current) * conv_rate * (0.5 + critique_strength)

            noise = rng.gauss(0, 0.02)
            new_score = round(max(0.02, min(0.98, current + delta + noise)), 4)
            agent["score_history"].append(new_score)
            agent["critique_received"] = f"Challenged by {opponent['name']} ({opponent['background_category']})"
            agent["disagreement_type"] = disagreement

            agent["confidence"] = round(min(0.95, agent["confidence"] + rng.uniform(0.02, 0.05)), 4)
            agent["memory_stream"].append(
                f"Round 3 (Critique): Debated {opponent['name']} ({opponent['background_category']}, "
                f"P(YES)={opp_score:.2f}). Disagreement: {disagreement}. "
                f"Updated to P(YES) = {new_score:.2f}. {rng.choice(insights)}"
            )

    # Uncritiqued agents still get a round (self-reflection)
    for agent in agents:
        if agent["id"] not in critiqued_ids:
            current = agent["score_history"][-1]
            noise = rng.gauss(0, 0.015)
            new_score = round(max(0.02, min(0.98, current + noise)), 4)
            agent["score_history"].append(new_score)
            agent["memory_stream"].append(
                f"Round 3 (Self-reflection): No direct critique received. "
                f"Maintained P(YES) = {new_score:.2f} with minor refinement. {rng.choice(insights)}"
            )

    # --- ROUND 4: Updated Forecast ---
    for agent in agents:
        peers = [a for a in agents if a["id"] != agent["id"]]
        sampled = rng.sample(peers, min(peer_sample_size, len(peers)))

        peer_scores = [p["score_history"][-1] for p in sampled]
        peer_avg = statistics.mean(peer_scores)

        current = agent["score_history"][-1]
        initial = agent["score_history"][0]
        conv_rate = agent["_convergence_rate"]
        contra = agent["_contrarian_factor"]

        noise = rng.gauss(0, 0.02)
        if contra > 0.1 and abs(current - peer_avg) < 0.12:
            delta = (current - peer_avg) * contra * 0.3
        else:
            delta = (peer_avg - current) * conv_rate * 0.8

        new_score = round(max(0.02, min(0.98, current + delta + noise)), 4)
        agent["score_history"].append(new_score)

        # Track mind-changers
        total_shift = new_score - initial
        agent["mind_changed"] = abs(total_shift) > 0.15

        # Final confidence update
        agent["confidence"] = round(min(0.95, agent["confidence"] + rng.uniform(0.01, 0.04)), 4)

        shift_desc = f"shifted {'up' if total_shift > 0 else 'down'} by {abs(total_shift):.2f}" if abs(total_shift) > 0.05 else "remained largely stable"
        agent["memory_stream"].append(
            f"Round 4 (Final Forecast): P(YES) = {new_score:.2f} "
            f"(from initial {initial:.2f} — {shift_desc}). "
            f"Confidence: {agent['confidence']:.2f}. {rng.choice(insights)}"
        )

    # --- Extra rounds (if n_rounds > 4) — standard peer influence ---
    for round_num in range(5, n_rounds + 1):
        for agent in agents:
            peers = [a for a in agents if a["id"] != agent["id"]]
            sampled = rng.sample(peers, min(peer_sample_size, len(peers)))

            peer_scores = [p["score_history"][-1] for p in sampled]
            peer_avg = statistics.mean(peer_scores)
            current = agent["score_history"][-1]
            conv_rate = agent["_convergence_rate"]
            contra = agent["_contrarian_factor"]

            noise = rng.gauss(0, 0.02 / (round_num - 3))
            if contra > 0.1 and abs(current - peer_avg) < 0.12:
                delta = (current - peer_avg) * contra * 0.2
            else:
                delta = (peer_avg - current) * conv_rate * 0.5

            new_score = round(max(0.02, min(0.98, current + delta + noise)), 4)
            agent["score_history"].append(new_score)
            agent["memory_stream"].append(
                f"Round {round_num} (Extra): P(YES) = {new_score:.2f}. {rng.choice(insights)}"
            )

    elapsed_ms = int((time.time() - start) * 1000)
    return agents, elapsed_ms


# ---------------------------------------------------------------------------
# Full pipeline (offline)
# ---------------------------------------------------------------------------

def swarm_score_offline(
    question: str,
    context: str = "",
    n_agents: int = 50,
    rounds: int = 3,
    market_price: float | None = None,
    peer_sample_size: int = 5,
    use_web_research: bool = False,
) -> dict:
    """Full offline pipeline: world build -> (optional RAG) -> agent gen -> sim -> aggregation."""
    from src.aggregator import aggregate

    world = build_world_offline(question, context)

    # If no market price, try LLM anchor; fall back to keyword-based estimate
    if market_price is None:
        try:
            from src.llm_engine import LLMEngine, ANCHOR_PROMPT
            engine = LLMEngine()
            if engine.is_available():
                result = engine.generate_json(
                    ANCHOR_PROMPT.format(question=question, context=context or "No additional context."),
                    temperature=0.3, max_tokens=128,
                )
                if result and "probability" in result:
                    market_price = max(0.03, min(0.97, float(result["probability"])))
        except Exception:
            pass

    agents, agent_gen_ms = generate_population_offline(question, world, n_agents, anchor=market_price)

    # Optional: web research for information-grounded reasoning
    if use_web_research:
        try:
            from src.web_research import research_question, assign_research_to_agents
            research_bundles = research_question(question, n_perspectives=4)
            agents = assign_research_to_agents(agents, research_bundles)
            world["web_research"] = [
                {"perspective": b["perspective"], "query": b["query"], "n_results": len(b["search_results"])}
                for b in research_bundles
            ]
        except Exception:
            pass  # graceful fallback if web research fails

    agents, sim_loop_ms = run_simulation_offline(question, agents, rounds, peer_sample_size)
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
            "evidence": a.get("evidence", []),
            "critique_received": a.get("critique_received", ""),
            "disagreement_type": a.get("disagreement_type", ""),
            "mind_changed": a.get("mind_changed", False),
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
        "mode": "offline",
    }

    return result
