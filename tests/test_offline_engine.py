"""Tests for the offline simulation engine."""
from src.core.offline_engine import (
    swarm_score_offline,
    generate_population_offline,
    run_simulation_offline,
)
from src.agents.world_templates import build_world_offline, _detect_category


def test_detect_category():
    assert _detect_category("Will the Fed cut rates?") == "econ"
    assert _detect_category("Will Congress pass a bill?") == "political"
    assert _detect_category("Will AI replace jobs?") == "tech"


def test_build_world():
    world = build_world_offline("Will the Fed cut rates?")
    assert "entities" in world
    assert "pressures" in world
    assert "timeline" in world
    assert "key_uncertainties" in world
    assert world["question_category"] == "econ"


def test_generate_population():
    world = build_world_offline("Will the Fed cut rates?")
    agents, ms = generate_population_offline("Will the Fed cut rates?", world, n_agents=10)
    assert len(agents) == 10
    assert all("score_history" in a for a in agents)
    assert all("reasoning" in a for a in agents)
    assert all(0 < a["initial_score"] < 1 for a in agents)


def test_generate_population_with_anchor():
    world = build_world_offline("Will the Fed cut rates?")
    agents_low, _ = generate_population_offline("Will the Fed cut rates?", world, n_agents=20, anchor=0.10)
    agents_high, _ = generate_population_offline("Will the Fed cut rates?", world, n_agents=20, anchor=0.90)

    import statistics
    mean_low = statistics.mean([a["initial_score"] for a in agents_low])
    mean_high = statistics.mean([a["initial_score"] for a in agents_high])
    assert mean_low < mean_high, "Higher anchor should produce higher mean scores"


def test_run_simulation():
    world = build_world_offline("Will the Fed cut rates?")
    agents, _ = generate_population_offline("Will the Fed cut rates?", world, n_agents=10, anchor=0.40)
    agents, ms = run_simulation_offline("Will the Fed cut rates?", agents, n_rounds=2)

    # Each agent should have initial + 3 rounds of scores (R1=evidence, R2=critique, R3=self, R4=update)
    # But with n_rounds=2 it adapts
    assert all(len(a["score_history"]) > 1 for a in agents)
    assert all(len(a["memory_stream"]) > 1 for a in agents)


def test_swarm_score_offline():
    result = swarm_score_offline("Will it rain?", n_agents=10, rounds=2, market_price=0.50)
    assert "swarm_probability_yes" in result
    assert "agents" in result
    assert "timing" in result
    assert len(result["agents"]) == 10
    assert 0 <= result["swarm_probability_yes"] <= 1


def test_diversity_preserved():
    """Swarm should maintain diversity (not mode-collapse)."""
    import statistics
    result = swarm_score_offline("Will AI replace 10% of jobs?", n_agents=30, rounds=3, market_price=0.30)
    scores = [a["final_score"] for a in result["agents"]]
    std = statistics.stdev(scores)
    assert std > 0.05, f"StdDev {std} too low — possible mode collapse"
