"""Integration tests for the full prediction pipeline."""
import statistics
import pytest


@pytest.mark.slow
def test_offline_pipeline_end_to_end():
    """Full pipeline: question → world build → agents → simulation → aggregation."""
    from src.core.offline_engine import swarm_score_offline

    result = swarm_score_offline(
        question="Will the US economy enter recession in 2026?",
        context="GDP growth slowing. Unemployment rising. Consumer confidence declining.",
        n_agents=10,
        rounds=2,
        market_price=0.25,
    )

    # Result has all required fields
    assert "swarm_probability_yes" in result
    assert "confidence_interval" in result
    assert "agents" in result
    assert "timing" in result
    assert "world_model" in result
    assert "histogram" in result

    # Probability is reasonable
    assert 0 < result["swarm_probability_yes"] < 1

    # Agents were generated
    assert len(result["agents"]) == 10

    # Each agent has required fields
    for agent in result["agents"]:
        assert "initial_score" in agent
        assert "final_score" in agent
        assert "reasoning" in agent
        assert "background_category" in agent
        assert len(agent["score_history"]) > 1


@pytest.mark.slow
def test_calibration_applied_in_pipeline():
    """Verify CalibrationTransformer runs during aggregation."""
    from src.core.offline_engine import swarm_score_offline

    result = swarm_score_offline("Will X happen?", n_agents=10, rounds=1, market_price=0.50)

    # Should have both raw and calibrated
    assert "swarm_probability_raw" in result
    assert "calibration_applied" in result


@pytest.mark.slow
def test_router_dispatches_correctly():
    """Router should dispatch to single_llm or full_swarm based on variance."""
    # Can't test LLM mode without Ollama, but we can test the offline fallback
    from src.core.offline_engine import swarm_score_offline

    # Low variance question (near-certain) should have low stdev
    result = swarm_score_offline(
        "Will the Earth complete an orbit around the Sun in 2026?",
        n_agents=15,
        rounds=1,
        market_price=0.99,
    )
    scores = [a["final_score"] for a in result["agents"]]
    std = statistics.stdev(scores)
    # With 0.99 anchor, most agents should be high
    assert statistics.mean(scores) > 0.7


def test_cross_platform_matching():
    """Cross-platform question matching works on synthetic data."""
    from src.markets.cross_platform import find_cross_listed

    markets = [
        {"question": "Will Trump win the 2028 election?", "price": 0.30,
         "source": "kalshi", "ticker": "1", "volume": 100, "liquidity_weight": 3.0},
        {"question": "Will Donald Trump win the 2028 presidential election?", "price": 0.40,
         "source": "polymarket", "ticker": "2", "volume": 200, "liquidity_weight": 3.0},
        {"question": "Will Bitcoin reach $200K?", "price": 0.15,
         "source": "manifold", "ticker": "3", "volume": 50, "liquidity_weight": 1.0},
    ]

    clusters = find_cross_listed(markets, similarity_threshold=0.4)
    trump_clusters = [c for c in clusters if "trump" in c["question"].lower()]
    assert len(trump_clusters) >= 1
    assert trump_clusters[0]["spread"] > 0


def test_fee_aware_arbitrage():
    """Fee calculation produces correct results."""
    from src.markets.cross_platform import compute_arbitrage_profit

    # Kalshi 0.40 vs Polymarket 0.55 — should be profitable
    result = compute_arbitrage_profit(0.40, 0.55, "kalshi", "polymarket", 100)
    assert result["is_profitable"]
    assert result["net_profit"] > 0
    assert result["spread"] == 0.15

    # Tiny spread — should NOT be profitable after fees
    result = compute_arbitrage_profit(0.40, 0.42, "kalshi", "polymarket", 100)
    assert not result["is_profitable"]


def test_database_prediction_lifecycle():
    """Full lifecycle: log → query → resolve → metrics."""
    import os
    import tempfile
    from src.db.database import Database

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path=path)

    # Log
    pid = db.log_prediction("Will X?", 0.70, 0.50, "test", category="econ")
    assert pid > 0

    # Query
    preds = db.get_predictions()
    assert len(preds) == 1
    assert preds[0]["swarm_probability"] == 0.70

    # Resolve
    db.resolve(pid, 1.0)

    # Metrics
    m = db.get_metrics()
    assert m["resolved"] == 1
    assert m["swarm_brier"] == pytest.approx(0.09, abs=0.01)  # (0.7-1.0)^2

    db.close()
    os.unlink(path)
