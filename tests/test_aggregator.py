"""Tests for the calibrated aggregation engine."""
import pytest
from src.aggregator import aggregate


def _make_agents(scores, confidences=None):
    """Helper to create agent dicts for testing."""
    if confidences is None:
        confidences = [0.5] * len(scores)
    return [
        {
            "id": i,
            "name": f"Agent_{i}",
            "background_category": "Test",
            "score_history": [s, s],
            "confidence": c,
            "reasoning": "test",
            "memory_stream": ["test"],
        }
        for i, (s, c) in enumerate(zip(scores, confidences))
    ]


def test_aggregate_returns_required_fields():
    agents = _make_agents([0.3, 0.5, 0.7])
    result = aggregate(agents)
    assert "swarm_probability_yes" in result
    assert "confidence_interval" in result
    assert "mean_score" in result
    assert "stdev" in result
    assert "clusters" in result
    assert "top_yes_voices" in result
    assert "top_no_voices" in result
    assert "histogram" in result


def test_aggregate_probability_in_range():
    agents = _make_agents([0.1, 0.3, 0.5, 0.7, 0.9])
    result = aggregate(agents)
    assert 0.0 <= result["swarm_probability_yes"] <= 1.0


def test_aggregate_confidence_interval():
    agents = _make_agents([0.4, 0.5, 0.6])
    result = aggregate(agents)
    ci = result["confidence_interval"]
    assert len(ci) == 2
    assert ci[0] <= result["mean_score"] <= ci[1]


def test_aggregate_with_market_price():
    agents = _make_agents([0.3, 0.5, 0.7])
    result = aggregate(agents, market_price=0.40)
    assert "market_price" in result
    assert "edge" in result
    assert result["market_price"] == 0.40


def test_aggregate_clusters():
    # Clear YES/NO split
    agents = _make_agents([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    result = aggregate(agents)
    assert result["clusters"]["yes_leaning"] >= 2
    assert result["clusters"]["no_leaning"] >= 2


def test_aggregate_histogram_sums_to_n():
    agents = _make_agents([0.1, 0.3, 0.5, 0.7, 0.9])
    result = aggregate(agents)
    total = sum(result["histogram"].values())
    assert total == len(agents)


def test_aggregate_extremized_amplifies():
    # All agents agree -> extremization should amplify
    agents = _make_agents([0.7, 0.72, 0.68, 0.71, 0.69])
    result = aggregate(agents)
    assert result["swarm_probability_yes"] > result["mean_score"]


def test_aggregate_mind_changers():
    agents = [
        {"id": 0, "name": "A", "background_category": "T", "score_history": [0.2, 0.8],
         "confidence": 0.5, "reasoning": "x", "memory_stream": ["x"]},
        {"id": 1, "name": "B", "background_category": "T", "score_history": [0.5, 0.5],
         "confidence": 0.5, "reasoning": "x", "memory_stream": ["x"]},
    ]
    result = aggregate(agents)
    assert len(result["mind_changers"]) >= 1
    assert result["mind_changers"][0]["name"] == "A"


def test_aggregate_dissenting_voices():
    scores = [0.5] * 9 + [0.95]  # one outlier
    agents = _make_agents(scores)
    result = aggregate(agents)
    assert len(result["dissenting_voices"]) >= 1
