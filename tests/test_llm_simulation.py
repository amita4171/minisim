"""Tests for LLM-powered simulation — mocked LLM calls."""
from unittest.mock import patch, MagicMock
import pytest


def _mock_engine(responses=None):
    """Create a mock LLM engine that returns canned responses."""
    engine = MagicMock()
    engine.is_available.return_value = True
    engine.backend = "ollama"
    engine.model = "test-model"
    engine.get_stats.return_value = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0, "total_ms": 0, "retries": 0, "avg_ms_per_call": 0, "backend": "ollama", "model": "test"}

    if responses is None:
        responses = {"initial_score": 0.45, "confidence": 0.7, "reasoning": "Test reasoning", "key_factors": ["a", "b", "c"]}

    engine.generate_json.return_value = responses
    return engine


@pytest.mark.slow
def test_llm_simulation_runs_with_mock():
    """Full LLM simulation with mocked engine."""
    from src.core.llm_simulation import run_llm_simulation

    engine = _mock_engine()
    # Also mock the anchor prompt
    engine.generate_json.side_effect = [
        # Context-to-anchor call
        {"probability": 0.40, "reasoning": "test"},
    ] + [
        # Agent initial forecasts (5 agents)
        {"initial_score": 0.3 + i * 0.1, "confidence": 0.7, "reasoning": f"Agent {i}", "key_factors": ["a"]}
        for i in range(5)
    ] + [
        # Deliberation round 1 (5 agents)
        {"updated_score": 0.35 + i * 0.08, "confidence": 0.75, "reflection": "Updated", "new_insight": "New"}
        for i in range(5)
    ]

    result = run_llm_simulation(
        question="Will test happen?",
        n_agents=5,
        n_rounds=1,
        engine=engine,
    )

    assert "swarm_probability_yes" in result
    assert len(result["agents"]) == 5
    assert result["config"]["mode"] == "llm"
    assert "agents_from_llm" in result
    assert "agents_from_fallback" in result


@pytest.mark.slow
def test_llm_simulation_fallback_on_failure():
    """When LLM calls fail, agents should use offline fallback."""
    from src.core.llm_simulation import run_llm_simulation

    engine = _mock_engine()
    engine.generate_json.return_value = None  # all calls fail

    result = run_llm_simulation(
        question="Will test happen?",
        n_agents=5,
        n_rounds=0,  # skip deliberation
        market_price=0.50,
        engine=engine,
    )

    assert result["agents_from_llm"] == 0
    assert result["agents_from_fallback"] == 5


def test_llm_simulation_offline_fallback_when_unavailable():
    """When engine is offline, should fall back to offline engine."""
    from src.core.llm_simulation import run_llm_simulation

    engine = MagicMock()
    engine.is_available.return_value = False

    with patch("src.core.offline_engine.swarm_score_offline") as mock_offline:
        mock_offline.return_value = {
            "swarm_probability_yes": 0.42,
            "agents": [],
            "timing": {"world_build_ms": 0, "agent_gen_ms": 0, "sim_loop_ms": 0},
        }
        result = run_llm_simulation("Test?", engine=engine)
        mock_offline.assert_called_once()


def test_concurrency_from_env():
    """MINISIM_CONCURRENCY env var should be respected."""
    import os
    with patch.dict(os.environ, {"MINISIM_CONCURRENCY": "3"}):
        # Re-import to pick up env var
        import importlib
        import src.core.llm_simulation as mod
        importlib.reload(mod)
        assert mod.CONCURRENCY == 3
        # Reset
        with patch.dict(os.environ, {"MINISIM_CONCURRENCY": "2"}):
            importlib.reload(mod)
