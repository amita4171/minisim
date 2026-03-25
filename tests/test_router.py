"""Tests for the variance-based prediction router and LLM engine."""
from unittest.mock import patch, MagicMock
import pytest

from src.core.llm_engine import LLMEngine


# ── LLM Engine Tests ──

def test_engine_offline_mode_works():
    """Engine in explicit offline mode is not available."""
    engine = LLMEngine(backend="offline")
    assert engine.backend == "offline"
    assert not engine.is_available()


def test_engine_generate_returns_error_when_offline():
    engine = LLMEngine(backend="offline")
    result = engine.generate("test prompt")
    assert result["error"] == "No LLM backend available"
    assert result["text"] == ""


def test_engine_stats_tracking():
    engine = LLMEngine(backend="offline")
    engine.generate("test")
    stats = engine.get_stats()
    assert stats["calls"] == 1
    assert stats["backend"] == "offline"


def test_engine_generate_json_returns_none_on_empty():
    engine = LLMEngine(backend="offline")
    result = engine.generate_json("test")
    assert result is None


def test_engine_json_parsing_strips_markdown():
    """Test that JSON parsing handles markdown fences."""
    engine = LLMEngine(backend="offline")

    # Mock generate to return markdown-wrapped JSON
    with patch.object(engine, "generate", return_value={
        "text": '```json\n{"score": 0.5}\n```',
        "tokens_in": 10, "tokens_out": 5,
    }):
        engine.backend = "ollama"  # trick to bypass offline check
        result = engine.generate_json("test")
        assert result == {"score": 0.5}


def test_engine_json_parsing_extracts_from_mixed_text():
    engine = LLMEngine(backend="offline")

    with patch.object(engine, "generate", return_value={
        "text": 'Here is the result: {"probability": 0.35, "reasoning": "test"} hope that helps',
        "tokens_in": 10, "tokens_out": 5,
    }):
        engine.backend = "ollama"
        result = engine.generate_json("test")
        assert result["probability"] == 0.35


def test_engine_retry_increments_counter():
    engine = LLMEngine(backend="ollama")
    engine.max_retries = 2

    with patch("requests.post", side_effect=Exception("429 rate limited")):
        result = engine.generate("test")
        assert result["error"]
        assert engine.stats["retries"] >= 1


# ── Router Tests ──

def test_router_single_llm_on_low_variance():
    """When agents agree (low stdev), router should use single LLM."""
    from src.core.router import LOW_VARIANCE_THRESHOLD

    # Mock the LLM engine and simulation
    mock_engine = MagicMock()
    mock_engine.is_available.return_value = True
    mock_engine.generate_json.return_value = {"probability": 0.15, "reasoning": "unlikely"}

    # Mock run_with_initial_only to return agents with low variance
    low_var_result = {
        "swarm_probability_yes": 0.15,
        "agents": [
            {"initial_score": 0.14, "final_score": 0.14},
            {"initial_score": 0.15, "final_score": 0.15},
            {"initial_score": 0.16, "final_score": 0.16},
        ],
        "confidence_interval": [0.10, 0.20],
        "diversity_score": 0.01,
    }

    with patch("src.core.router._run_with_initial_only", return_value=low_var_result):
        with patch("src.core.router._get_single_llm_prediction", return_value=0.12):
            from src.core.router import routed_predict
            result = routed_predict(
                "Will a meteor hit Earth tomorrow?",
                engine=mock_engine,
                n_agents=3,
            )
            assert result["routing"]["route"] == "single_llm"


def test_router_full_swarm_on_high_variance():
    """When agents disagree (high stdev), router should use full swarm."""
    from src.core.router import HIGH_VARIANCE_THRESHOLD

    mock_engine = MagicMock()
    mock_engine.is_available.return_value = True

    high_var_result = {
        "swarm_probability_yes": 0.45,
        "agents": [
            {"initial_score": 0.20, "final_score": 0.30},
            {"initial_score": 0.50, "final_score": 0.50},
            {"initial_score": 0.80, "final_score": 0.70},
        ],
        "confidence_interval": [0.25, 0.65],
        "diversity_score": 0.25,
    }

    swarm_result = {**high_var_result, "routing": None}

    with patch("src.core.router._run_with_initial_only", return_value=high_var_result):
        with patch("src.core.router._run_deliberation", return_value=swarm_result):
            from src.core.router import routed_predict
            result = routed_predict(
                "Will AI replace 10% of jobs by 2028?",
                engine=mock_engine,
                n_agents=3,
                max_rounds=3,
            )
            assert result["routing"]["route"] == "full_swarm"
            assert result["routing"]["rounds_used"] == 3


def test_router_includes_metadata():
    """Router result should include routing metadata."""
    mock_engine = MagicMock()
    mock_engine.is_available.return_value = True
    mock_engine.generate_json.return_value = {"probability": 0.50}

    mid_var_result = {
        "swarm_probability_yes": 0.50,
        "agents": [
            {"initial_score": 0.42},
            {"initial_score": 0.50},
            {"initial_score": 0.58},
        ],
        "confidence_interval": [0.40, 0.60],
        "diversity_score": 0.08,
    }

    with patch("src.core.router._run_with_initial_only", return_value=mid_var_result):
        with patch("src.core.router._run_deliberation", return_value=mid_var_result):
            from src.core.router import routed_predict
            result = routed_predict("Test question?", engine=mock_engine, n_agents=3)
            assert "routing" in result
            assert "route" in result["routing"]
            assert "initial_std" in result["routing"]
            assert "rounds_used" in result["routing"]


def test_router_thresholds():
    """Verify threshold constants are reasonable."""
    from src.core.router import LOW_VARIANCE_THRESHOLD, HIGH_VARIANCE_THRESHOLD
    assert 0 < LOW_VARIANCE_THRESHOLD < HIGH_VARIANCE_THRESHOLD < 1
    assert LOW_VARIANCE_THRESHOLD == 0.05
    assert HIGH_VARIANCE_THRESHOLD == 0.10
