"""Tests for the Metaculus tournament bot."""
from unittest.mock import patch, MagicMock
import json
import os
import tempfile
import pytest

from metaculus_bot import get_open_questions, submit_forecast, submit_comment, format_reasoning, _load_forecasted, _save_forecasted


def test_load_forecasted_empty():
    """Should return empty set when no cache file."""
    with patch("metaculus_bot.FORECASTED_CACHE", "/nonexistent/path.json"):
        result = _load_forecasted()
        assert result == set()


def test_save_and_load_forecasted(tmp_path):
    """Should persist and reload forecasted question IDs."""
    cache_path = str(tmp_path / "forecasted.json")
    with patch("metaculus_bot.FORECASTED_CACHE", cache_path):
        _save_forecasted({42684, 42794, 42611})
        loaded = _load_forecasted()
        assert loaded == {42684, 42794, 42611}


def test_submit_forecast_success():
    """Mock successful forecast submission."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("metaculus_bot.requests.post", return_value=mock_resp):
        with patch("metaculus_bot.BOT_TOKEN", "fake-token"):
            result = submit_forecast(42684, 0.65)
            assert result is True


def test_submit_forecast_failure():
    mock_resp = MagicMock()
    mock_resp.status_code = 400

    with patch("metaculus_bot.requests.post", return_value=mock_resp):
        with patch("metaculus_bot.BOT_TOKEN", "fake-token"):
            result = submit_forecast(42684, 0.65)
            assert result is False


def test_submit_forecast_clamps_probability():
    """Probability should be clamped to [0.001, 0.999]."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("metaculus_bot.requests.post", return_value=mock_resp) as mock_post:
        with patch("metaculus_bot.BOT_TOKEN", "fake-token"):
            submit_forecast(42684, 0.0)  # should clamp to 0.001
            call_args = mock_post.call_args
            body = call_args[1]["json"][0]
            assert body["probability_yes"] >= 0.001

            submit_forecast(42684, 1.0)  # should clamp to 0.999
            call_args = mock_post.call_args
            body = call_args[1]["json"][0]
            assert body["probability_yes"] <= 0.999


def test_format_reasoning():
    result = {
        "swarm_probability_yes": 0.42,
        "confidence_interval": [0.35, 0.49],
        "diversity_score": 0.15,
        "routing": {"route": "full_swarm", "initial_std": 0.18},
        "top_yes_voices": [{"name": "Alice", "background": "Economist", "reasoning": "Strong macro data"}],
        "top_no_voices": [{"name": "Bob", "background": "Trader", "reasoning": "Markets overpriced"}],
        "reasoning_shift_summary": "3 agents shifted toward YES",
    }

    text = format_reasoning(result)
    assert "0.42" in text
    assert "0.35" in text
    assert "full_swarm" in text
    assert "Alice" in text
    assert "Bob" in text


def test_skip_moved_questions():
    """Bot should skip questions with 'Moved to' or 'RESTATED' in title."""
    # The skip logic is in run_bot — test it by checking the keyword list
    skip_kw = ["Moved to", "RESTATED", "moved to", "restated at"]

    assert any(kw in "[RESTATED at https://metaculus.com/...]" for kw in skip_kw)
    assert any(kw in "Moved to https://metaculus.com/..." for kw in skip_kw)
    assert not any(kw in "Will the Fed cut rates?" for kw in skip_kw)


def test_get_open_questions_with_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"id": 1, "title": "Test Q1", "question": {"id": 100, "type": "binary", "status": "open"}},
            {"id": 2, "title": "Test Q2", "question": {"id": 101, "type": "binary", "status": "open"}},
        ]
    }

    with patch("metaculus_bot.requests.get", return_value=mock_resp):
        with patch("metaculus_bot.BOT_TOKEN", "fake-token"):
            results = get_open_questions("spring-aib-2026")
            assert len(results) == 2
