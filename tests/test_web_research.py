"""Tests for web research module — Tavily + DuckDuckGo."""
from unittest.mock import patch, MagicMock
import pytest

from src.research.web_research import search_web, _search_tavily, _search_duckduckgo, research_question, assign_research_to_agents


def test_search_web_falls_back_to_ddg_without_tavily_key():
    """Without TAVILY_API_KEY, should use DuckDuckGo."""
    with patch("src.research.web_research.TAVILY_API_KEY", ""):
        with patch("src.research.web_research._search_duckduckgo", return_value=[{"title": "test", "snippet": "test", "url": "", "source": "ddg"}]) as mock_ddg:
            results = search_web("test query")
            mock_ddg.assert_called_once()


def test_search_web_uses_tavily_when_key_set():
    """With TAVILY_API_KEY, should try Tavily first."""
    with patch("src.research.web_research.TAVILY_API_KEY", "fake-key"):
        with patch("src.research.web_research._search_tavily", return_value=[{"title": "test", "snippet": "test", "url": "", "source": "tavily"}]) as mock_tav:
            results = search_web("test query")
            mock_tav.assert_called_once()
            assert results[0]["source"] == "tavily"


def test_search_web_tavily_fallback_on_failure():
    """If Tavily fails, should fall back to DuckDuckGo."""
    with patch("src.research.web_research.TAVILY_API_KEY", "fake-key"):
        with patch("src.research.web_research._search_tavily", return_value=[]):
            with patch("src.research.web_research._search_duckduckgo", return_value=[{"title": "ddg", "snippet": "fallback", "url": "", "source": "ddg"}]) as mock_ddg:
                results = search_web("test query")
                mock_ddg.assert_called_once()


def test_duckduckgo_returns_structured_results():
    """DuckDuckGo results should have title, snippet, url, source."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "Abstract": "The Federal Reserve is the central bank.",
        "Heading": "Federal Reserve",
        "AbstractURL": "https://example.com",
        "RelatedTopics": [
            {"Text": "Interest rates are set by the FOMC", "FirstURL": "https://example.com/fomc"},
        ],
    }

    with patch("src.research.web_research.requests.get", return_value=mock_response):
        results = _search_duckduckgo("Federal Reserve")
        assert len(results) >= 1
        assert results[0]["source"] == "duckduckgo_abstract"
        assert "Federal Reserve" in results[0]["title"]


def test_research_question_generates_perspectives():
    """research_question should generate multiple search perspectives."""
    with patch("src.research.web_research.search_web", return_value=[{"title": "t", "snippet": "s", "url": "", "source": "test"}]):
        bundles = research_question("Will the Fed cut rates?", n_perspectives=3)
        assert len(bundles) == 3
        labels = [b["perspective"] for b in bundles]
        assert "direct" in labels


def test_assign_research_creates_asymmetry():
    """Different agents should get different research bundles."""
    bundles = [
        {"perspective": "bull", "summary": "Bull case: likely", "search_results": []},
        {"perspective": "bear", "summary": "Bear case: unlikely", "search_results": []},
    ]
    agents = [
        {"id": 0, "memory_stream": [], "_contrarian_factor": 0.0},
        {"id": 1, "memory_stream": [], "_contrarian_factor": 0.5},
    ]

    result = assign_research_to_agents(agents, bundles)

    # Contrarian agent (id=1) should get bear case
    assert any("bear" in r["perspective"] for r in result[1].get("research", []))


def test_search_web_respects_max_results():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "Abstract": "test",
        "Heading": "test",
        "AbstractURL": "",
        "RelatedTopics": [
            {"Text": f"Topic {i}", "FirstURL": f"url{i}"} for i in range(10)
        ],
    }

    with patch("src.research.web_research.requests.get", return_value=mock_response):
        results = _search_duckduckgo("test", max_results=3)
        assert len(results) <= 3
