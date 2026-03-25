"""Tests for data feeds — FRED, Yahoo Finance, Google News."""
from unittest.mock import patch, MagicMock
import pytest

from src.research.data_feeds import get_macro_context, get_stock_price, get_news, get_news_context, build_rich_context, _cached


def test_cached_returns_same_value():
    """Cache should return same value within TTL."""
    call_count = 0
    def fetch():
        nonlocal call_count
        call_count += 1
        return "result"

    r1 = _cached("test_key_123", fetch, ttl=60)
    r2 = _cached("test_key_123", fetch, ttl=60)
    assert r1 == r2
    assert call_count == 1  # only called once


def test_get_macro_context_returns_string():
    """Should return a string with macro indicators."""
    # Without FRED_API_KEY, falls back to hardcoded data
    result = get_macro_context()
    assert isinstance(result, str)
    assert "macro" in result.lower() or "rate" in result.lower() or "inflation" in result.lower()


def test_get_stock_price_with_mock():
    """Mock Yahoo Finance response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "chart": {
            "result": [{
                "meta": {
                    "regularMarketPrice": 185.50,
                    "previousClose": 183.20,
                    "currency": "USD",
                }
            }]
        }
    }

    with patch("src.research.data_feeds.requests.get", return_value=mock_resp):
        result = get_stock_price("AAPL")
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 185.50
        assert result["change_pct"] is not None


def test_get_stock_price_handles_failure():
    """Should return None on API failure."""
    with patch("src.research.data_feeds.requests.get", side_effect=Exception("timeout")):
        result = get_stock_price("INVALID")
        assert result is None


def test_get_news_with_mock():
    """Mock Google News RSS response."""
    mock_xml = """<?xml version="1.0"?>
    <rss><channel>
        <item><title>Fed holds rates steady</title><pubDate>Mon, 24 Mar 2026</pubDate><source>Reuters</source></item>
        <item><title>Inflation drops to 2.5%</title><pubDate>Mon, 24 Mar 2026</pubDate><source>Bloomberg</source></item>
    </channel></rss>"""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = mock_xml.encode()

    with patch("src.research.data_feeds.requests.get", return_value=mock_resp):
        results = get_news("Federal Reserve", max_results=5)
        assert len(results) == 2
        assert "Fed" in results[0]["title"]


def test_get_news_handles_failure():
    """Should return empty list on failure."""
    with patch("src.research.data_feeds.requests.get", side_effect=Exception("network error")):
        results = get_news("test")
        assert results == []


def test_get_news_context_returns_string():
    """Should return formatted string with headlines."""
    with patch("src.research.data_feeds._cached", return_value=[
        {"title": "Test headline 1"},
        {"title": "Test headline 2"},
    ]):
        result = get_news_context("test question")
        assert isinstance(result, str)


def test_build_rich_context_combines_feeds():
    """Should combine news + macro + market data."""
    with patch("src.research.data_feeds.get_news_context", return_value="News: test headlines"):
        with patch("src.research.data_feeds.get_macro_context", return_value="Macro: rates at 4.5%"):
            with patch("src.research.data_feeds.get_market_snapshot", return_value="Markets: S&P at 5800"):
                result = build_rich_context("test question")
                assert "News" in result or "Macro" in result
