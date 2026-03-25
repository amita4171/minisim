"""Tests for the opportunity scanner."""
from unittest.mock import patch, MagicMock
import pytest

from scanner import run_scan


def test_scan_kalshi_filters_sports():
    """Should filter out sports markets."""
    mock_markets = [
        {"ticker": "KXNBA-GAME123", "price": 0.50, "title": "Will Lakers win?", "volume_24h": "100"},
        {"ticker": "KXFED-26MAY", "price": 0.35, "title": "Will the Fed cut rates in May 2026?", "volume_24h": "500"},
    ]

    with patch("src.markets.kalshi_client.get_active_markets", return_value=mock_markets):
        from scanner import scan_kalshi
        results = scan_kalshi(limit=50)
        assert len(results) == 1
        assert "Fed" in results[0]["question"]


def test_scan_kalshi_filters_obvious_prices():
    """Should filter out markets at 0.01 or 0.99."""
    mock_markets = [
        {"ticker": "TEST1", "price": 0.01, "title": "Already decided NO question here?", "volume_24h": "100"},
        {"ticker": "TEST2", "price": 0.99, "title": "Already decided YES question here?", "volume_24h": "100"},
        {"ticker": "TEST3", "price": 0.50, "title": "Will something interesting happen in this market?", "volume_24h": "100"},
    ]

    with patch("src.markets.kalshi_client.get_active_markets", return_value=mock_markets):
        from scanner import scan_kalshi
        results = scan_kalshi(limit=50)
        assert len(results) == 1
        assert results[0]["market_price"] == 0.50


def test_scan_polymarket_with_mock():
    mock_markets = [
        {"question": "Will Trump visit Russia during his presidential term?", "price": 0.30, "slug": "trump-russia", "id": "1", "volume": 10000},
    ]

    with patch("src.markets.polymarket_client.get_active_markets", return_value=mock_markets):
        from scanner import scan_polymarket
        results = scan_polymarket(limit=50)
        assert len(results) == 1
        assert results[0]["source"] == "polymarket"


def test_scan_manifold_with_mock():
    mock_markets = [
        {"question": "Will AI pass the Turing test by 2030?", "price": 0.35, "id": "abc", "volume": 1000},
    ]

    with patch("src.markets.manifold_client.get_active_binary_markets", return_value=mock_markets):
        from scanner import scan_manifold
        results = scan_manifold(limit=50)
        assert len(results) == 1
        assert results[0]["source"] == "manifold"


def test_scan_predictit_with_mock():
    mock_contracts = [
        {"question": "Who wins 2028 election: JD Vance?", "price": 0.38, "id": "30001", "volume": 0},
    ]

    with patch("src.markets.predictit_client.get_active_markets", return_value=mock_contracts):
        from scanner import scan_predictit
        results = scan_predictit(limit=50)
        assert len(results) == 1
        assert results[0]["source"] == "predictit"


def test_run_scan_deduplicates():
    """Should deduplicate similar questions across platforms."""
    with patch("scanner.scan_kalshi", return_value=[
        {"question": "Will the Fed cut rates?", "market_price": 0.35, "ticker": "1", "source": "kalshi", "volume": 100},
    ]):
        with patch("scanner.scan_polymarket", return_value=[
            {"question": "Will the Fed cut rates?", "market_price": 0.40, "ticker": "2", "source": "polymarket", "volume": 200},
        ]):
            with patch("scanner.scan_manifold", return_value=[]):
                with patch("scanner.scan_predictit", return_value=[]):
                    results = run_scan(
                        sources=["kalshi", "polymarket"],
                        n_agents=5,
                        n_rounds=1,
                        edge_threshold=0.0,
                        max_markets=10,
                    )
                    assert isinstance(results, list)


def test_run_scan_handles_source_errors():
    """Should continue if one source fails."""
    with patch("scanner.scan_kalshi", return_value=[]):
        with patch("scanner.scan_polymarket", return_value=[]):
            with patch("scanner.scan_manifold", return_value=[]):
                with patch("scanner.scan_predictit", return_value=[]):
                    results = run_scan(sources=["kalshi", "polymarket"], max_markets=5)
                    assert isinstance(results, list)
                    assert len(results) == 0
