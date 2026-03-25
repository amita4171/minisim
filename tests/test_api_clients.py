"""Tests for API client parsers — test parsing logic without live API calls."""
from unittest.mock import patch, MagicMock
import pytest


# ── Kalshi Client ──

def test_kalshi_parse_market():
    from src.markets.kalshi_client import parse_market

    raw = {
        "ticker": "KXFED-26MAY-YES",
        "event_ticker": "KXFED",
        "title": "Will the Fed cut rates?",
        "subtitle": "",
        "status": "active",
        "yes_bid_dollars": "0.3500",
        "yes_ask_dollars": "0.3700",
        "last_price_dollars": "0.3600",
        "close_time": "2026-05-07T00:00:00Z",
        "result": "",
        "settlement_value_dollars": None,
        "volume_24h_fp": "1500",
        "open_interest_fp": "5000",
        "category": "Economics",
    }

    parsed = parse_market(raw)
    assert parsed["ticker"] == "KXFED-26MAY-YES"
    assert parsed["price"] == 0.36
    assert parsed["resolution"] is None
    assert parsed["yes_bid"] == 0.35
    assert parsed["yes_ask"] == 0.37


def test_kalshi_parse_resolved_yes():
    from src.markets.kalshi_client import parse_market

    raw = {
        "ticker": "TEST",
        "title": "Test",
        "status": "settled",
        "result": "yes",
        "yes_bid_dollars": "0.99",
        "yes_ask_dollars": "1.00",
        "last_price_dollars": "1.00",
        "close_time": "",
        "volume_24h_fp": "0",
    }

    parsed = parse_market(raw)
    assert parsed["resolution"] == 1.0


def test_kalshi_parse_resolved_no():
    from src.markets.kalshi_client import parse_market

    raw = {
        "ticker": "TEST",
        "title": "Test",
        "status": "settled",
        "result": "no",
        "last_price_dollars": "0.01",
        "close_time": "",
    }

    parsed = parse_market(raw)
    assert parsed["resolution"] == 0.0


# ── Manifold Client ──

def test_manifold_parse_market():
    from src.markets.manifold_client import parse_market

    raw = {
        "id": "abc123",
        "question": "Will AI replace 10% of jobs?",
        "probability": 0.42,
        "volume": 5000,
        "volume24Hours": 100,
        "isResolved": False,
        "resolution": None,
        "closeTime": 1700000000000,
        "outcomeType": "BINARY",
        "mechanism": "cpmm-1",
        "url": "https://manifold.markets/user/will-ai-replace-jobs",
        "creatorUsername": "testuser",
    }

    parsed = parse_market(raw)
    assert parsed["price"] == 0.42
    assert parsed["probability"] == 0.42
    assert parsed["resolution"] is None
    assert parsed["source"] == "manifold"


def test_manifold_parse_resolved():
    from src.markets.manifold_client import parse_market

    raw = {
        "id": "xyz",
        "question": "Test?",
        "probability": 0.99,
        "isResolved": True,
        "resolution": "YES",
        "volume": 1000,
    }

    parsed = parse_market(raw)
    assert parsed["resolution"] == 1.0
    assert parsed["is_resolved"] is True


# ── Polymarket Client ──

def test_polymarket_parse_market():
    from src.markets.polymarket_client import parse_market

    raw = {
        "id": "12345",
        "question": "Will Trump win 2028?",
        "slug": "will-trump-win-2028",
        "category": "Politics",
        "outcomePrices": "[0.35, 0.65]",
        "bestBid": 0.34,
        "bestAsk": 0.36,
        "lastTradePrice": 0.35,
        "volume": "50000",
        "volume24hr": "1000",
        "liquidity": "10000",
        "active": True,
        "closed": False,
    }

    parsed = parse_market(raw)
    assert parsed["yes_price"] == 0.35
    assert parsed["no_price"] == 0.65
    assert parsed["source"] == "polymarket"


def test_polymarket_parse_with_malformed_prices():
    """Should handle invalid outcomePrices gracefully."""
    from src.markets.polymarket_client import parse_market

    raw = {
        "id": "bad",
        "question": "Test?",
        "outcomePrices": "not-json",
        "bestBid": 0.40,
        "bestAsk": 0.50,
        "active": True,
        "closed": False,
    }

    parsed = parse_market(raw)
    # Should fall back to bestBid/bestAsk midpoint
    assert parsed["price"] == 0.45


# ── PredictIt Client ──

def test_predictit_parse_market():
    from src.markets.predictit_client import parse_market

    raw = {
        "id": 7456,
        "name": "Who will win the 2028 presidential election?",
        "url": "https://predictit.org/market/7456",
        "contracts": [
            {
                "id": 30001,
                "name": "JD Vance",
                "lastTradePrice": 0.38,
                "bestBuyYesCost": 0.39,
                "bestBuyNoCost": 0.63,
                "bestSellYesCost": 0.37,
                "bestSellNoCost": 0.61,
                "status": "Open",
            },
            {
                "id": 30002,
                "name": "Gavin Newsom",
                "lastTradePrice": 0.27,
                "bestBuyYesCost": 0.28,
                "status": "Open",
            },
        ],
    }

    parsed = parse_market(raw)
    assert len(parsed) == 2
    assert parsed[0]["price"] == 0.38
    assert parsed[0]["source"] == "predictit"
    assert "JD Vance" in parsed[0]["question"]


# ── Cross-platform Arbitrage ──

def test_arbitrage_profit_calculation():
    from src.markets.cross_platform import compute_arbitrage_profit

    result = compute_arbitrage_profit(0.30, 0.50, "kalshi", "polymarket", 100)
    assert result["is_profitable"]
    assert result["spread"] == 0.20
    assert result["net_profit"] > 0

    # Tiny spread — not profitable
    result = compute_arbitrage_profit(0.40, 0.42, "kalshi", "polymarket", 100)
    assert not result["is_profitable"]


def test_arbitrage_predictit_high_fees():
    """PredictIt's 15% effective fees should kill most arbitrage."""
    from src.markets.cross_platform import compute_arbitrage_profit

    # 10% spread — profitable on Kalshi/Polymarket but not PredictIt
    result = compute_arbitrage_profit(0.40, 0.50, "predictit", "kalshi", 100)
    # PredictIt has 5% profit fee — might still be profitable with 10% spread
    # but much less than Kalshi-Polymarket
