"""
Polymarket API client — fetches live and historical market data.
No authentication required for public read endpoints.

Uses the Gamma API for market metadata/discovery.
Base URL: https://gamma-api.polymarket.com

Reference: docs.polymarket.com
"""
from __future__ import annotations

import time

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DEFAULT_TIMEOUT = 15


def get_events(
    active: bool | None = None,
    closed: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    order: str = "volume",
    ascending: bool = False,
    tag_id: str | None = None,
) -> list[dict]:
    """Fetch events from Polymarket Gamma API.

    Each event contains nested markets with outcome prices.
    """
    params = {"limit": limit, "offset": offset, "order": order, "ascending": str(ascending).lower()}
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()
    if tag_id:
        params["tag_id"] = tag_id

    resp = requests.get(f"{GAMMA_BASE}/events", params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 429:
        time.sleep(2)
        resp = requests.get(f"{GAMMA_BASE}/events", params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_markets(
    closed: bool | None = None,
    active: bool | None = None,
    limit: int = 50,
    offset: int = 0,
    order: str = "volume",
    ascending: bool = False,
) -> list[dict]:
    """Fetch individual markets from Polymarket Gamma API."""
    params = {"limit": limit, "offset": offset, "order": order, "ascending": str(ascending).lower()}
    if closed is not None:
        params["closed"] = str(closed).lower()
    if active is not None:
        params["active"] = str(active).lower()

    resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 429:
        time.sleep(2)
        resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_event(event_id: str) -> dict:
    """Fetch a single event by ID or slug."""
    resp = requests.get(f"{GAMMA_BASE}/events/{event_id}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_market(market_id: str) -> dict:
    """Fetch a single market by ID or slug."""
    resp = requests.get(f"{GAMMA_BASE}/markets/{market_id}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def search(query: str) -> dict:
    """Search markets, events, and profiles."""
    resp = requests.get(f"{GAMMA_BASE}/search", params={"q": query}, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------

def parse_market(m: dict) -> dict:
    """Normalize a Polymarket market dict into a clean format for MiniSim."""
    import json as _json

    # Parse outcome prices
    prices_raw = m.get("outcomePrices", "")
    yes_price = None
    no_price = None
    if prices_raw:
        try:
            prices = _json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            if isinstance(prices, list) and len(prices) >= 2:
                yes_price = float(prices[0])
                no_price = float(prices[1])
            elif isinstance(prices, list) and len(prices) == 1:
                yes_price = float(prices[0])
        except (ValueError, TypeError, _json.JSONDecodeError):
            pass

    # Fallback to bestBid/bestAsk
    if yes_price is None:
        bid = _safe_float(m.get("bestBid"))
        ask = _safe_float(m.get("bestAsk"))
        if bid is not None and ask is not None:
            yes_price = (bid + ask) / 2
        elif bid is not None:
            yes_price = bid
        elif ask is not None:
            yes_price = ask

    # Determine resolution
    active = m.get("active", False)
    closed = m.get("closed", False)
    resolution = None
    if closed and yes_price is not None:
        # If closed and price near 0 or 1, it's resolved
        if yes_price > 0.95:
            resolution = 1.0
        elif yes_price < 0.05:
            resolution = 0.0

    return {
        "id": m.get("id", ""),
        "question": m.get("question", ""),
        "slug": m.get("slug", ""),
        "category": m.get("category", ""),
        "price": round(yes_price, 4) if yes_price is not None else 0.5,
        "yes_price": yes_price,
        "no_price": no_price,
        "best_bid": _safe_float(m.get("bestBid")),
        "best_ask": _safe_float(m.get("bestAsk")),
        "last_trade_price": _safe_float(m.get("lastTradePrice")),
        "volume": _safe_float(m.get("volume", "0")),
        "volume_24h": _safe_float(m.get("volume24hr", "0")),
        "liquidity": _safe_float(m.get("liquidity", "0")),
        "open_interest": m.get("openInterest", 0),
        "active": active,
        "closed": closed,
        "resolution": resolution,
        "one_day_change": _safe_float(m.get("oneDayPriceChange")),
        "start_date": m.get("startDate", ""),
        "end_date": m.get("endDate", ""),
        "source": "polymarket",
    }


def parse_event(e: dict) -> dict:
    """Normalize a Polymarket event dict."""
    markets = [parse_market(m) for m in e.get("markets", [])]

    return {
        "id": e.get("id", ""),
        "title": e.get("title", ""),
        "slug": e.get("slug", ""),
        "category": e.get("category", ""),
        "active": e.get("active", False),
        "closed": e.get("closed", False),
        "volume": _safe_float(e.get("volume", "0")),
        "liquidity": _safe_float(e.get("liquidity", "0")),
        "markets": markets,
        "source": "polymarket",
    }


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

def get_active_markets(limit: int = 100, min_volume: float = 1000) -> list[dict]:
    """Get currently active Polymarket markets with meaningful volume."""
    raw = get_markets(closed=False, limit=limit, order="volume", ascending=False)
    parsed = [parse_market(m) for m in raw]
    return [m for m in parsed if m["volume"] >= min_volume and m["question"]]


def get_active_events(limit: int = 50) -> list[dict]:
    """Get currently active Polymarket events sorted by volume."""
    raw = get_events(closed=False, limit=limit, order="volume", ascending=False)
    return [parse_event(e) for e in raw]


def get_resolved_markets(limit: int = 100, min_volume: float = 1000) -> list[dict]:
    """Get resolved Polymarket markets (closed with clear YES/NO resolution)."""
    raw = get_markets(closed=True, limit=limit, order="volume", ascending=False)
    parsed = [parse_market(m) for m in raw]
    return [m for m in parsed if m["resolution"] is not None and m["volume"] >= min_volume]


def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
