"""
Kalshi API client for fetching live and historical market data.
No authentication required for public market data endpoints.

Base URL: https://api.elections.kalshi.com/trade-api/v2
"""
from __future__ import annotations

import time
from typing import Iterator

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT = 15


def get_markets(
    status: str | None = None,
    series_ticker: str | None = None,
    event_ticker: str | None = None,
    limit: int = 200,
    max_pages: int = 50,
    min_close_ts: int | None = None,
    min_settled_ts: int | None = None,
) -> list[dict]:
    """Fetch markets from Kalshi API with pagination.

    Args:
        status: Filter by status — 'open', 'closed', 'settled', 'unopened'
        series_ticker: Filter by series (e.g., 'KXFED' for Fed rate decisions)
        event_ticker: Filter by event
        limit: Results per page (max 1000)
        max_pages: Maximum pages to fetch
        min_close_ts: Unix timestamp — only markets closing after this time
        min_settled_ts: Unix timestamp — only markets settled after this time

    Returns:
        List of market dicts with ticker, title, prices, status, result, etc.
    """
    all_markets = []
    cursor = None

    for page in range(max_pages):
        params = {"limit": min(limit, 1000)}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if min_settled_ts:
            params["min_settled_ts"] = min_settled_ts
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(f"{BASE_URL}/markets", params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        markets = data.get("markets", [])
        all_markets.extend(markets)

        cursor = data.get("cursor")
        if not cursor or not markets:
            break

        # Rate limiting — Kalshi enforces strict limits
        time.sleep(1.0)

    return all_markets


def get_market(ticker: str) -> dict:
    """Fetch a single market by ticker."""
    resp = requests.get(f"{BASE_URL}/markets/{ticker}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("market", {})


def get_events(
    status: str | None = None,
    with_nested_markets: bool = False,
    limit: int = 200,
    max_pages: int = 25,
) -> list[dict]:
    """Fetch events from Kalshi API with pagination."""
    all_events = []
    cursor = None

    for page in range(max_pages):
        params = {"limit": min(limit, 200)}
        if status:
            params["status"] = status
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(f"{BASE_URL}/events", params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(f"{BASE_URL}/events", params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        events = data.get("events", [])
        all_events.extend(events)

        cursor = data.get("cursor")
        if not cursor or not events:
            break

        time.sleep(1.0)

    return all_events


def get_event(ticker: str) -> dict:
    """Fetch a single event by ticker."""
    resp = requests.get(f"{BASE_URL}/events/{ticker}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("event", {})


# ---------------------------------------------------------------------------
# Higher-level helpers
# ---------------------------------------------------------------------------

def parse_market(m: dict) -> dict:
    """Normalize a raw Kalshi market dict into a clean format."""
    # Price: use last_price_dollars or midpoint of bid/ask
    last_price = _parse_price(m.get("last_price_dollars"))
    yes_bid = _parse_price(m.get("yes_bid_dollars"))
    yes_ask = _parse_price(m.get("yes_ask_dollars"))

    if last_price is not None:
        price = last_price
    elif yes_bid is not None and yes_ask is not None:
        price = (yes_bid + yes_ask) / 2
    elif yes_bid is not None:
        price = yes_bid
    else:
        price = 0.5

    # Result
    result_raw = m.get("result", "")
    if result_raw == "yes":
        resolution = 1.0
    elif result_raw == "no":
        resolution = 0.0
    else:
        resolution = None

    # Settlement value
    settlement = _parse_price(m.get("settlement_value_dollars"))

    return {
        "ticker": m.get("ticker", ""),
        "event_ticker": m.get("event_ticker", ""),
        "title": m.get("title", ""),
        "subtitle": m.get("subtitle", ""),
        "status": m.get("status", ""),
        "price": round(price, 4),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "last_price": last_price,
        "result": result_raw,
        "resolution": resolution,
        "settlement_value": settlement,
        "close_time": m.get("close_time", ""),
        "volume_24h": m.get("volume_24h_fp", "0"),
        "open_interest": m.get("open_interest_fp", "0"),
        "category": m.get("category", ""),
        "raw": m,
    }


def get_active_markets(limit: int = 200) -> list[dict]:
    """Get currently active/open markets, parsed."""
    raw = get_markets(status="open", limit=limit)
    return [parse_market(m) for m in raw]


def get_settled_markets(
    limit: int = 1000,
    max_pages: int = 50,
    min_settled_ts: int | None = None,
) -> list[dict]:
    """Get settled (resolved) markets, parsed."""
    raw = get_markets(
        status="settled",
        limit=min(limit, 1000),
        max_pages=max_pages,
        min_settled_ts=min_settled_ts,
    )
    return [parse_market(m) for m in raw]


def _parse_price(val) -> float | None:
    """Parse a price value that could be string, float, int, or None."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
