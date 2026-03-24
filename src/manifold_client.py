"""
Manifold Markets API client — play-money prediction markets.
No authentication required for public reads.
Base URL: https://api.manifold.markets

Manifold has the broadest coverage: politics, tech, AI, science, culture.
Play-money, but predictions are often well-calibrated due to active community.
"""
from __future__ import annotations

import time
import requests

BASE_URL = "https://api.manifold.markets"
DEFAULT_TIMEOUT = 15


def get_markets(
    limit: int = 100,
    sort: str = "last-bet-time",
    order: str = "desc",
    before: str | None = None,
) -> list[dict]:
    """Fetch markets from Manifold API."""
    params = {"limit": min(limit, 1000), "sort": sort, "order": order}
    if before:
        params["before"] = before

    resp = requests.get(f"{BASE_URL}/v0/markets", params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def search_markets(
    term: str = "",
    sort: str = "most-popular",
    filter: str = "open",
    contract_type: str = "BINARY",
    limit: int = 50,
    offset: int = 0,
    topic_slug: str | None = None,
) -> list[dict]:
    """Search and filter markets."""
    params = {
        "term": term,
        "sort": sort,
        "filter": filter,
        "contractType": contract_type,
        "limit": min(limit, 1000),
        "offset": offset,
    }
    if topic_slug:
        params["topicSlug"] = topic_slug

    resp = requests.get(f"{BASE_URL}/v0/search-markets", params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_market(market_id: str) -> dict:
    """Fetch a single market by ID."""
    resp = requests.get(f"{BASE_URL}/v0/market/{market_id}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_market_by_slug(slug: str) -> dict:
    """Fetch a single market by slug."""
    resp = requests.get(f"{BASE_URL}/v0/slug/{slug}", timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_market(m: dict) -> dict:
    """Normalize a Manifold market into MiniSim format."""
    probability = m.get("probability")
    if probability is None:
        probability = 0.5

    resolution = None
    if m.get("isResolved"):
        res = m.get("resolution", "")
        if res == "YES":
            resolution = 1.0
        elif res == "NO":
            resolution = 0.0

    return {
        "id": m.get("id", ""),
        "question": m.get("question", ""),
        "url": m.get("url", ""),
        "slug": m.get("url", "").split("/")[-1] if m.get("url") else "",
        "price": round(probability, 4),
        "probability": probability,
        "volume": m.get("volume", 0),
        "volume_24h": m.get("volume24Hours", 0),
        "is_resolved": m.get("isResolved", False),
        "resolution": resolution,
        "close_time": m.get("closeTime"),
        "outcome_type": m.get("outcomeType", ""),
        "mechanism": m.get("mechanism", ""),
        "creator": m.get("creatorUsername", ""),
        "source": "manifold",
    }


def get_active_binary_markets(
    limit: int = 100,
    sort: str = "most-popular",
    min_volume: float = 100,
) -> list[dict]:
    """Get active binary markets with meaningful volume."""
    raw = search_markets(
        term="",
        sort=sort,
        filter="open",
        contract_type="BINARY",
        limit=limit,
    )
    parsed = [parse_market(m) for m in raw]
    return [m for m in parsed if m["volume"] >= min_volume and m["question"]]


def get_resolved_binary_markets(limit: int = 100) -> list[dict]:
    """Get resolved binary markets."""
    raw = search_markets(
        term="",
        sort="most-popular",
        filter="resolved",
        contract_type="BINARY",
        limit=limit,
    )
    parsed = [parse_market(m) for m in raw]
    return [m for m in parsed if m["resolution"] is not None]


def search_topic(topic: str, limit: int = 20) -> list[dict]:
    """Search for markets on a specific topic."""
    raw = search_markets(term=topic, sort="most-popular", filter="open", limit=limit)
    return [parse_market(m) for m in raw]
