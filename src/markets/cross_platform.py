"""
Cross-Platform Prediction Aggregator — matches questions across platforms
and detects arbitrage opportunities.

Sources: Kalshi, Polymarket, Manifold Markets, PredictIt

Key features:
- Fuzzy question matching across platforms
- Price discrepancy detection (arbitrage signals)
- Consensus probability (weighted by platform liquidity)
- Swarm vs multi-platform consensus comparison
"""
from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher

from src.markets.arbitrage import compute_arbitrage_profit, find_profitable_arbitrage, PLATFORM_FEES


def similarity(a: str, b: str) -> float:
    """Compute string similarity between two questions."""
    a_clean = _normalize(a)
    b_clean = _normalize(b)
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def _normalize(text: str) -> str:
    """Normalize question text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[?!.,;:\'"()\[\]{}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove common filler words
    for word in ["will", "the", "be", "in", "by", "before", "after", "a", "an", "of", "to", "and", "or"]:
        text = re.sub(rf'\b{word}\b', '', text)
    return text.strip()


def fetch_all_markets(
    sources: list[str] | None = None,
    limit_per_source: int = 100,
) -> list[dict]:
    """Fetch active markets from all configured sources."""
    if sources is None:
        sources = ["kalshi", "polymarket", "manifold", "predictit"]

    all_markets = []

    if "kalshi" in sources:
        try:
            from src.markets.kalshi_client import get_active_markets
            markets = get_active_markets(limit=limit_per_source)
            for m in markets:
                if m["price"] <= 0.05 or m["price"] >= 0.95:
                    continue
                q = m.get("title", "")
                if not q.endswith("?"):
                    q += "?"
                all_markets.append({
                    "question": q,
                    "price": m["price"],
                    "source": "kalshi",
                    "ticker": m.get("ticker", ""),
                    "volume": float(m.get("volume_24h", 0) or 0),
                    "liquidity_weight": 3.0,  # real money, high trust
                })
        except Exception as e:
            print(f"  Kalshi error: {e}")

    if "polymarket" in sources:
        try:
            from src.markets.polymarket_client import get_active_markets
            markets = get_active_markets(limit=limit_per_source, min_volume=1000)
            for m in markets:
                if m["price"] <= 0.05 or m["price"] >= 0.95:
                    continue
                all_markets.append({
                    "question": m["question"],
                    "price": m["price"],
                    "source": "polymarket",
                    "ticker": m.get("slug", m["id"]),
                    "volume": float(m.get("volume", 0) or 0),
                    "liquidity_weight": 3.0,  # real money (crypto)
                })
        except Exception as e:
            print(f"  Polymarket error: {e}")

    if "manifold" in sources:
        try:
            from src.markets.manifold_client import get_active_binary_markets
            markets = get_active_binary_markets(limit=limit_per_source, min_volume=100)
            for m in markets:
                if m["price"] <= 0.05 or m["price"] >= 0.95:
                    continue
                all_markets.append({
                    "question": m["question"],
                    "price": m["price"],
                    "source": "manifold",
                    "ticker": m.get("id", ""),
                    "volume": float(m.get("volume", 0) or 0),
                    "liquidity_weight": 1.0,  # play money, lower trust
                })
        except Exception as e:
            print(f"  Manifold error: {e}")

    if "predictit" in sources:
        try:
            from src.markets.predictit_client import get_active_markets
            markets = get_active_markets()
            for m in markets:
                all_markets.append({
                    "question": m["question"],
                    "price": m["price"],
                    "source": "predictit",
                    "ticker": m.get("id", ""),
                    "volume": float(m.get("volume", 0) or 0),
                    "liquidity_weight": 2.5,  # real money, US regulated
                })
        except Exception as e:
            print(f"  PredictIt error: {e}")

    return all_markets


def find_cross_listed(
    markets: list[dict],
    similarity_threshold: float = 0.55,
) -> list[dict]:
    """Find questions that appear on multiple platforms.

    Returns clusters of matching markets grouped by question.
    """
    # Group by normalized question
    clusters = []
    used = set()

    for i, m1 in enumerate(markets):
        if i in used:
            continue

        cluster = [m1]
        used.add(i)

        for j, m2 in enumerate(markets):
            if j in used or j <= i:
                continue
            if m1["source"] == m2["source"]:
                continue  # skip same-platform duplicates

            sim = similarity(m1["question"], m2["question"])
            if sim >= similarity_threshold:
                cluster.append(m2)
                used.add(j)

        if len(cluster) >= 2:
            # Compute consensus and arbitrage metrics
            prices = [c["price"] for c in cluster]
            weights = [c["liquidity_weight"] for c in cluster]
            total_w = sum(weights)
            weighted_price = sum(p * w for p, w in zip(prices, weights)) / total_w if total_w > 0 else statistics.mean(prices)

            spread = max(prices) - min(prices)
            sources = [c["source"] for c in cluster]

            clusters.append({
                "question": m1["question"],
                "platforms": sources,
                "n_platforms": len(set(sources)),
                "prices": {c["source"]: c["price"] for c in cluster},
                "consensus_price": round(weighted_price, 4),
                "spread": round(spread, 4),
                "is_arbitrage": spread > 0.08,  # >8% spread = potential arb
                "markets": cluster,
            })

    # Sort by spread (arbitrage opportunities first)
    clusters.sort(key=lambda c: c["spread"], reverse=True)
    return clusters


def find_arbitrage(
    markets: list[dict] | None = None,
    sources: list[str] | None = None,
    min_spread: float = 0.05,
) -> list[dict]:
    """Find arbitrage opportunities across platforms.

    An arbitrage exists when the same question has significantly different
    prices on different platforms.
    """
    if markets is None:
        markets = fetch_all_markets(sources=sources)

    cross_listed = find_cross_listed(markets)

    arbitrages = []
    for cluster in cross_listed:
        if cluster["spread"] >= min_spread:
            prices = cluster["prices"]
            min_source = min(prices, key=prices.get)
            max_source = max(prices, key=prices.get)

            arbitrages.append({
                "question": cluster["question"],
                "spread": cluster["spread"],
                "buy_on": min_source,
                "buy_price": prices[min_source],
                "sell_on": max_source,
                "sell_price": prices[max_source],
                "consensus": cluster["consensus_price"],
                "n_platforms": cluster["n_platforms"],
                "all_prices": prices,
            })

    arbitrages.sort(key=lambda a: a["spread"], reverse=True)
    return arbitrages


def get_consensus_for_question(
    question: str,
    sources: list[str] | None = None,
    limit_per_source: int = 50,
) -> dict:
    """Get cross-platform consensus probability for a specific question.

    Searches all platforms for the closest matching market and computes
    a liquidity-weighted consensus.
    """
    all_markets = fetch_all_markets(sources=sources, limit_per_source=limit_per_source)

    # Find best match on each platform
    matches = []
    for m in all_markets:
        sim = similarity(question, m["question"])
        if sim > 0.40:
            matches.append({**m, "similarity": sim})

    if not matches:
        return {"question": question, "consensus": None, "message": "No matching markets found."}

    # Group by source, take best match per platform
    by_source = {}
    for m in matches:
        src = m["source"]
        if src not in by_source or m["similarity"] > by_source[src]["similarity"]:
            by_source[src] = m

    prices = []
    weights = []
    platform_data = {}
    for src, m in by_source.items():
        prices.append(m["price"])
        weights.append(m["liquidity_weight"])
        platform_data[src] = {
            "price": m["price"],
            "matched_question": m["question"],
            "similarity": round(m["similarity"], 3),
        }

    consensus = sum(p * w for p, w in zip(prices, weights)) / sum(weights) if weights else None

    return {
        "question": question,
        "consensus": round(consensus, 4) if consensus else None,
        "n_platforms": len(by_source),
        "platforms": platform_data,
        "spread": round(max(prices) - min(prices), 4) if len(prices) > 1 else 0,
    }
