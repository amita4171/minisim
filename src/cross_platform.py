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
            from src.kalshi_client import get_active_markets
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
            from src.polymarket_client import get_active_markets
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
            from src.manifold_client import get_active_binary_markets
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
            from src.predictit_client import get_active_markets
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


# Platform fee structures (as of March 2026)
PLATFORM_FEES = {
    "kalshi": {
        "maker_fee": 0.00,      # 0% maker
        "taker_fee": 0.01,      # 1% taker (1 cent per contract on $1)
        "withdrawal_fee": 0.00,
        "description": "CFTC-regulated, $1 binary contracts, 1% taker fee",
    },
    "polymarket": {
        "maker_fee": 0.00,      # 0% maker (rebates available)
        "taker_fee": 0.02,      # ~2% taker (varies by market)
        "withdrawal_fee": 0.001, # gas fees for Polygon withdrawal
        "description": "Crypto (Polygon/USDC), 0% maker / ~2% taker",
    },
    "predictit": {
        "maker_fee": 0.00,
        "taker_fee": 0.05,      # 5% on profits (not on trade)
        "profit_fee": 0.10,     # 10% fee on profit withdrawals
        "withdrawal_fee": 0.05, # 5% withdrawal fee
        "description": "CFTC-authorized, 5% profit fee + 10% withdrawal",
    },
    "manifold": {
        "maker_fee": 0.00,
        "taker_fee": 0.00,      # play money, no real fees
        "withdrawal_fee": 0.00,
        "description": "Play money (Mana), no fees, not real-money arbitrage",
    },
}


def compute_arbitrage_profit(
    buy_price: float,
    sell_price: float,
    buy_platform: str,
    sell_platform: str,
    position_size: float = 100.0,
) -> dict:
    """Compute net profit from a cross-platform arbitrage after fees.

    Strategy: Buy YES on the cheap platform, Buy NO (sell YES) on the expensive platform.
    If the question resolves YES, you win on the cheap platform and lose on the expensive.
    If the question resolves NO, the opposite.
    The arbitrage profit comes from the spread exceeding total fees.

    Args:
        buy_price: YES price on cheap platform (e.g., 0.40)
        sell_price: YES price on expensive platform (e.g., 0.55)
        buy_platform: platform to buy YES on
        sell_platform: platform to buy NO on (sell YES equivalent)
        position_size: dollars to deploy per side

    Returns:
        dict with gross_profit, total_fees, net_profit, roi, is_profitable
    """
    buy_fees = PLATFORM_FEES.get(buy_platform, {"taker_fee": 0.02})
    sell_fees = PLATFORM_FEES.get(sell_platform, {"taker_fee": 0.02})

    spread = sell_price - buy_price

    # Cost to buy YES at buy_price on cheap platform
    buy_cost = buy_price + buy_fees.get("taker_fee", 0)

    # Cost to buy NO at (1 - sell_price) on expensive platform
    no_price = 1 - sell_price
    sell_cost = no_price + sell_fees.get("taker_fee", 0)

    # Total cost per "complete set" (buy YES on A + buy NO on B)
    total_cost_per_unit = buy_cost + sell_cost

    # A complete set always pays out $1 regardless of resolution
    # (either YES wins on A or NO wins on B)
    payout_per_unit = 1.0

    # Gross profit per unit
    gross_profit_per_unit = payout_per_unit - total_cost_per_unit

    # Number of units we can buy with position_size per side
    n_units = position_size  # assuming $1 contracts

    gross_profit = gross_profit_per_unit * n_units
    total_fees = (buy_fees.get("taker_fee", 0) + sell_fees.get("taker_fee", 0)) * n_units

    # PredictIt has additional profit fee on winnings
    profit_fee = 0
    for platform, fees in [(buy_platform, buy_fees), (sell_platform, sell_fees)]:
        if "profit_fee" in fees and gross_profit > 0:
            profit_fee += fees["profit_fee"] * gross_profit * 0.5  # apply to winning side

    net_profit = gross_profit - profit_fee
    total_deployed = position_size * 2  # deployed on both sides
    roi = net_profit / total_deployed if total_deployed > 0 else 0

    return {
        "buy_platform": buy_platform,
        "sell_platform": sell_platform,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "spread": round(spread, 4),
        "buy_cost_per_unit": round(buy_cost, 4),
        "sell_cost_per_unit": round(sell_cost, 4),
        "total_cost_per_unit": round(total_cost_per_unit, 4),
        "gross_profit_per_unit": round(gross_profit_per_unit, 4),
        "gross_profit": round(gross_profit, 2),
        "trading_fees": round(total_fees, 2),
        "profit_fees": round(profit_fee, 2),
        "net_profit": round(net_profit, 2),
        "total_deployed": round(total_deployed, 2),
        "roi_pct": round(roi * 100, 2),
        "is_profitable": net_profit > 0,
        "break_even_spread": round(buy_fees.get("taker_fee", 0) + sell_fees.get("taker_fee", 0), 4),
    }


def find_profitable_arbitrage(
    markets: list[dict] | None = None,
    sources: list[str] | None = None,
    position_size: float = 100.0,
) -> list[dict]:
    """Find arbitrage opportunities that are profitable AFTER fees."""
    if markets is None:
        markets = fetch_all_markets(sources=sources)

    cross_listed = find_cross_listed(markets)

    profitable = []
    for cluster in cross_listed:
        prices = cluster["prices"]
        if len(prices) < 2:
            continue

        # Try all platform pairs
        platforms = list(prices.keys())
        for i, p1 in enumerate(platforms):
            for p2 in platforms[i+1:]:
                # Try both directions
                for buy_p, sell_p in [(p1, p2), (p2, p1)]:
                    if prices[buy_p] >= prices[sell_p]:
                        continue  # no spread in this direction

                    result = compute_arbitrage_profit(
                        buy_price=prices[buy_p],
                        sell_price=prices[sell_p],
                        buy_platform=buy_p,
                        sell_platform=sell_p,
                        position_size=position_size,
                    )

                    if result["is_profitable"]:
                        result["question"] = cluster["question"]
                        profitable.append(result)

    profitable.sort(key=lambda a: a["net_profit"], reverse=True)
    return profitable


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
