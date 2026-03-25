"""
MiniSim Live Backtest — Fetch real settled Kalshi markets and evaluate swarm accuracy.

Pulls resolved binary markets from the Kalshi API, runs the swarm on each,
and computes Brier scores, calibration curves, and alpha analysis.

Usage: python live_backtest.py [--limit 200] [--agents 30] [--rounds 2]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import statistics
import time

from src.kalshi_client import get_events, parse_market
from src.offline_engine import swarm_score_offline


def fetch_backtestable_markets(max_events: int = 500) -> list[dict]:
    """Fetch settled binary markets from Kalshi that are suitable for backtesting.

    Filters:
    - Binary markets (yes/no result) only
    - Price between 0.05 and 0.95 (not already decided)
    - Title long enough to be a real prediction question
    - Excludes sports props, crypto 15-min, price range buckets
    """
    print("Fetching settled events from Kalshi API...")
    events = get_events(status="settled", with_nested_markets=True, limit=200, max_pages=max_events // 200 + 1)
    print(f"  Fetched {len(events)} events")

    markets = []
    skip_title_kw = [
        "price up in next 15", "price up in next 30",
        "win set", "win map", "points", "rebounds", "assists", "threes",
        "goals scored", "touchdowns", "strikeouts", "hits in",
        "price range", "be between", "be above", "be below",
        "total points", "spread", "over/under",
    ]

    # Skip sports-heavy series
    skip_series = [
        "KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXSOCCER", "KXVALORANT",
        "KXCOL", "KXNCAA", "KXMMA", "KXLEAGUE", "KXTENNIS", "KXCRYPTO15",
        "KXLOL", "KXAPL", "KXMVE",
    ]

    for event in events:
        event_markets = event.get("markets", [])
        for raw_m in event_markets:
            m = parse_market(raw_m)

            # Must have clear yes/no result
            if m["resolution"] is None:
                continue

            # Price must be in interesting range (not already decided)
            if m["price"] <= 0.05 or m["price"] >= 0.95:
                continue

            # Skip sports/crypto props
            ticker = m["ticker"]
            if any(ticker.startswith(p) for p in skip_series):
                continue

            # Skip by title keywords
            title_lower = m["title"].lower()
            if any(kw in title_lower for kw in skip_title_kw):
                continue

            # Must have meaningful title
            if len(m["title"]) < 25:
                continue

            # Build the question string from event title + market title
            event_title = event.get("title", "")
            q = m["title"]
            if not q.endswith("?"):
                q = q + "?"

            markets.append({
                "question": q,
                "event_title": event_title,
                "ticker": m["ticker"],
                "market_price": m["price"],
                "resolution": m["resolution"],
                "result": m["result"],
                "category": _categorize(q, event_title),
            })

    print(f"  Backtestable markets after filtering: {len(markets)}")
    return markets


def _categorize(question: str, event_title: str) -> str:
    """Categorize a market question."""
    text = (question + " " + event_title).lower()

    if any(kw in text for kw in ["fed", "rate", "inflation", "gdp", "recession", "unemployment", "treasury", "jobless", "economic"]):
        return "econ"
    if any(kw in text for kw in ["trump", "congress", "senate", "house", "election", "governor", "scotus", "supreme court", "veto", "cabinet", "secretary", "attorney general", "legislation", "government", "partisan", "republican", "democrat", "political"]):
        return "political"
    if any(kw in text for kw in ["ai", "gpt", "openai", "anthropic", "model", "deepseek", "llm", "machine learning", "autonomous", "robot"]):
        return "tech"
    if any(kw in text for kw in ["bitcoin", "ethereum", "crypto", "sol", "xrp", "btc", "eth", "defi", "blockchain"]):
        return "crypto"
    if any(kw in text for kw in ["ipo", "stock", "market cap", "nasdaq", "s&p", "dow", "revenue", "earnings", "valuation", "company", "tesla", "amazon", "apple", "nvidia"]):
        return "corporate"
    if any(kw in text for kw in ["war", "ukraine", "russia", "china", "taiwan", "nato", "sanction", "tariff", "trade", "military", "iran", "nuclear"]):
        return "geopolitics"
    if any(kw in text for kw in ["pope", "religion", "church", "vatican"]):
        return "world"
    if any(kw in text for kw in ["measles", "vaccine", "health", "fda", "drug", "pandemic", "who"]):
        return "health"
    if any(kw in text for kw in ["climate", "temperature", "emission", "oil", "gas price", "energy", "renewable"]):
        return "climate"
    if any(kw in text for kw in ["oscar", "grammy", "emmy", "album", "movie", "song", "tour", "concert", "celebrity"]):
        return "entertainment"

    return "other"


def run_live_backtest(
    n_agents: int = 30,
    n_rounds: int = 2,
    max_events: int = 500,
    max_markets: int = 300,
):
    markets = fetch_backtestable_markets(max_events)
    if not markets:
        print("No backtestable markets found!")
        return

    # Limit to max_markets
    markets = markets[:max_markets]
    n_markets = len(markets)

    print(f"\n{'=' * 70}")
    print(f"MiniSim Live Backtest — {n_markets} Real Kalshi Markets")
    print(f"Config: {n_agents} agents x {n_rounds} rounds")
    print(f"{'=' * 70}")

    results = []
    total_start = time.time()

    for i, market in enumerate(markets):
        sim = swarm_score_offline(
            question=market["question"],
            n_agents=n_agents,
            rounds=n_rounds,
            market_price=market["market_price"],
            peer_sample_size=5,
        )

        swarm_p = sim["swarm_probability_yes"]
        market_p = market["market_price"]
        actual = market["resolution"]

        swarm_brier = (swarm_p - actual) ** 2
        market_brier = (market_p - actual) ** 2

        result = {
            "question": market["question"],
            "event_title": market["event_title"],
            "ticker": market["ticker"],
            "category": market["category"],
            "market_price": market_p,
            "swarm_probability": round(swarm_p, 4),
            "resolution": actual,
            "swarm_brier": round(swarm_brier, 4),
            "market_brier": round(market_brier, 4),
            "swarm_beat_market": swarm_brier < market_brier,
            "alpha": round(swarm_p - market_p, 4),
        }
        results.append(result)

        indicator = "+" if result["swarm_beat_market"] else " "
        print(f"  [{i+1:3d}/{n_markets}] {indicator} S={swarm_p:.2f} M={market_p:.2f} A={int(actual)} "
              f"| {market['question'][:55]}")

    total_time = time.time() - total_start

    # --- Compute metrics ---
    swarm_briers = [r["swarm_brier"] for r in results]
    market_briers = [r["market_brier"] for r in results]
    overall_swarm = statistics.mean(swarm_briers)
    overall_market = statistics.mean(market_briers)
    wins = sum(1 for r in results if r["swarm_beat_market"])

    # Calibration curve
    buckets = {}
    for bstart in [i / 10 for i in range(10)]:
        bend = bstart + 0.1
        label = f"{bstart:.1f}-{bend:.1f}"
        items = [r for r in results if bstart <= r["swarm_probability"] < bend]
        if bend == 1.0:
            items += [r for r in results if r["swarm_probability"] == 1.0]
        if items:
            buckets[label] = {
                "count": len(items),
                "mean_predicted": round(statistics.mean([r["swarm_probability"] for r in items]), 4),
                "actual_rate": round(statistics.mean([r["resolution"] for r in items]), 4),
            }

    # By category
    categories = {}
    for cat in sorted(set(r["category"] for r in results)):
        cat_r = [r for r in results if r["category"] == cat]
        categories[cat] = {
            "n": len(cat_r),
            "swarm_brier": round(statistics.mean([r["swarm_brier"] for r in cat_r]), 4),
            "market_brier": round(statistics.mean([r["market_brier"] for r in cat_r]), 4),
            "wins": sum(1 for r in cat_r if r["swarm_beat_market"]),
        }

    # Alpha analysis
    alpha_improvement = sum(r["market_brier"] - r["swarm_brier"] for r in results if r["swarm_beat_market"])
    alpha_degradation = sum(r["swarm_brier"] - r["market_brier"] for r in results if not r["swarm_beat_market"])

    # Top alpha calls
    sorted_alpha = sorted(results, key=lambda r: r["market_brier"] - r["swarm_brier"], reverse=True)

    # --- Print ---
    print(f"\n{'=' * 70}")
    print(f"LIVE BACKTEST RESULTS — {n_markets} Real Kalshi Markets")
    print(f"{'=' * 70}")
    print(f"Overall Swarm Brier:  {overall_swarm:.4f}")
    print(f"Overall Market Brier: {overall_market:.4f}")
    print(f"Swarm beat market:    {wins}/{n_markets} ({100*wins/n_markets:.0f}%)")
    print(f"Total time:           {total_time:.1f}s")

    print(f"\n--- By Category ---")
    print(f"  {'Category':<14} {'N':>3}  {'Swarm':>7}  {'Market':>7}  {'Wins':>6}")
    for cat, d in sorted(categories.items()):
        print(f"  {cat:<14} {d['n']:>3}  {d['swarm_brier']:>7.4f}  {d['market_brier']:>7.4f}  {d['wins']:>3}/{d['n']}")

    print(f"\n--- Calibration Curve ---")
    print(f"  {'Bucket':<12} {'Count':>5}  {'Predicted':>10}  {'Actual':>10}")
    for label, d in sorted(buckets.items()):
        print(f"  {label:<12} {d['count']:>5}  {d['mean_predicted']:>10.3f}  {d['actual_rate']:>10.3f}")

    print(f"\n--- Alpha ---")
    print(f"  Improvement (wins): {alpha_improvement:.3f}")
    print(f"  Degradation (loss): {alpha_degradation:.3f}")
    print(f"  Net: {alpha_improvement - alpha_degradation:+.3f}")

    print(f"\n--- Top 5 Alpha Calls ---")
    for r in sorted_alpha[:5]:
        imp = r["market_brier"] - r["swarm_brier"]
        print(f"  Alpha={imp:+.3f} S={r['swarm_probability']:.2f} M={r['market_price']:.2f} "
              f"A={int(r['resolution'])} | {r['question'][:55]}")

    print(f"\n--- Top 5 Worst Calls ---")
    for r in sorted_alpha[-5:]:
        imp = r["market_brier"] - r["swarm_brier"]
        print(f"  Alpha={imp:+.3f} S={r['swarm_probability']:.2f} M={r['market_price']:.2f} "
              f"A={int(r['resolution'])} | {r['question'][:55]}")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        "n_markets": n_markets,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "overall_swarm_brier": round(overall_swarm, 4),
        "overall_market_brier": round(overall_market, 4),
        "win_rate": round(wins / n_markets, 4),
        "wins": wins,
        "net_alpha": round(alpha_improvement - alpha_degradation, 4),
        "total_time_seconds": round(total_time, 2),
        "calibration_curve": buckets,
        "by_category": categories,
        "top5_alpha": sorted_alpha[:5],
        "top5_worst": sorted_alpha[-5:],
        "all_results": results,
    }
    with open("results/live_backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/live_backtest_results.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=500)
    parser.add_argument("--max-markets", type=int, default=300)
    args = parser.parse_args()

    run_live_backtest(
        n_agents=args.agents,
        n_rounds=args.rounds,
        max_events=args.max_events,
        max_markets=args.max_markets,
    )
