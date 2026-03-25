"""
Build 500+ Question Eval Dataset — Pulls resolved questions from all available sources.

Sources:
- Manifold Markets: 500+ resolved binary questions (play money, broadest coverage)
- Kalshi: 200+ settled markets with meaningful prices
- Curated: 100 hand-picked questions from backtest.py

Stores in SQLite for fast querying. Outputs JSON for eval runner.

Usage: python build_eval_dataset.py [--target 500]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import statistics
import time

from src.database import Database


def pull_manifold_resolved(target: int = 400) -> list[dict]:
    """Pull resolved binary markets from Manifold."""
    from src.manifold_client import search_markets, parse_market

    print(f"  Pulling Manifold resolved markets (target: {target})...")
    all_markets = []

    # Pull in batches with different sort orders to get diversity
    for sort in ["most-popular", "newest"]:
        for offset in range(0, target, 100):
            try:
                raw = search_markets(
                    term="",
                    sort=sort,
                    filter="resolved",
                    contract_type="BINARY",
                    limit=100,
                    offset=offset,
                )
                parsed = [parse_market(m) for m in raw]
                # Filter: must have clear resolution and non-trivial price
                good = [
                    m for m in parsed
                    if m["resolution"] is not None
                    and 0.03 < m["price"] < 0.97
                    and len(m["question"]) > 25
                    and m["volume"] > 50
                ]
                all_markets.extend(good)
                time.sleep(0.3)
            except Exception as e:
                print(f"    Manifold error at offset {offset}: {e}")
                break

    # Deduplicate by question
    seen = set()
    unique = []
    for m in all_markets:
        key = m["question"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append({
                "question": m["question"],
                "market_price": m["price"],
                "resolution": m["resolution"],
                "source": "manifold",
                "category": _categorize(m["question"]),
                "volume": m.get("volume", 0),
            })

    print(f"  Manifold: {len(unique)} unique resolved questions")
    return unique[:target]


def pull_kalshi_resolved(target: int = 300) -> list[dict]:
    """Pull settled markets from Kalshi."""
    from src.kalshi_client import get_settled_markets

    print(f"  Pulling Kalshi settled markets (target: {target})...")
    min_ts = int(time.time()) - (180 * 86400)  # last 180 days

    try:
        raw = get_settled_markets(limit=1000, max_pages=5, min_settled_ts=min_ts)
    except Exception as e:
        print(f"    Kalshi error: {e}")
        return []

    # Filter
    skip_prefixes = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXSOCCER", "KXVALORANT",
                     "KXMVE", "KXMMA", "KXLEAGUE", "KXTENNIS", "KXCRYPTO15", "KXLOL"]
    skip_kw = ["price up in next", "price range", "be between", "be above", "be below",
               "win set", "win map", "points", "rebounds", "assists", "threes"]

    good = []
    for m in raw:
        if any(m["ticker"].startswith(p) for p in skip_prefixes):
            continue
        if m["result"] not in ("yes", "no"):
            continue
        if m["price"] <= 0.03 or m["price"] >= 0.97:
            continue
        title_lower = m.get("title", "").lower()
        if any(kw in title_lower for kw in skip_kw):
            continue
        if len(m.get("title", "")) < 25:
            continue

        resolution = 1.0 if m["result"] == "yes" else 0.0
        good.append({
            "question": m["title"] if m["title"].endswith("?") else m["title"] + "?",
            "market_price": m["price"],
            "resolution": resolution,
            "source": "kalshi",
            "category": _categorize(m.get("title", "")),
            "volume": 0,
        })

    # Deduplicate
    seen = set()
    unique = []
    for m in good:
        key = m["question"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(m)

    print(f"  Kalshi: {len(unique)} unique resolved questions")
    return unique[:target]


def get_curated() -> list[dict]:
    """Get our hand-curated 100 questions from backtest.py."""
    try:
        from scripts.backtest import RESOLVED_MARKETS
        curated = [{
            "question": m["q"],
            "market_price": m["market_price"],
            "resolution": m["resolution"],
            "source": "curated",
            "category": m.get("category", "other"),
            "volume": 0,
        } for m in RESOLVED_MARKETS]
        print(f"  Curated: {len(curated)} questions")
        return curated
    except Exception:
        return []


def _categorize(question: str) -> str:
    """Quick categorization."""
    q = question.lower()
    if any(kw in q for kw in ["fed", "rate", "inflation", "gdp", "recession", "stock", "s&p", "treasury"]):
        return "econ"
    if any(kw in q for kw in ["trump", "congress", "election", "senate", "vote", "supreme court", "government"]):
        return "political"
    if any(kw in q for kw in ["ai", "gpt", "openai", "model", "autonomous", "robot", "tech"]):
        return "tech"
    if any(kw in q for kw in ["war", "ukraine", "russia", "china", "nato", "military", "sanction"]):
        return "geopolitics"
    if any(kw in q for kw in ["bitcoin", "crypto", "ethereum", "btc"]):
        return "crypto"
    return "other"


def build_dataset(target: int = 500):
    print("=" * 70)
    print(f"Building Eval Dataset (target: {target}+ questions)")
    print("=" * 70)

    all_questions = []

    # Pull from each source
    curated = get_curated()
    all_questions.extend(curated)

    manifold = pull_manifold_resolved(target=max(300, target - len(all_questions)))
    all_questions.extend(manifold)

    if len(all_questions) < target:
        kalshi = pull_kalshi_resolved(target=target - len(all_questions))
        all_questions.extend(kalshi)

    # Global dedup
    seen = set()
    unique = []
    for q in all_questions:
        key = q["question"].lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    # Stats
    by_source = {}
    by_category = {}
    for q in unique:
        by_source[q["source"]] = by_source.get(q["source"], 0) + 1
        by_category[q["category"]] = by_category.get(q["category"], 0) + 1

    yes_rate = statistics.mean([q["resolution"] for q in unique])

    print(f"\n{'=' * 70}")
    print(f"Dataset Built: {len(unique)} questions")
    print(f"{'=' * 70}")
    print(f"  By source: {by_source}")
    print(f"  By category: {by_category}")
    print(f"  YES rate: {yes_rate:.1%}")
    print(f"  Price range: {min(q['market_price'] for q in unique):.2f} - {max(q['market_price'] for q in unique):.2f}")

    # Save
    os.makedirs("results", exist_ok=True)
    dataset = {
        "n_questions": len(unique),
        "by_source": by_source,
        "by_category": by_category,
        "yes_rate": round(yes_rate, 4),
        "questions": unique,
    }
    with open("results/eval_dataset_500.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved to results/eval_dataset_500.json")

    # Also store in database
    try:
        db = Database()
        for q in unique:
            db.snapshot_market(
                source=q["source"],
                ticker="",
                question=q["question"],
                price=q["market_price"],
                category=q["category"],
                status="resolved",
            )
        db.close()
        print("Stored in SQLite database")
    except Exception as e:
        print(f"DB error: {e}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()
    build_dataset(target=args.target)
