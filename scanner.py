"""
MiniSim Opportunity Scanner — Finds markets where swarm disagrees with market price.

Scans both Kalshi and Polymarket for active markets, runs the swarm on each,
and flags opportunities where the swarm's estimate differs from market price
by more than a configurable threshold.

Usage:
  python scanner.py                    # one-time scan
  python scanner.py --watch --interval 300  # continuous scan every 5 min
  python scanner.py --source polymarket     # scan only Polymarket
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from src.offline_engine import swarm_score_offline


def scan_kalshi(limit: int = 50) -> list[dict]:
    """Fetch active Kalshi markets suitable for scanning."""
    from src.kalshi_client import get_active_markets
    try:
        markets = get_active_markets(limit=limit)
        # Filter to interesting ones (not sports props, meaningful title)
        interesting = []
        skip = ["KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXSOCCER", "KXMVE",
                "KXVALORANT", "KXMMA", "KXLEAGUE", "KXTENNIS", "KXCRYPTO15", "KXLOL"]
        for m in markets:
            if any(m["ticker"].startswith(p) for p in skip):
                continue
            if m["price"] <= 0.05 or m["price"] >= 0.95:
                continue
            if len(m.get("title", "")) < 25:
                continue
            interesting.append({
                "question": m["title"] if m["title"].endswith("?") else m["title"] + "?",
                "market_price": m["price"],
                "ticker": m["ticker"],
                "source": "kalshi",
                "volume": m.get("volume_24h", "0"),
            })
        return interesting
    except Exception as e:
        print(f"  Kalshi fetch error: {e}")
        return []


def scan_polymarket(limit: int = 50) -> list[dict]:
    """Fetch active Polymarket markets suitable for scanning."""
    from src.polymarket_client import get_active_markets
    try:
        markets = get_active_markets(limit=limit, min_volume=5000)
        interesting = []
        for m in markets:
            if m["price"] <= 0.05 or m["price"] >= 0.95:
                continue
            if len(m.get("question", "")) < 25:
                continue
            interesting.append({
                "question": m["question"],
                "market_price": m["price"],
                "ticker": m.get("slug", m["id"]),
                "source": "polymarket",
                "volume": m.get("volume", 0),
            })
        return interesting
    except Exception as e:
        print(f"  Polymarket fetch error: {e}")
        return []


def run_scan(
    sources: list[str] = None,
    n_agents: int = 20,
    n_rounds: int = 2,
    edge_threshold: float = 0.05,
    max_markets: int = 30,
) -> list[dict]:
    """Run a single scan across configured sources."""
    if sources is None:
        sources = ["kalshi", "polymarket"]

    all_markets = []
    for source in sources:
        print(f"  Fetching {source}...")
        if source == "kalshi":
            all_markets.extend(scan_kalshi(limit=100))
        elif source == "polymarket":
            all_markets.extend(scan_polymarket(limit=100))

    # Deduplicate by question similarity (simple)
    seen_questions = set()
    unique = []
    for m in all_markets:
        q_key = m["question"].lower()[:50]
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            unique.append(m)
    all_markets = unique[:max_markets]

    print(f"  Scanning {len(all_markets)} markets...")

    opportunities = []
    for i, market in enumerate(all_markets):
        sim = swarm_score_offline(
            question=market["question"],
            n_agents=n_agents,
            rounds=n_rounds,
            market_price=market["market_price"],
        )

        swarm_p = sim["swarm_probability_yes"]
        edge = swarm_p - market["market_price"]

        result = {
            "question": market["question"],
            "source": market["source"],
            "ticker": market["ticker"],
            "market_price": market["market_price"],
            "swarm_probability": round(swarm_p, 4),
            "edge": round(edge, 4),
            "abs_edge": round(abs(edge), 4),
            "direction": "BUY YES" if edge > 0 else "BUY NO",
            "confidence_interval": sim.get("confidence_interval", []),
            "diversity_score": sim.get("diversity_score", 0),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Log every prediction to track record
        try:
            from src.track_record import TrackRecord
            tr = TrackRecord()
            tr.log_prediction(
                question=market["question"],
                swarm_probability=swarm_p,
                market_price=market["market_price"],
                source=market["source"],
                ticker=market["ticker"],
                confidence_interval=sim.get("confidence_interval"),
                n_agents=n_agents,
                n_rounds=n_rounds,
                edge=edge,
            )
        except Exception:
            pass

        if abs(edge) >= edge_threshold:
            opportunities.append(result)

    # Sort by absolute edge (strongest signals first)
    opportunities.sort(key=lambda x: x["abs_edge"], reverse=True)
    return opportunities


def print_opportunities(opps: list[dict]):
    """Print opportunities in a readable format."""
    if not opps:
        print("\n  No opportunities found above threshold.")
        return

    print(f"\n{'=' * 80}")
    print(f"  OPPORTUNITIES FOUND: {len(opps)}")
    print(f"{'=' * 80}")

    for i, opp in enumerate(opps):
        edge_pct = opp["edge"] * 100
        print(f"\n  [{i+1}] {opp['direction']} | Edge: {edge_pct:+.1f}%")
        print(f"      {opp['question'][:70]}")
        print(f"      Market: {opp['market_price']:.2f} | Swarm: {opp['swarm_probability']:.2f} | Source: {opp['source']}")
        if opp.get("confidence_interval"):
            ci = opp["confidence_interval"]
            print(f"      95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")


def save_scan(opps: list[dict], all_scanned: int):
    """Save scan results to track record."""
    os.makedirs("results", exist_ok=True)

    scan_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "markets_scanned": all_scanned,
        "opportunities_found": len(opps),
        "opportunities": opps,
    }

    # Append to scan history
    history_path = "results/scan_history.json"
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(scan_record)

    # Keep last 1000 scans
    history = history[-1000:]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Also save latest scan
    with open("results/latest_scan.json", "w") as f:
        json.dump(scan_record, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MiniSim Opportunity Scanner")
    parser.add_argument("--source", choices=["kalshi", "polymarket", "both"], default="both")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--edge", type=float, default=0.05, help="Minimum edge threshold (default: 0.05 = 5%)")
    parser.add_argument("--max-markets", type=int, default=30)
    parser.add_argument("--watch", action="store_true", help="Continuous scanning mode")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between scans in watch mode")
    args = parser.parse_args()

    sources = ["kalshi", "polymarket"] if args.source == "both" else [args.source]

    scan_num = 0
    while True:
        scan_num += 1
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'#' * 80}")
        print(f"  SCAN #{scan_num} — {now}")
        print(f"  Sources: {', '.join(sources)} | Agents: {args.agents} | Edge threshold: {args.edge*100:.0f}%")
        print(f"{'#' * 80}")

        opps = run_scan(
            sources=sources,
            n_agents=args.agents,
            n_rounds=args.rounds,
            edge_threshold=args.edge,
            max_markets=args.max_markets,
        )

        print_opportunities(opps)
        save_scan(opps, args.max_markets)
        print(f"\n  Scan saved to results/latest_scan.json")

        if not args.watch:
            break

        print(f"\n  Next scan in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
