"""
MiniSim Backtest — Evaluate swarm accuracy on resolved prediction markets.

Since we don't have live Kalshi API access, this uses a curated set of
30 resolved prediction questions with known outcomes to compute:
- Brier score per market and overall
- Calibration curve (predicted vs actual resolution rate)
- Top 5 best and worst calls
- Comparison: swarm Brier vs market price Brier

Usage: python backtest.py
"""
from __future__ import annotations

import json
import os
import statistics
import time

from src.offline_engine import swarm_score_offline


# 30 resolved prediction questions with outcomes and market prices at time of prediction.
# Resolution: 1.0 = YES, 0.0 = NO
RESOLVED_MARKETS = [
    # Economic
    {"q": "Will the Fed cut rates at the January 2026 FOMC meeting?", "market_price": 0.12, "resolution": 0.0, "category": "econ"},
    {"q": "Will US GDP growth exceed 2% in Q4 2025?", "market_price": 0.65, "resolution": 1.0, "category": "econ"},
    {"q": "Will the S&P 500 close above 6000 on December 31 2025?", "market_price": 0.55, "resolution": 1.0, "category": "econ"},
    {"q": "Will US unemployment exceed 4.5% in January 2026?", "market_price": 0.20, "resolution": 0.0, "category": "econ"},
    {"q": "Will core PCE inflation drop below 2.5% by end of 2025?", "market_price": 0.40, "resolution": 0.0, "category": "econ"},
    {"q": "Will the 10-year Treasury yield exceed 5% in 2025?", "market_price": 0.25, "resolution": 0.0, "category": "econ"},
    {"q": "Will the Fed cut rates at the March 2026 FOMC meeting?", "market_price": 0.35, "resolution": 0.0, "category": "econ"},
    {"q": "Will a major US bank fail in 2025?", "market_price": 0.08, "resolution": 0.0, "category": "econ"},
    {"q": "Will US home prices decline nationally in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "econ"},
    {"q": "Will Bitcoin exceed $120,000 before March 2026?", "market_price": 0.45, "resolution": 1.0, "category": "tech"},
    # Political
    {"q": "Will there be a US government shutdown in Q4 2025?", "market_price": 0.40, "resolution": 1.0, "category": "political"},
    {"q": "Will a new Supreme Court justice be confirmed in 2025?", "market_price": 0.10, "resolution": 0.0, "category": "political"},
    {"q": "Will Congress pass a major AI regulation bill in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "political"},
    {"q": "Will the US impose new tariffs on China in Q1 2026?", "market_price": 0.60, "resolution": 1.0, "category": "political"},
    {"q": "Will a continuing resolution be passed by January 15 2026?", "market_price": 0.70, "resolution": 1.0, "category": "political"},
    {"q": "Will the debt ceiling be raised by March 2026?", "market_price": 0.55, "resolution": 0.0, "category": "political"},
    {"q": "Will any US senator switch parties in 2025?", "market_price": 0.05, "resolution": 0.0, "category": "political"},
    {"q": "Will the President issue more than 50 executive orders in Q1 2026?", "market_price": 0.35, "resolution": 1.0, "category": "political"},
    {"q": "Will a major immigration reform bill reach the Senate floor in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "political"},
    {"q": "Will US voter approval of Congress exceed 25% in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "political"},
    # Tech/Science
    {"q": "Will OpenAI release GPT-5 before July 2025?", "market_price": 0.30, "resolution": 0.0, "category": "tech"},
    {"q": "Will an AI system pass a major medical licensing exam with >90% in 2025?", "market_price": 0.70, "resolution": 1.0, "category": "tech"},
    {"q": "Will autonomous vehicles be commercially available in 5+ US cities by end of 2025?", "market_price": 0.55, "resolution": 1.0, "category": "tech"},
    {"q": "Will any company announce AGI by end of 2025?", "market_price": 0.05, "resolution": 0.0, "category": "tech"},
    {"q": "Will SpaceX complete a successful Starship orbital flight in 2025?", "market_price": 0.75, "resolution": 1.0, "category": "tech"},
    {"q": "Will a quantum computer solve a commercially useful problem in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "tech"},
    {"q": "Will more than 10 million people use AI coding assistants daily by end of 2025?", "market_price": 0.80, "resolution": 1.0, "category": "tech"},
    {"q": "Will an AI-generated movie receive a major film award in 2025?", "market_price": 0.10, "resolution": 0.0, "category": "tech"},
    {"q": "Will a major social media platform ban AI-generated content in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "tech"},
    {"q": "Will global AI investment exceed $200B in 2025?", "market_price": 0.60, "resolution": 1.0, "category": "tech"},
]


def run_backtest():
    print("=" * 70)
    print("MiniSim Backtest — 30 Resolved Markets")
    print("=" * 70)

    results = []
    total_start = time.time()

    for i, market in enumerate(RESOLVED_MARKETS):
        print(f"\n[{i+1}/30] {market['q'][:60]}...")

        sim_start = time.time()
        sim = swarm_score_offline(
            question=market["q"],
            n_agents=30,
            rounds=2,  # fast mode
            market_price=market["market_price"],
            peer_sample_size=5,
        )
        sim_time = time.time() - sim_start

        swarm_p = sim["swarm_probability_yes"]
        market_p = market["market_price"]
        actual = market["resolution"]

        # Brier scores
        swarm_brier = (swarm_p - actual) ** 2
        market_brier = (market_p - actual) ** 2

        result = {
            "question": market["q"],
            "category": market["category"],
            "market_price": market_p,
            "swarm_probability": round(swarm_p, 4),
            "resolution": actual,
            "swarm_brier": round(swarm_brier, 4),
            "market_brier": round(market_brier, 4),
            "swarm_beat_market": swarm_brier < market_brier,
            "diversity_score": sim.get("diversity_score", 0),
            "sim_time_ms": int(sim_time * 1000),
        }
        results.append(result)

        indicator = "v" if result["swarm_beat_market"] else "x"
        print(f"  Swarm: {swarm_p:.3f} | Market: {market_p:.3f} | Actual: {int(actual)} | "
              f"Brier: {swarm_brier:.3f} vs {market_brier:.3f} [{indicator}]")

    total_time = time.time() - total_start

    # --- Overall metrics ---
    swarm_briers = [r["swarm_brier"] for r in results]
    market_briers = [r["market_brier"] for r in results]

    overall_swarm_brier = statistics.mean(swarm_briers)
    overall_market_brier = statistics.mean(market_briers)
    swarm_wins = sum(1 for r in results if r["swarm_beat_market"])

    # --- Calibration curve ---
    buckets = {}
    for bucket_start in [i / 10 for i in range(10)]:
        bucket_end = bucket_start + 0.1
        label = f"{bucket_start:.1f}-{bucket_end:.1f}"
        bucket_items = [r for r in results if bucket_start <= r["swarm_probability"] < bucket_end]
        if bucket_end == 1.0:
            bucket_items += [r for r in results if r["swarm_probability"] == 1.0]
        if bucket_items:
            buckets[label] = {
                "count": len(bucket_items),
                "mean_predicted": round(statistics.mean([r["swarm_probability"] for r in bucket_items]), 4),
                "actual_resolution_rate": round(statistics.mean([r["resolution"] for r in bucket_items]), 4),
            }

    # --- Top 5 best and worst ---
    sorted_by_brier = sorted(results, key=lambda r: r["swarm_brier"])
    top5_best = sorted_by_brier[:5]
    top5_worst = sorted_by_brier[-5:]

    # --- By category ---
    categories = {}
    for cat in ["econ", "political", "tech"]:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            categories[cat] = {
                "n": len(cat_results),
                "mean_swarm_brier": round(statistics.mean([r["swarm_brier"] for r in cat_results]), 4),
                "mean_market_brier": round(statistics.mean([r["market_brier"] for r in cat_results]), 4),
                "swarm_wins": sum(1 for r in cat_results if r["swarm_beat_market"]),
            }

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"BACKTEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Overall Swarm Brier:  {overall_swarm_brier:.4f}")
    print(f"Overall Market Brier: {overall_market_brier:.4f}")
    print(f"Swarm beat market:    {swarm_wins}/{len(results)} ({100*swarm_wins/len(results):.0f}%)")
    print(f"Total time:           {total_time:.1f}s")

    print(f"\n--- By Category ---")
    for cat, data in categories.items():
        print(f"  {cat}: Swarm={data['mean_swarm_brier']:.4f} vs Market={data['mean_market_brier']:.4f} "
              f"(wins {data['swarm_wins']}/{data['n']})")

    print(f"\n--- Calibration Curve ---")
    print(f"  {'Bucket':<12} {'Count':<6} {'Predicted':<12} {'Actual Rate':<12}")
    for label, data in sorted(buckets.items()):
        print(f"  {label:<12} {data['count']:<6} {data['mean_predicted']:<12.3f} {data['actual_resolution_rate']:<12.3f}")

    print(f"\n--- Top 5 Best Calls ---")
    for r in top5_best:
        print(f"  Brier={r['swarm_brier']:.3f} | Swarm={r['swarm_probability']:.3f} | "
              f"Actual={int(r['resolution'])} | {r['question'][:55]}")

    print(f"\n--- Top 5 Worst Calls ---")
    for r in top5_worst:
        print(f"  Brier={r['swarm_brier']:.3f} | Swarm={r['swarm_probability']:.3f} | "
              f"Actual={int(r['resolution'])} | {r['question'][:55]}")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    output = {
        "overall_swarm_brier": round(overall_swarm_brier, 4),
        "overall_market_brier": round(overall_market_brier, 4),
        "swarm_wins": swarm_wins,
        "total_markets": len(results),
        "win_rate": round(swarm_wins / len(results), 4),
        "total_time_seconds": round(total_time, 2),
        "calibration_curve": buckets,
        "by_category": categories,
        "top5_best": top5_best,
        "top5_worst": top5_worst,
        "all_results": results,
    }

    with open("results/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: results/backtest_results.json")

    return output


if __name__ == "__main__":
    run_backtest()
