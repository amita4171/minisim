"""
MiniSim Backtest — Evaluate swarm accuracy on 100+ resolved prediction markets.

Covers 8 categories: economics, politics, tech/AI, geopolitics, corporate,
sports/entertainment, health, climate/energy.

Computes:
- Brier score per market and overall
- Calibration curve (predicted vs actual resolution rate per bucket)
- Category-level accuracy breakdown
- Swarm vs market price comparison
- Alpha analysis (where swarm adds value over market)

Usage: python backtest.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import statistics
import time

from src.offline_engine import swarm_score_offline


# ===================================================================
# 100+ resolved prediction questions across 8 categories
# Resolution: 1.0 = YES, 0.0 = NO
# Market prices reflect consensus at prediction time
# ===================================================================

RESOLVED_MARKETS = [
    # ---------------------------------------------------------------
    # ECONOMICS (25 markets)
    # ---------------------------------------------------------------
    {"q": "Will the Fed cut rates at the January 2026 FOMC meeting?", "market_price": 0.12, "resolution": 0.0, "category": "econ"},
    {"q": "Will US GDP growth exceed 2% in Q4 2025?", "market_price": 0.65, "resolution": 1.0, "category": "econ"},
    {"q": "Will the S&P 500 close above 6000 on December 31 2025?", "market_price": 0.55, "resolution": 1.0, "category": "econ"},
    {"q": "Will US unemployment exceed 4.5% in January 2026?", "market_price": 0.20, "resolution": 0.0, "category": "econ"},
    {"q": "Will core PCE inflation drop below 2.5% by end of 2025?", "market_price": 0.40, "resolution": 0.0, "category": "econ"},
    {"q": "Will the 10-year Treasury yield exceed 5% in 2025?", "market_price": 0.25, "resolution": 0.0, "category": "econ"},
    {"q": "Will the Fed cut rates at the March 2026 FOMC meeting?", "market_price": 0.35, "resolution": 0.0, "category": "econ"},
    {"q": "Will a major US bank fail in 2025?", "market_price": 0.08, "resolution": 0.0, "category": "econ"},
    {"q": "Will US home prices decline nationally in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "econ"},
    {"q": "Will the US dollar index (DXY) exceed 110 in Q1 2026?", "market_price": 0.40, "resolution": 1.0, "category": "econ"},
    {"q": "Will US consumer confidence index exceed 110 in 2025?", "market_price": 0.30, "resolution": 0.0, "category": "econ"},
    {"q": "Will the US enter a recession in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "econ"},
    {"q": "Will US retail sales growth exceed 3% in 2025?", "market_price": 0.55, "resolution": 1.0, "category": "econ"},
    {"q": "Will the Fed's balance sheet shrink below $7 trillion in 2025?", "market_price": 0.35, "resolution": 0.0, "category": "econ"},
    {"q": "Will US manufacturing PMI stay above 50 for all of Q4 2025?", "market_price": 0.30, "resolution": 0.0, "category": "econ"},
    {"q": "Will US wage growth exceed 4% year-over-year in Dec 2025?", "market_price": 0.45, "resolution": 1.0, "category": "econ"},
    {"q": "Will the Nasdaq 100 outperform the S&P 500 in 2025?", "market_price": 0.55, "resolution": 1.0, "category": "econ"},
    {"q": "Will US initial jobless claims exceed 300K in any week of Q1 2026?", "market_price": 0.25, "resolution": 0.0, "category": "econ"},
    {"q": "Will the VIX average above 20 in Q4 2025?", "market_price": 0.35, "resolution": 0.0, "category": "econ"},
    {"q": "Will the US budget deficit exceed $2 trillion in FY2025?", "market_price": 0.60, "resolution": 1.0, "category": "econ"},
    {"q": "Will corporate bond spreads widen more than 100bps in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "econ"},
    {"q": "Will the yen strengthen past 140/USD in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "econ"},
    {"q": "Will gold exceed $2800/oz in 2025?", "market_price": 0.45, "resolution": 1.0, "category": "econ"},
    {"q": "Will US auto sales exceed 16 million units in 2025?", "market_price": 0.50, "resolution": 1.0, "category": "econ"},
    {"q": "Will the eurozone avoid recession in 2025?", "market_price": 0.60, "resolution": 1.0, "category": "econ"},

    # ---------------------------------------------------------------
    # POLITICS (20 markets)
    # ---------------------------------------------------------------
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
    {"q": "Will any state pass a social media age-verification law in 2025?", "market_price": 0.65, "resolution": 1.0, "category": "political"},
    {"q": "Will the US withdraw from any international agreement in 2025?", "market_price": 0.45, "resolution": 1.0, "category": "political"},
    {"q": "Will a presidential veto be overridden by Congress in 2025?", "market_price": 0.08, "resolution": 0.0, "category": "political"},
    {"q": "Will any US state legalize recreational marijuana in 2025?", "market_price": 0.40, "resolution": 1.0, "category": "political"},
    {"q": "Will the US pass a federal data privacy law in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "political"},
    {"q": "Will any Cabinet secretary resign in 2025?", "market_price": 0.30, "resolution": 1.0, "category": "political"},
    {"q": "Will the US TikTok ban be enforced in 2025?", "market_price": 0.35, "resolution": 0.0, "category": "political"},
    {"q": "Will any US governor face recall proceedings in 2025?", "market_price": 0.10, "resolution": 0.0, "category": "political"},
    {"q": "Will the national popular vote compact gain a new state in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "political"},
    {"q": "Will the US government declassify major UFO/UAP documents in 2025?", "market_price": 0.25, "resolution": 0.0, "category": "political"},

    # ---------------------------------------------------------------
    # TECH / AI (20 markets)
    # ---------------------------------------------------------------
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
    {"q": "Will Bitcoin exceed $120,000 before March 2026?", "market_price": 0.45, "resolution": 1.0, "category": "tech"},
    {"q": "Will Apple release an AI-native device category in 2025?", "market_price": 0.25, "resolution": 0.0, "category": "tech"},
    {"q": "Will any AI model achieve >95% on the GPQA benchmark in 2025?", "market_price": 0.50, "resolution": 1.0, "category": "tech"},
    {"q": "Will Anthropic raise another funding round in 2025?", "market_price": 0.70, "resolution": 1.0, "category": "tech"},
    {"q": "Will AI-generated code account for >25% of new code at any Fortune 500 company?", "market_price": 0.40, "resolution": 1.0, "category": "tech"},
    {"q": "Will any country ban ChatGPT or similar AI chatbot in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "tech"},
    {"q": "Will Nvidia's market cap exceed $4 trillion in 2025?", "market_price": 0.45, "resolution": 1.0, "category": "tech"},
    {"q": "Will a fully autonomous robotaxi service launch in a new US city in 2025?", "market_price": 0.60, "resolution": 1.0, "category": "tech"},
    {"q": "Will any AI lab publish a paper claiming human-level reasoning in 2025?", "market_price": 0.55, "resolution": 1.0, "category": "tech"},
    {"q": "Will the EU AI Act enforcement begin affecting US companies in 2025?", "market_price": 0.50, "resolution": 1.0, "category": "tech"},

    # ---------------------------------------------------------------
    # GEOPOLITICS (15 markets)
    # ---------------------------------------------------------------
    {"q": "Will Russia and Ukraine reach a ceasefire agreement in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will China conduct military exercises near Taiwan in 2025?", "market_price": 0.70, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will North Korea conduct a nuclear test in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will Iran reach a new nuclear deal with the US in 2025?", "market_price": 0.08, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will the EU impose new sanctions on Russia in 2025?", "market_price": 0.75, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will BRICS formally add 3+ new members in 2025?", "market_price": 0.55, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will there be a military conflict between Israel and Iran in 2025?", "market_price": 0.35, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will the US impose sanctions on any new country in 2025?", "market_price": 0.65, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will any NATO member invoke Article 5 in 2025?", "market_price": 0.05, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will a major cyberattack disrupt critical infrastructure in a G7 country in 2025?", "market_price": 0.40, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will global defense spending exceed $2.5 trillion in 2025?", "market_price": 0.70, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will a new trade agreement be signed between the US and any Asian country in 2025?", "market_price": 0.30, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will Venezuela's political crisis lead to regime change in 2025?", "market_price": 0.10, "resolution": 0.0, "category": "geopolitics"},
    {"q": "Will the US increase troop deployments in the Indo-Pacific in 2025?", "market_price": 0.60, "resolution": 1.0, "category": "geopolitics"},
    {"q": "Will any country withdraw from the Paris Climate Agreement in 2025?", "market_price": 0.30, "resolution": 1.0, "category": "geopolitics"},

    # ---------------------------------------------------------------
    # CORPORATE (10 markets)
    # ---------------------------------------------------------------
    {"q": "Will Tesla deliver more than 2 million vehicles in 2025?", "market_price": 0.50, "resolution": 1.0, "category": "corporate"},
    {"q": "Will any FAANG company announce layoffs exceeding 10,000 in 2025?", "market_price": 0.35, "resolution": 0.0, "category": "corporate"},
    {"q": "Will OpenAI complete an IPO or direct listing by end of 2025?", "market_price": 0.15, "resolution": 0.0, "category": "corporate"},
    {"q": "Will a major tech acquisition exceed $50B in 2025?", "market_price": 0.30, "resolution": 0.0, "category": "corporate"},
    {"q": "Will any Fortune 500 CEO be indicted in 2025?", "market_price": 0.08, "resolution": 0.0, "category": "corporate"},
    {"q": "Will Disney's streaming business become profitable in 2025?", "market_price": 0.65, "resolution": 1.0, "category": "corporate"},
    {"q": "Will any US airline declare bankruptcy in 2025?", "market_price": 0.10, "resolution": 0.0, "category": "corporate"},
    {"q": "Will Amazon's revenue exceed $650B in 2025?", "market_price": 0.55, "resolution": 1.0, "category": "corporate"},
    {"q": "Will a major pharmaceutical company recall a blockbuster drug in 2025?", "market_price": 0.20, "resolution": 0.0, "category": "corporate"},
    {"q": "Will global EV sales exceed 20 million units in 2025?", "market_price": 0.60, "resolution": 1.0, "category": "corporate"},

    # ---------------------------------------------------------------
    # HEALTH / SCIENCE (5 markets)
    # ---------------------------------------------------------------
    {"q": "Will the WHO declare a new public health emergency of international concern in 2025?", "market_price": 0.25, "resolution": 0.0, "category": "health"},
    {"q": "Will an mRNA cancer vaccine enter Phase 3 trials in 2025?", "market_price": 0.55, "resolution": 1.0, "category": "health"},
    {"q": "Will US life expectancy increase in 2025 vs 2024?", "market_price": 0.60, "resolution": 1.0, "category": "health"},
    {"q": "Will a new pandemic-potential pathogen emerge requiring global response in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "health"},
    {"q": "Will the FDA approve a CRISPR-based therapy for a common disease in 2025?", "market_price": 0.30, "resolution": 0.0, "category": "health"},

    # ---------------------------------------------------------------
    # CLIMATE / ENERGY (5 markets)
    # ---------------------------------------------------------------
    {"q": "Will 2025 be the hottest year on record globally?", "market_price": 0.55, "resolution": 1.0, "category": "climate"},
    {"q": "Will US renewable energy generation exceed 25% of total in 2025?", "market_price": 0.50, "resolution": 1.0, "category": "climate"},
    {"q": "Will oil prices average above $80/barrel for all of 2025?", "market_price": 0.40, "resolution": 0.0, "category": "climate"},
    {"q": "Will a Category 5 hurricane make US landfall in 2025?", "market_price": 0.25, "resolution": 0.0, "category": "climate"},
    {"q": "Will global carbon emissions decline year-over-year in 2025?", "market_price": 0.15, "resolution": 0.0, "category": "climate"},
]


def run_backtest():
    n_markets = len(RESOLVED_MARKETS)
    print("=" * 70)
    print(f"MiniSim Backtest — {n_markets} Resolved Markets")
    print("=" * 70)

    results = []
    total_start = time.time()

    for i, market in enumerate(RESOLVED_MARKETS):
        q_short = market["q"][:55]
        sim = swarm_score_offline(
            question=market["q"],
            n_agents=30,
            rounds=2,
            market_price=market["market_price"],
            peer_sample_size=5,
        )

        swarm_p = sim["swarm_probability_yes"]
        market_p = market["market_price"]
        actual = market["resolution"]

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
            "alpha": round(swarm_p - market_p, 4),
            "diversity_score": sim.get("diversity_score", 0),
        }
        results.append(result)

        indicator = "+" if result["swarm_beat_market"] else " "
        print(f"  [{i+1:3d}/{n_markets}] {indicator} S={swarm_p:.2f} M={market_p:.2f} A={int(actual)} "
              f"Brier {swarm_brier:.3f}/{market_brier:.3f} | {q_short}")

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

    # --- By category ---
    categories = {}
    all_cats = sorted(set(r["category"] for r in results))
    for cat in all_cats:
        cat_results = [r for r in results if r["category"] == cat]
        categories[cat] = {
            "n": len(cat_results),
            "mean_swarm_brier": round(statistics.mean([r["swarm_brier"] for r in cat_results]), 4),
            "mean_market_brier": round(statistics.mean([r["market_brier"] for r in cat_results]), 4),
            "swarm_wins": sum(1 for r in cat_results if r["swarm_beat_market"]),
            "mean_alpha": round(statistics.mean([r["alpha"] for r in cat_results]), 4),
        }

    # --- Alpha analysis ---
    positive_alpha = [r for r in results if r["swarm_beat_market"]]
    negative_alpha = [r for r in results if not r["swarm_beat_market"]]
    alpha_improvement = sum(
        r["market_brier"] - r["swarm_brier"] for r in results if r["swarm_beat_market"]
    )
    alpha_degradation = sum(
        r["swarm_brier"] - r["market_brier"] for r in results if not r["swarm_beat_market"]
    )

    # --- Top best and worst ---
    sorted_by_brier = sorted(results, key=lambda r: r["swarm_brier"])
    top5_best = sorted_by_brier[:5]
    top5_worst = sorted_by_brier[-5:]

    # Best alpha (where swarm most beat market)
    sorted_by_alpha_value = sorted(results, key=lambda r: r["market_brier"] - r["swarm_brier"], reverse=True)
    top5_alpha = sorted_by_alpha_value[:5]

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(f"BACKTEST RESULTS — {n_markets} Markets")
    print(f"{'=' * 70}")
    print(f"Overall Swarm Brier:  {overall_swarm_brier:.4f}")
    print(f"Overall Market Brier: {overall_market_brier:.4f}")
    print(f"Swarm beat market:    {swarm_wins}/{n_markets} ({100*swarm_wins/n_markets:.0f}%)")
    print(f"Total time:           {total_time:.1f}s")

    print(f"\n--- By Category ---")
    print(f"  {'Category':<14} {'N':>3}  {'Swarm':>7}  {'Market':>7}  {'Wins':>6}  {'Alpha':>7}")
    for cat, data in sorted(categories.items()):
        print(f"  {cat:<14} {data['n']:>3}  {data['mean_swarm_brier']:>7.4f}  "
              f"{data['mean_market_brier']:>7.4f}  {data['swarm_wins']:>3}/{data['n']:<2}  "
              f"{data['mean_alpha']:>+7.4f}")

    print(f"\n--- Calibration Curve ---")
    print(f"  {'Bucket':<12} {'Count':>5}  {'Predicted':>10}  {'Actual':>10}  {'Gap':>6}")
    for label, data in sorted(buckets.items()):
        gap = data["actual_resolution_rate"] - data["mean_predicted"]
        print(f"  {label:<12} {data['count']:>5}  {data['mean_predicted']:>10.3f}  "
              f"{data['actual_resolution_rate']:>10.3f}  {gap:>+6.3f}")

    print(f"\n--- Alpha Analysis ---")
    print(f"  Total Brier improvement (wins):   {alpha_improvement:.3f} across {len(positive_alpha)} markets")
    print(f"  Total Brier degradation (losses):  {alpha_degradation:.3f} across {len(negative_alpha)} markets")
    print(f"  Net alpha: {alpha_improvement - alpha_degradation:+.3f}")

    print(f"\n--- Top 5 Best Calls ---")
    for r in top5_best:
        print(f"  Brier={r['swarm_brier']:.3f} | S={r['swarm_probability']:.2f} M={r['market_price']:.2f} "
              f"A={int(r['resolution'])} | {r['question'][:55]}")

    print(f"\n--- Top 5 Worst Calls ---")
    for r in top5_worst:
        print(f"  Brier={r['swarm_brier']:.3f} | S={r['swarm_probability']:.2f} M={r['market_price']:.2f} "
              f"A={int(r['resolution'])} | {r['question'][:55]}")

    print(f"\n--- Top 5 Alpha (Where Swarm Beat Market Most) ---")
    for r in top5_alpha:
        improvement = r["market_brier"] - r["swarm_brier"]
        print(f"  Alpha={improvement:+.3f} | S={r['swarm_probability']:.2f} M={r['market_price']:.2f} "
              f"A={int(r['resolution'])} | {r['question'][:55]}")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    output = {
        "n_markets": n_markets,
        "overall_swarm_brier": round(overall_swarm_brier, 4),
        "overall_market_brier": round(overall_market_brier, 4),
        "swarm_wins": swarm_wins,
        "win_rate": round(swarm_wins / n_markets, 4),
        "net_alpha": round(alpha_improvement - alpha_degradation, 4),
        "total_time_seconds": round(total_time, 2),
        "calibration_curve": buckets,
        "by_category": categories,
        "top5_best": top5_best,
        "top5_worst": top5_worst,
        "top5_alpha": top5_alpha,
        "all_results": results,
    }

    with open("results/backtest_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: results/backtest_results.json")

    return output


if __name__ == "__main__":
    run_backtest()
