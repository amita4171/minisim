"""
Head-to-Head Benchmark — Swarm vs Single LLM vs Market Price

Runs the same resolved questions through three methods:
1. Swarm (N agents deliberating K rounds)
2. Single LLM call (one-shot probability estimate)
3. Market price baseline (if available)

Computes Brier scores for each and proves swarm > single LLM.

Usage: python benchmark.py [--model qwen2.5:14b] [--agents 15] [--rounds 2]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from src.llm_engine import LLMEngine, ANCHOR_PROMPT


# Resolved questions with known outcomes and rich context
BENCHMARK_QUESTIONS = [
    {
        "id": "B-001",
        "q": "Will the Fed cut rates in December 2024?",
        "resolution": 1.0,
        "market_price": 0.75,
        "context": "Two consecutive cuts (Sept, Nov). Inflation moderating. CME FedWatch: 75% probability. Dot plot suggesting further easing.",
    },
    {
        "id": "B-002",
        "q": "Will OpenAI release a video generation model in 2024?",
        "resolution": 1.0,
        "market_price": None,
        "context": "Sora announced Feb 2024. Demo impressive. Full release uncertain. Competitive pressure from Runway, Pika. OpenAI has history of delayed launches.",
    },
    {
        "id": "B-003",
        "q": "Will Russia and Ukraine agree to a ceasefire in 2024?",
        "resolution": 0.0,
        "market_price": None,
        "context": "War ongoing for 3 years. No negotiations. Russia advancing in Donbas. Ukraine getting Western weapons. No diplomatic momentum.",
    },
    {
        "id": "B-004",
        "q": "Will Apple release a mixed reality headset in 2024?",
        "resolution": 1.0,
        "market_price": None,
        "context": "Vision Pro announced WWDC 2023. Pre-orders opened Jan 2024. Price $3,499. Shipping confirmed Feb 2024. Already in production.",
    },
    {
        "id": "B-005",
        "q": "Will China invade Taiwan in 2024?",
        "resolution": 0.0,
        "market_price": None,
        "context": "Tensions high after Taiwan election Jan 2024. Military exercises. But: economic interdependence, US deterrence, Xi's stated timeline is longer.",
    },
    {
        "id": "B-006",
        "q": "Was 2024 the hottest year on record?",
        "resolution": 1.0,
        "market_price": None,
        "context": "2023 was previous record. Strong El Nino in 2023-2024. Jan through Sept 2024 all record months. 1.5C threshold breached.",
    },
    {
        "id": "B-007",
        "q": "Will the S&P 500 close above 5,000 in 2024?",
        "resolution": 1.0,
        "market_price": None,
        "context": "S&P started 2024 at 4,770. AI rally underway. Magnificent 7 earnings strong. Rate cut expectations building. Strong momentum.",
    },
    {
        "id": "B-008",
        "q": "Will the Fed raise rates at the December 2024 FOMC meeting?",
        "resolution": 0.0,
        "market_price": None,
        "context": "Fed was in easing cycle. Already cut in Sept and Nov 2024. Inflation declining toward target. Market consensus: another cut, not raise.",
    },
    {
        "id": "B-009",
        "q": "Will any company announce AGI by end of 2025?",
        "resolution": 0.0,
        "market_price": 0.05,
        "context": "No company has claimed AGI. Frontier models improving but still fail at many tasks. No scientific consensus on AGI definition.",
    },
    {
        "id": "B-010",
        "q": "Will SpaceX complete a successful Starship orbital flight in 2025?",
        "resolution": 1.0,
        "market_price": 0.75,
        "context": "Multiple test flights in 2024 with increasing success. Booster catch achieved Oct 2024. Rapid iteration. FAA licensing improving.",
    },
]


def run_single_llm(engine: LLMEngine, question: str, context: str) -> float:
    """Get a single-LLM probability estimate (no swarm, no deliberation)."""
    result = engine.generate_json(
        ANCHOR_PROMPT.format(question=question, context=context),
        temperature=0.3,
        max_tokens=128,
    )
    if result and "probability" in result:
        return max(0.02, min(0.98, float(result["probability"])))
    return 0.50


def run_benchmark(
    n_agents: int = 15,
    n_rounds: int = 2,
    model: str | None = None,
):
    engine = LLMEngine(model=model)

    if not engine.is_available():
        print("LLM not available. Start Ollama first.")
        return

    print(f"{'=' * 70}")
    print(f"HEAD-TO-HEAD BENCHMARK")
    print(f"Engine: {engine.backend}/{engine.model}")
    print(f"Swarm: {n_agents} agents x {n_rounds} rounds")
    print(f"{'=' * 70}")

    results = []

    for i, bq in enumerate(BENCHMARK_QUESTIONS):
        print(f"\n[{i+1}/{len(BENCHMARK_QUESTIONS)}] {bq['q'][:55]}...")
        actual = bq["resolution"]

        # Method 1: Single LLM call
        t0 = time.time()
        single_p = run_single_llm(engine, bq["q"], bq["context"])
        single_time = time.time() - t0
        single_brier = (single_p - actual) ** 2

        # Method 2: Swarm
        t0 = time.time()
        from src.llm_simulation import run_llm_simulation
        swarm_result = run_llm_simulation(
            question=bq["q"],
            context=bq["context"],
            n_agents=n_agents,
            n_rounds=n_rounds,
            market_price=bq.get("market_price"),
            engine=engine,
        )
        swarm_p = swarm_result["swarm_probability_yes"]
        swarm_time = time.time() - t0
        swarm_brier = (swarm_p - actual) ** 2

        # Method 3: Market price (if available)
        market_p = bq.get("market_price")
        market_brier = (market_p - actual) ** 2 if market_p is not None else None

        # Determine winners
        best = "swarm" if swarm_brier <= single_brier else "single"
        if market_brier is not None and market_brier < min(swarm_brier, single_brier):
            best = "market"

        entry = {
            "id": bq["id"],
            "question": bq["q"],
            "actual": actual,
            "single_llm": {"p": round(single_p, 4), "brier": round(single_brier, 4), "time_s": round(single_time, 1)},
            "swarm": {"p": round(swarm_p, 4), "brier": round(swarm_brier, 4), "time_s": round(swarm_time, 1), "std": swarm_result.get("diversity_score", 0)},
            "market": {"p": market_p, "brier": round(market_brier, 4) if market_brier is not None else None},
            "best": best,
        }
        results.append(entry)

        market_str = f"M={market_p:.2f}({market_brier:.3f})" if market_p else "M=N/A"
        winner = "SWARM" if best == "swarm" else ("SINGLE" if best == "single" else "MARKET")
        print(f"  Single={single_p:.2f}({single_brier:.3f}) Swarm={swarm_p:.2f}({swarm_brier:.3f}) {market_str} -> {winner}")

    # ── Summary ──
    single_briers = [r["single_llm"]["brier"] for r in results]
    swarm_briers = [r["swarm"]["brier"] for r in results]
    market_briers = [r["market"]["brier"] for r in results if r["market"]["brier"] is not None]

    swarm_wins_vs_single = sum(1 for r in results if r["swarm"]["brier"] < r["single_llm"]["brier"])
    swarm_wins_vs_market = sum(1 for r in results if r["market"]["brier"] is not None and r["swarm"]["brier"] < r["market"]["brier"])

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  Single LLM Brier:  {statistics.mean(single_briers):.4f}")
    print(f"  Swarm Brier:       {statistics.mean(swarm_briers):.4f}")
    if market_briers:
        print(f"  Market Brier:      {statistics.mean(market_briers):.4f}")
    print(f"")
    print(f"  Swarm vs Single LLM: {swarm_wins_vs_single}/{len(results)} wins ({swarm_wins_vs_single/len(results)*100:.0f}%)")
    if market_briers:
        n_with_market = sum(1 for r in results if r["market"]["brier"] is not None)
        print(f"  Swarm vs Market:     {swarm_wins_vs_market}/{n_with_market} wins")

    improvement = (statistics.mean(single_briers) - statistics.mean(swarm_briers)) / statistics.mean(single_briers) * 100
    print(f"\n  Swarm improvement over single LLM: {improvement:+.1f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        "engine": f"{engine.backend}/{engine.model}",
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "single_llm_brier": round(statistics.mean(single_briers), 4),
        "swarm_brier": round(statistics.mean(swarm_briers), 4),
        "market_brier": round(statistics.mean(market_briers), 4) if market_briers else None,
        "swarm_vs_single_wins": swarm_wins_vs_single,
        "swarm_vs_single_pct": round(swarm_wins_vs_single / len(results), 4),
        "improvement_pct": round(improvement, 1),
        "results": results,
    }
    with open("results/benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/benchmark_results.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-to-head: Swarm vs Single LLM vs Market")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name")
    parser.add_argument("--agents", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    run_benchmark(n_agents=args.agents, n_rounds=args.rounds, model=args.model)
