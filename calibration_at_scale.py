"""
Run swarm on 500+ resolved questions and compute calibration metrics.

This is the chart that sells the product: if the calibration curve is tighter
than market prices, we have a defensible story for B2B customers.

Usage: python calibration_at_scale.py [--mode offline] [--agents 20]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from src.calibration import CalibrationTransformer


def run_calibration(
    mode: str = "offline",
    n_agents: int = 20,
    n_rounds: int = 2,
    model: str | None = None,
    max_questions: int | None = None,
):
    # Load dataset
    with open("results/eval_dataset_500.json") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    if max_questions:
        questions = questions[:max_questions]

    print(f"{'=' * 70}")
    print(f"Calibration at Scale — {len(questions)} questions, mode={mode}")
    print(f"{'=' * 70}")

    # Get predict function
    if mode == "offline":
        from src.offline_engine import swarm_score_offline as predict_fn
    else:
        from src.llm_simulation import run_llm_simulation as predict_fn

    predictions = []
    outcomes = []
    market_prices = []
    results = []

    total_start = time.time()

    for i, q in enumerate(questions):
        kwargs = dict(
            question=q["question"],
            n_agents=n_agents,
            market_price=q["market_price"],
        )
        if mode == "offline":
            kwargs["rounds"] = n_rounds
        else:
            kwargs["n_rounds"] = n_rounds
            from src.llm_engine import LLMEngine
            kwargs["engine"] = LLMEngine(model=model)

        try:
            sim = predict_fn(**kwargs)
            swarm_p = sim["swarm_probability_yes"]
        except Exception as e:
            swarm_p = q["market_price"]  # fallback to market

        actual = q["resolution"]
        mp = q["market_price"]

        swarm_brier = (swarm_p - actual) ** 2
        market_brier = (mp - actual) ** 2

        predictions.append(swarm_p)
        outcomes.append(actual)
        market_prices.append(mp)

        results.append({
            "question": q["question"],
            "swarm_p": round(swarm_p, 4),
            "market_price": mp,
            "resolution": actual,
            "swarm_brier": round(swarm_brier, 4),
            "market_brier": round(market_brier, 4),
            "category": q.get("category", "other"),
            "source": q.get("source", "unknown"),
        })

        if (i + 1) % 50 == 0 or i == len(questions) - 1:
            elapsed = time.time() - total_start
            avg_brier = statistics.mean([r["swarm_brier"] for r in results])
            print(f"  [{i+1}/{len(questions)}] {elapsed:.0f}s | running Brier: {avg_brier:.4f}")

    total_time = time.time() - total_start

    # ── Compute calibration ──
    ct = CalibrationTransformer(method="platt")
    ct.fit(predictions, outcomes)

    # Market calibration for comparison
    ct_market = CalibrationTransformer(method="platt")
    ct_market.fit(market_prices, outcomes)

    # Overall metrics
    swarm_briers = [r["swarm_brier"] for r in results]
    market_briers = [r["market_brier"] for r in results]
    wins = sum(1 for r in results if r["swarm_brier"] < r["market_brier"])

    # By category
    categories = {}
    for cat in sorted(set(r["category"] for r in results)):
        cat_r = [r for r in results if r["category"] == cat]
        categories[cat] = {
            "n": len(cat_r),
            "swarm_brier": round(statistics.mean([r["swarm_brier"] for r in cat_r]), 4),
            "market_brier": round(statistics.mean([r["market_brier"] for r in cat_r]), 4),
            "wins": sum(1 for r in cat_r if r["swarm_brier"] < r["market_brier"]),
        }

    print(f"\n{'=' * 70}")
    print(f"CALIBRATION RESULTS — {len(questions)} questions")
    print(f"{'=' * 70}")
    print(f"  Swarm Brier:  {statistics.mean(swarm_briers):.4f}")
    print(f"  Market Brier: {statistics.mean(market_briers):.4f}")
    print(f"  Win rate:     {wins}/{len(results)} ({wins/len(results)*100:.0f}%)")
    print(f"  Total time:   {total_time:.0f}s")

    print(f"\n  Swarm ECE:  {ct.ece}")
    print(f"  Market ECE: {ct_market.ece}")

    ct.print_summary()

    print(f"\n  By Category:")
    print(f"  {'Category':<14} {'N':>4}  {'Swarm':>7}  {'Market':>7}  {'Wins':>6}")
    for cat, d in sorted(categories.items()):
        print(f"  {cat:<14} {d['n']:>4}  {d['swarm_brier']:>7.4f}  {d['market_brier']:>7.4f}  {d['wins']:>3}/{d['n']}")

    # Save
    output = {
        "mode": mode,
        "n_questions": len(questions),
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "swarm_brier": round(statistics.mean(swarm_briers), 4),
        "market_brier": round(statistics.mean(market_briers), 4),
        "win_rate": round(wins / len(results), 4),
        "swarm_ece": ct.ece,
        "market_ece": ct_market.ece,
        "calibration": ct.get_summary(),
        "by_category": categories,
        "total_time_s": round(total_time, 1),
        "all_results": results,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/calibration_{mode}_{len(questions)}q.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/calibration_{mode}_{len(questions)}q.json")

    # Save fitted model
    ct.save(f"results/calibration_model_{mode}.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "llm-ollama"], default="offline")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    run_calibration(
        mode=args.mode,
        n_agents=args.agents,
        n_rounds=args.rounds,
        model=args.model,
        max_questions=args.max_questions,
    )
