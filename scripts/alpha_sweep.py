"""
Alpha Sweep — Tests different EXTREMIZATION_ALPHA values to find the optimal one.

Temporarily patches src.core.aggregator.EXTREMIZATION_ALPHA for each test value,
runs the 10 resolved HIST questions from the eval set, and compares Brier scores.

Usage:
  python scripts/alpha_sweep.py
  python scripts/alpha_sweep.py --alphas 1.0,1.25,1.5,2.0
  python scripts/alpha_sweep.py --agents 20 --rounds 3
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time

from scripts.eval_runner import EVAL_QUESTIONS
from src.core.offline_engine import swarm_score_offline
import src.core.aggregator as aggregator


DEFAULT_ALPHAS = "1.0,1.15,1.25,1.35,1.5,1.75,2.0"


def run_alpha_sweep(
    alphas: list[float],
    n_agents: int = 15,
    n_rounds: int = 2,
) -> dict:
    """Run Brier-score evaluation across multiple EXTREMIZATION_ALPHA values."""
    resolved_questions = [eq for eq in EVAL_QUESTIONS if eq["resolution"] is not None]

    print("=" * 70)
    print("Alpha Sweep — EXTREMIZATION_ALPHA Optimization")
    print(f"  Testing alphas: {alphas}")
    print(f"  Resolved questions: {len(resolved_questions)}")
    print(f"  Agents: {n_agents}, Rounds: {n_rounds}")
    print("=" * 70)

    original_alpha = aggregator.EXTREMIZATION_ALPHA
    all_results = {}

    try:
        for alpha in alphas:
            print(f"\n--- Alpha = {alpha:.2f} ---")
            aggregator.EXTREMIZATION_ALPHA = alpha

            brier_scores = []
            in_range = 0
            question_details = []

            for eq in resolved_questions:
                sim = swarm_score_offline(
                    question=eq["q"],
                    context=eq.get("context", ""),
                    n_agents=n_agents,
                    rounds=n_rounds,
                    market_price=eq["market_price"],
                )
                swarm_p = sim["swarm_probability_yes"]
                brier = (swarm_p - eq["resolution"]) ** 2
                brier_scores.append(brier)

                in_expected = eq["expected_low"] <= swarm_p <= eq["expected_high"]
                if in_expected:
                    in_range += 1

                question_details.append({
                    "id": eq["id"],
                    "question": eq["q"],
                    "resolution": eq["resolution"],
                    "swarm_p": round(swarm_p, 4),
                    "brier": round(brier, 4),
                    "in_range": in_expected,
                })

                status = " OK " if in_expected else "MISS"
                print(
                    f"  [{status}] {eq['id']:10} P={swarm_p:.3f} "
                    f"res={eq['resolution']:.0f} B={brier:.4f}"
                )

            avg_brier = sum(brier_scores) / len(brier_scores)
            print(f"  Avg Brier: {avg_brier:.4f}  In Range: {in_range}/{len(resolved_questions)}")

            all_results[alpha] = {
                "alpha": alpha,
                "avg_brier": round(avg_brier, 4),
                "in_range": in_range,
                "total_questions": len(resolved_questions),
                "questions": question_details,
            }
    finally:
        aggregator.EXTREMIZATION_ALPHA = original_alpha

    # --- Comparison table ---
    best_alpha = min(all_results, key=lambda a: all_results[a]["avg_brier"])

    print("\n" + "=" * 70)
    print(f"{'Alpha':<8} {'Brier':<8} {'In Range':<11} {'Best For'}")
    print("-" * 50)
    for alpha in alphas:
        r = all_results[alpha]
        label = ""
        if alpha == original_alpha:
            label = "Current default"
        if alpha == best_alpha:
            label = ("Lowest Brier, " + label) if label else "Lowest Brier"
        if alpha <= 1.0:
            label = label or "Conservative"
        print(
            f"{alpha:<8.2f} {r['avg_brier']:<8.3f} "
            f"{r['in_range']}/{r['total_questions']:<7}  {label}"
        )
    print("=" * 70)

    best_brier = all_results[best_alpha]["avg_brier"]
    print(f"\nRecommendation: alpha={best_alpha:.2f} (Brier={best_brier:.4f})")
    if best_alpha != original_alpha:
        current_brier = all_results.get(original_alpha, {}).get("avg_brier")
        if current_brier is not None:
            improvement = (current_brier - best_brier) / current_brier * 100
            print(
                f"  vs current default ({original_alpha}): "
                f"{improvement:+.1f}% Brier improvement"
            )

    # --- Save results ---
    output = {
        "sweep_config": {
            "alphas": alphas,
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "n_resolved_questions": len(resolved_questions),
            "original_alpha": original_alpha,
        },
        "results": {str(a): all_results[a] for a in alphas},
        "best_alpha": best_alpha,
        "best_brier": best_brier,
    }

    os.makedirs("results", exist_ok=True)
    output_path = "results/alpha_sweep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Sweep — EXTREMIZATION_ALPHA optimizer")
    parser.add_argument(
        "--alphas",
        type=str,
        default=DEFAULT_ALPHAS,
        help=f"Comma-separated alpha values to test (default: {DEFAULT_ALPHAS})",
    )
    parser.add_argument("--agents", type=int, default=15, help="Number of agents (default: 15)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds (default: 2)")
    args = parser.parse_args()

    alpha_list = [float(a.strip()) for a in args.alphas.split(",")]
    run_alpha_sweep(alphas=alpha_list, n_agents=args.agents, n_rounds=args.rounds)
