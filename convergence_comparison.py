"""
Phase 2: Compare 3-round vs 5-round convergence.
Outputs convergence_comparison.json with mean opinion shift per round.
Also reports diversity metrics (std dev) to verify no mode collapse.
"""
from __future__ import annotations

import json
import statistics
import os

from src.offline_engine import (
    build_world_offline,
    generate_population_offline,
    run_simulation_offline,
    BACKGROUNDS,
)
from src.aggregator import aggregate


QUESTIONS = [
    {"q": "Will the Fed cut rates in May 2026?", "market_price": 0.40},
    {"q": "Will there be a US government shutdown before July 2026?", "market_price": 0.35},
    {"q": "Will AI replace more than 10% of US white-collar jobs by 2028?", "market_price": 0.25},
]


def run_comparison():
    results = []

    for qdata in QUESTIONS:
        question = qdata["q"]
        print(f"\n{'='*60}")
        print(f"Question: {question}")

        world = build_world_offline(question)

        for n_rounds in [3, 5]:
            # Generate fresh population each time (same seed for fair comparison)
            agents, _ = generate_population_offline(question, world, n_agents=50, seed=42)
            agents, _ = run_simulation_offline(question, agents, n_rounds=n_rounds, peer_sample_size=5, seed=42)

            # Compute per-round metrics
            round_data = []
            for r in range(n_rounds + 1):  # +1 for initial (round 0)
                scores = [a["score_history"][r] for a in agents if r < len(a["score_history"])]
                if not scores:
                    continue
                mean_s = statistics.mean(scores)
                std_s = statistics.stdev(scores) if len(scores) > 1 else 0
                round_data.append({
                    "round": r,
                    "mean_score": round(mean_s, 4),
                    "stdev": round(std_s, 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                })

            # Compute mean opinion shift per round
            shifts = []
            for r in range(1, n_rounds + 1):
                round_shifts = [
                    abs(a["score_history"][r] - a["score_history"][r - 1])
                    for a in agents
                    if r < len(a["score_history"])
                ]
                if round_shifts:
                    shifts.append({
                        "round": r,
                        "mean_shift": round(statistics.mean(round_shifts), 4),
                        "max_shift": round(max(round_shifts), 4),
                    })

            final_scores = [a["score_history"][-1] for a in agents]
            initial_scores = [a["score_history"][0] for a in agents]
            final_std = statistics.stdev(final_scores) if len(final_scores) > 1 else 0
            initial_std = statistics.stdev(initial_scores) if len(initial_scores) > 1 else 0

            # Diversity by temp tier
            tier_diversity = {}
            for tier in ["analyst", "calibrator", "contrarian", "creative"]:
                tier_agents = [a for a in agents if a.get("temp_tier") == tier]
                if tier_agents:
                    tier_scores = [a["score_history"][-1] for a in tier_agents]
                    tier_diversity[tier] = {
                        "n": len(tier_agents),
                        "mean": round(statistics.mean(tier_scores), 4),
                        "stdev": round(statistics.stdev(tier_scores), 4) if len(tier_scores) > 1 else 0,
                    }

            entry = {
                "question": question,
                "n_rounds": n_rounds,
                "n_agents": 50,
                "round_data": round_data,
                "opinion_shifts": shifts,
                "initial_std": round(initial_std, 4),
                "final_std": round(final_std, 4),
                "diversity_preserved": final_std > 0.15,
                "convergence_round": _find_convergence_round(shifts),
                "tier_diversity": tier_diversity,
            }
            results.append(entry)

            # Print summary
            print(f"\n  {n_rounds} rounds:")
            print(f"    Initial std: {initial_std:.4f} | Final std: {final_std:.4f}")
            print(f"    Diversity preserved (>0.15): {'YES' if final_std > 0.15 else 'NO'}")
            print(f"    Shifts per round: {[s['mean_shift'] for s in shifts]}")
            if tier_diversity:
                print(f"    Tier diversity:")
                for tier, d in tier_diversity.items():
                    print(f"      {tier}: n={d['n']}, mean={d['mean']:.3f}, std={d['stdev']:.3f}")

    # Save
    os.makedirs("results", exist_ok=True)
    output = {
        "comparison": results,
        "n_archetypes": len(BACKGROUNDS),
        "summary": {
            "3_round_final_stds": [r["final_std"] for r in results if r["n_rounds"] == 3],
            "5_round_final_stds": [r["final_std"] for r in results if r["n_rounds"] == 5],
            "mode_collapse_detected": any(r["final_std"] < 0.10 for r in results),
        },
    }

    with open("results/convergence_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total archetypes: {len(BACKGROUNDS)}")
    print(f"Results saved to: results/convergence_comparison.json")
    print(f"Mode collapse detected: {output['summary']['mode_collapse_detected']}")

    return output


def _find_convergence_round(shifts: list[dict]) -> int | None:
    """Find the round where mean shift drops below 0.01 (convergence)."""
    for s in shifts:
        if s["mean_shift"] < 0.01:
            return s["round"]
    return None


if __name__ == "__main__":
    run_comparison()
