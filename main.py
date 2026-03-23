"""
MiniSim CLI — Swarm prediction engine.
Usage: python main.py --question "Will X happen?" --agents 50 --rounds 3 --market-price 0.40
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.kalshi_bridge import swarm_score_kalshi_market


def print_histogram(histogram: dict):
    """Print a text-based histogram of opinion distribution."""
    print("\n--- Opinion Distribution ---")
    max_count = max(histogram.values()) if histogram else 1
    for bucket, count in sorted(histogram.items()):
        bar = "#" * int(40 * count / max_count) if max_count > 0 else ""
        print(f"  {bucket}: {bar} ({count})")


def print_top_voices(result: dict):
    """Print top YES and NO voices."""
    print("\n--- Top 3 YES Voices ---")
    for v in result.get("top_yes_voices", []):
        print(f"  {v['name']} ({v['background']})")
        print(f"    Score: {v['final_score']:.2f} (from {v['initial_score']:.2f}), Confidence: {v['confidence']:.2f}")
        reasoning = v.get("reasoning", "")
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        print(f"    Reasoning: {reasoning}")
        print()

    print("--- Top 3 NO Voices ---")
    for v in result.get("top_no_voices", []):
        print(f"  {v['name']} ({v['background']})")
        print(f"    Score: {v['final_score']:.2f} (from {v['initial_score']:.2f}), Confidence: {v['confidence']:.2f}")
        reasoning = v.get("reasoning", "")
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        print(f"    Reasoning: {reasoning}")
        print()


def print_timing(timing: dict):
    """Print per-stage timing."""
    print("\n--- Timing ---")
    print(f"  World Build:   {timing['world_build_ms']:>8,} ms")
    print(f"  Agent Gen:     {timing['agent_gen_ms']:>8,} ms")
    print(f"  Sim Loop:      {timing['sim_loop_ms']:>8,} ms")
    total = timing['world_build_ms'] + timing['agent_gen_ms'] + timing['sim_loop_ms']
    print(f"  Total:         {total:>8,} ms")

    # Identify bottleneck
    stages = {
        "World Build": timing['world_build_ms'],
        "Agent Gen": timing['agent_gen_ms'],
        "Sim Loop": timing['sim_loop_ms'],
    }
    bottleneck = max(stages, key=stages.get)
    print(f"  Bottleneck:    {bottleneck} ({stages[bottleneck]:,} ms)")


def main():
    parser = argparse.ArgumentParser(description="MiniSim Swarm Prediction Engine")
    parser.add_argument("--question", "-q", required=True, help="Prediction question")
    parser.add_argument("--context", "-c", default="", help="Additional context")
    parser.add_argument("--agents", "-a", type=int, default=50, help="Number of agents")
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Number of deliberation rounds")
    parser.add_argument("--market-price", "-m", type=float, default=None, help="Current market price (0-1)")
    parser.add_argument("--peer-sample-size", "-p", type=int, default=5, help="Peers seen per round")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    print(f"MiniSim — Swarm Prediction Engine")
    print(f"Question: {args.question}")
    print(f"Agents: {args.agents}, Rounds: {args.rounds}, Peer Sample: {args.peer_sample_size}")
    if args.market_price is not None:
        print(f"Market Price: {args.market_price:.2f}")
    print("-" * 60)

    result = swarm_score_kalshi_market(
        question=args.question,
        context=args.context,
        n_agents=args.agents,
        rounds=args.rounds,
        market_price=args.market_price,
        peer_sample_size=args.peer_sample_size,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"SWARM PROBABILITY (YES): {result['swarm_probability_yes']:.4f}")
    print(f"Mean: {result['mean_score']:.4f} | Median: {result['median_score']:.4f} | StDev: {result['stdev']:.4f}")
    print(f"95% CI: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
    print(f"Clusters — YES: {result['clusters']['yes_leaning']}, NO: {result['clusters']['no_leaning']}, Uncertain: {result['clusters']['uncertain']}")

    if args.market_price is not None:
        delta = result.get("swarm_vs_market_delta", 0)
        direction = "ABOVE" if delta > 0 else "BELOW"
        print(f"Swarm vs Market: {direction} by {abs(delta):.4f}")

    print_timing(result["timing"])
    print_histogram(result.get("histogram", {}))
    print_top_voices(result)

    # Save to file
    output_path = args.output
    if not output_path:
        os.makedirs("results", exist_ok=True)
        safe_q = "".join(c if c.isalnum() or c in " -_" else "" for c in args.question)[:60].strip().replace(" ", "_")
        output_path = f"results/{safe_q}.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
