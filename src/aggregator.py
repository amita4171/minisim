"""
Aggregation: weighted mean with outlier detection, confidence interval,
opinion cluster identification, and top voices extraction.
"""
from __future__ import annotations

import statistics


def aggregate(agents: list[dict], market_price: float | None = None) -> dict:
    """Aggregate agent opinions into a swarm prediction."""
    final_scores = [a["score_history"][-1] for a in agents]
    confidences = [a.get("confidence", 0.5) for a in agents]

    # Weighted mean (weight by confidence)
    total_weight = sum(confidences)
    if total_weight == 0:
        swarm_prob = statistics.mean(final_scores)
    else:
        swarm_prob = sum(s * c for s, c in zip(final_scores, confidences)) / total_weight

    # Simple stats
    mean_score = statistics.mean(final_scores)
    median_score = statistics.median(final_scores)
    stdev = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0

    # Confidence interval (mean +/- 1.96 * SE)
    se = stdev / (len(final_scores) ** 0.5) if final_scores else 0
    ci_lower = max(0.0, mean_score - 1.96 * se)
    ci_upper = min(1.0, mean_score + 1.96 * se)

    # Opinion clusters: YES-leaning (>0.6), NO-leaning (<0.4), Uncertain (0.4-0.6)
    yes_cluster = [a for a in agents if a["score_history"][-1] > 0.6]
    no_cluster = [a for a in agents if a["score_history"][-1] < 0.4]
    uncertain_cluster = [a for a in agents if 0.4 <= a["score_history"][-1] <= 0.6]

    # Top voices: sort by confidence * extremity of position
    sorted_yes = sorted(
        agents,
        key=lambda a: a["score_history"][-1] * a.get("confidence", 0.5),
        reverse=True,
    )
    sorted_no = sorted(
        agents,
        key=lambda a: (1 - a["score_history"][-1]) * a.get("confidence", 0.5),
        reverse=True,
    )

    top_yes = [_voice_summary(a) for a in sorted_yes[:3]]
    top_no = [_voice_summary(a) for a in sorted_no[:3]]

    # Per-round convergence data
    n_rounds = max(len(a["score_history"]) for a in agents)
    convergence = []
    for r in range(n_rounds):
        round_scores = [
            a["score_history"][r] for a in agents if r < len(a["score_history"])
        ]
        if round_scores:
            convergence.append({
                "round": r,
                "mean_score": statistics.mean(round_scores),
                "stdev": statistics.stdev(round_scores) if len(round_scores) > 1 else 0.0,
                "min": min(round_scores),
                "max": max(round_scores),
            })

    # Opinion shift per round
    opinion_shifts = []
    for r in range(1, n_rounds):
        shifts = [
            abs(a["score_history"][r] - a["score_history"][r - 1])
            for a in agents
            if r < len(a["score_history"])
        ]
        if shifts:
            opinion_shifts.append({
                "round": r,
                "mean_shift": statistics.mean(shifts),
                "max_shift": max(shifts),
            })

    # Histogram buckets
    histogram = {}
    for bucket_start in [i / 10 for i in range(10)]:
        bucket_end = bucket_start + 0.1
        label = f"{bucket_start:.1f}-{bucket_end:.1f}"
        count = sum(1 for s in final_scores if bucket_start <= s < bucket_end)
        histogram[label] = count
    # Include 1.0 in the last bucket
    histogram["0.9-1.0"] += sum(1 for s in final_scores if s == 1.0)

    result = {
        "swarm_probability_yes": round(swarm_prob, 4),
        "mean_score": round(mean_score, 4),
        "median_score": round(median_score, 4),
        "stdev": round(stdev, 4),
        "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
        "n_agents": len(agents),
        "clusters": {
            "yes_leaning": len(yes_cluster),
            "no_leaning": len(no_cluster),
            "uncertain": len(uncertain_cluster),
        },
        "top_yes_voices": top_yes,
        "top_no_voices": top_no,
        "convergence": convergence,
        "opinion_shifts": opinion_shifts,
        "histogram": histogram,
    }

    if market_price is not None:
        result["market_price"] = market_price
        result["swarm_vs_market_delta"] = round(swarm_prob - market_price, 4)

    return result


def _voice_summary(agent: dict) -> dict:
    """Extract a summary of an agent for top voices display."""
    return {
        "name": agent["name"],
        "background": agent["background_category"],
        "final_score": agent["score_history"][-1],
        "initial_score": agent["score_history"][0],
        "confidence": agent.get("confidence", 0.5),
        "reasoning": agent.get("reasoning", ""),
        "last_reflection": agent["memory_stream"][-1] if agent["memory_stream"] else "",
    }
