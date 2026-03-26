"""
Alpha Sweep — Find optimal EXTREMIZATION_ALPHA from resolved predictions.

Re-extremizes stored swarm_probability values with different alpha values
using the formula: p_ext = p^a / (p^a + (1-p)^a)

No LLM calls needed — works purely from the SQLite database.

Usage:
  python scripts/alpha_sweep.py
  python scripts/alpha_sweep.py --alphas 1.0,1.1,1.2,1.3,1.5,1.8,2.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import Database
import src.core.aggregator as aggregator

DEFAULT_ALPHAS = "1.0,1.1,1.2,1.3,1.5,1.8,2.0"


def extremize(p: float, alpha: float) -> float:
    """Apply extremization: p^a / (p^a + (1-p)^a)"""
    if p <= 0.0 or p >= 1.0:
        return p
    pa = p ** alpha
    qa = (1.0 - p) ** alpha
    return pa / (pa + qa)


def de_extremize(p_ext: float, old_alpha: float) -> float:
    """Reverse extremization to recover the raw probability."""
    if p_ext <= 0.0 or p_ext >= 1.0:
        return p_ext
    # Inverse: solve p^a / (p^a + (1-p)^a) = p_ext for p
    # p_ext * (1-p)^a = (1-p_ext) * p^a
    # (p/(1-p))^a = p_ext / (1-p_ext)
    # p/(1-p) = (p_ext / (1-p_ext))^(1/a)
    ratio = (p_ext / (1.0 - p_ext)) ** (1.0 / old_alpha)
    return ratio / (1.0 + ratio)


def run_alpha_sweep(alphas: list[float]) -> dict:
    """Run Brier-score evaluation across multiple alpha values."""
    db = Database()

    rows = db.conn.execute(
        "SELECT id, question, swarm_probability, market_price, resolution "
        "FROM predictions WHERE resolution IS NOT NULL"
    ).fetchall()
    predictions = [dict(r) for r in rows]
    db.close()

    if not predictions:
        print("No resolved predictions found. Run resolve_manual.py first.")
        return {}

    current_alpha = aggregator.EXTREMIZATION_ALPHA

    print("=" * 70)
    print("Alpha Sweep — EXTREMIZATION_ALPHA Optimization")
    print(f"  Current alpha: {current_alpha}")
    print(f"  Testing alphas: {alphas}")
    print(f"  Resolved predictions: {len(predictions)}")
    print("=" * 70)

    all_results = {}

    for alpha in alphas:
        brier_scores = []
        details = []

        for pred in predictions:
            p_stored = pred["swarm_probability"]
            resolution = pred["resolution"]

            # De-extremize from current alpha, then re-extremize with test alpha
            p_raw = de_extremize(p_stored, current_alpha)
            p_new = extremize(p_raw, alpha)

            brier = (p_new - resolution) ** 2
            brier_scores.append(brier)

            details.append({
                "id": pred["id"],
                "question": pred["question"][:60],
                "resolution": resolution,
                "p_stored": round(p_stored, 4),
                "p_raw": round(p_raw, 4),
                "p_new": round(p_new, 4),
                "brier": round(brier, 4),
            })

        avg_brier = sum(brier_scores) / len(brier_scores)
        all_results[alpha] = {
            "alpha": alpha,
            "avg_brier": round(avg_brier, 4),
            "predictions": details,
        }

    # --- Comparison table ---
    best_alpha = min(all_results, key=lambda a: all_results[a]["avg_brier"])

    print(f"\n{'Alpha':<8} {'Avg Brier':<12} {'Notes'}")
    print("-" * 50)
    for alpha in alphas:
        r = all_results[alpha]
        notes = []
        if alpha == current_alpha:
            notes.append("Current default")
        if alpha == best_alpha:
            notes.append("BEST")
        if alpha == 1.0:
            notes.append("No extremization")
        print(f"{alpha:<8.2f} {r['avg_brier']:<12.4f} {', '.join(notes)}")

    # Per-prediction breakdown for best alpha
    print(f"\n--- Best alpha = {best_alpha:.2f} detail ---")
    for d in all_results[best_alpha]["predictions"]:
        print(
            f"  {d['id']:3d}  P_stored={d['p_stored']:.3f} -> P_raw={d['p_raw']:.3f} "
            f"-> P_new={d['p_new']:.3f}  res={d['resolution']:.0f}  B={d['brier']:.4f}  "
            f"{d['question']}"
        )

    best_brier = all_results[best_alpha]["avg_brier"]
    current_brier = all_results.get(current_alpha, {}).get("avg_brier")

    print(f"\n{'='*70}")
    print(f"Recommendation: alpha={best_alpha:.2f} (Avg Brier={best_brier:.4f})")
    if current_brier is not None and best_alpha != current_alpha:
        improvement = (current_brier - best_brier) / current_brier * 100
        print(f"  vs current ({current_alpha}): {improvement:+.1f}% Brier improvement")
    print(f"{'='*70}")

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "current_alpha": current_alpha,
        "best_alpha": best_alpha,
        "best_brier": best_brier,
        "n_predictions": len(predictions),
        "results": {str(a): {"alpha": a, "avg_brier": all_results[a]["avg_brier"]} for a in alphas},
    }
    with open("results/alpha_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/alpha_sweep.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha Sweep — EXTREMIZATION_ALPHA optimizer")
    parser.add_argument(
        "--alphas", type=str, default=DEFAULT_ALPHAS,
        help=f"Comma-separated alpha values (default: {DEFAULT_ALPHAS})",
    )
    args = parser.parse_args()
    alpha_list = [float(a.strip()) for a in args.alphas.split(",")]
    run_alpha_sweep(alphas=alpha_list)
