#!/usr/bin/env python3
"""
Resolve Metaculus predictions manually when API resolution data is unavailable.

Supports two modes:
  Interactive:  python scripts/resolve_manual.py
  Batch:        python scripts/resolve_manual.py --batch resolutions.json

Batch JSON format: {"42684": 1.0, "42779": 0.0, ...}
  Keys are question ticker IDs, values are resolution outcomes (1.0=YES, 0.0=NO).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow direct execution from scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import Database


def fetch_unresolved(db: Database) -> list[dict]:
    """Return all unresolved metaculus predictions."""
    rows = db.conn.execute(
        "SELECT id, question, swarm_probability, ticker "
        "FROM predictions "
        "WHERE source='metaculus' AND resolution IS NULL"
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_interactive(db: Database, predictions: list[dict]) -> list[tuple[int, float]]:
    """Present each question interactively and collect resolutions.

    Returns list of (pred_id, resolution) tuples that were resolved.
    """
    resolved = []
    for pred in predictions:
        print(f"\n--- Prediction #{pred['id']} ---")
        print(f"  Ticker:     {pred['ticker']}")
        print(f"  Question:   {pred['question']}")
        print(f"  Our P(YES): {pred['swarm_probability']:.1%}")
        answer = input("  Resolution [1.0=YES / 0.0=NO / Enter=skip]: ").strip()
        if answer == "":
            continue
        try:
            val = float(answer)
            if val not in (0.0, 1.0):
                print("  Skipping — must be 0.0 or 1.0")
                continue
        except ValueError:
            print("  Skipping — invalid input")
            continue

        db.resolve(pred["id"], val)
        resolved.append((pred["id"], val))
        print(f"  -> Resolved as {val}")
    return resolved


def resolve_batch(db: Database, predictions: list[dict], batch_path: str) -> list[tuple[int, float]]:
    """Resolve predictions from a JSON file mapping ticker -> resolution."""
    with open(batch_path) as f:
        batch_data: dict[str, float] = json.load(f)

    # Build ticker -> prediction mapping
    ticker_map: dict[str, dict] = {}
    for pred in predictions:
        ticker = pred.get("ticker")
        if ticker:
            ticker_map[ticker] = pred

    resolved = []
    for ticker, resolution in batch_data.items():
        if ticker not in ticker_map:
            print(f"  Warning: ticker '{ticker}' not found among unresolved predictions, skipping")
            continue
        pred = ticker_map[ticker]
        db.resolve(pred["id"], resolution)
        resolved.append((pred["id"], resolution))
        print(f"  Resolved #{pred['id']} (ticker={ticker}): {resolution}")

    return resolved


def print_summary(db: Database, resolved: list[tuple[int, float]]) -> str:
    """Print resolution summary and return the output string."""
    lines = []
    n = len(resolved)
    lines.append(f"\n{'='*50}")
    lines.append(f"RESOLUTION SUMMARY")
    lines.append(f"{'='*50}")
    lines.append(f"Resolved: {n}")

    if n == 0:
        lines.append("No predictions were resolved.")
        output = "\n".join(lines)
        print(output)
        return output

    # Gather resolved predictions with Brier scores
    pred_ids = [pid for pid, _ in resolved]
    placeholders = ",".join("?" * len(pred_ids))
    rows = db.conn.execute(
        f"SELECT id, question, swarm_probability, market_price, "
        f"swarm_brier, market_brier, swarm_beat_market, resolution "
        f"FROM predictions WHERE id IN ({placeholders})",
        pred_ids,
    ).fetchall()
    rows = [dict(r) for r in rows]

    briers = [r["swarm_brier"] for r in rows if r["swarm_brier"] is not None]
    avg_brier = sum(briers) / len(briers) if briers else None

    wins = sum(1 for r in rows if r["swarm_beat_market"] == 1)
    has_market = sum(1 for r in rows if r["market_brier"] is not None)
    win_rate = wins / has_market if has_market > 0 else None

    if avg_brier is not None:
        lines.append(f"Avg Brier:  {avg_brier:.4f}")
    if win_rate is not None:
        lines.append(f"Win rate vs market: {wins}/{has_market} ({win_rate:.1%})")

    # Best/worst 5 by Brier score
    scored = [r for r in rows if r["swarm_brier"] is not None]
    scored.sort(key=lambda r: r["swarm_brier"])

    if scored:
        lines.append(f"\nBest 5 (lowest Brier):")
        for r in scored[:5]:
            q = r["question"][:60]
            lines.append(f"  {r['swarm_brier']:.4f}  #{r['id']}  {q}")

        lines.append(f"\nWorst 5 (highest Brier):")
        for r in scored[-5:]:
            q = r["question"][:60]
            lines.append(f"  {r['swarm_brier']:.4f}  #{r['id']}  {q}")

    output = "\n".join(lines)
    print(output)
    return output


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Manually resolve Metaculus predictions"
    )
    parser.add_argument(
        "--batch",
        metavar="JSON_FILE",
        help='Path to JSON file with resolutions: {"ticker_id": 1.0, ...}',
    )
    parsed = parser.parse_args(args)

    db = Database()

    try:
        predictions = fetch_unresolved(db)
        print(f"Found {len(predictions)} unresolved Metaculus predictions.")

        if not predictions:
            print("Nothing to resolve.")
            return

        if parsed.batch:
            resolved = resolve_batch(db, predictions, parsed.batch)
        else:
            resolved = resolve_interactive(db, predictions)

        print_summary(db, resolved)
    finally:
        db.close()


if __name__ == "__main__":
    main()
