"""
Track Record System — Persists every prediction and auto-resolves when markets settle.

Maintains a JSON-based ledger of all swarm predictions with timestamps.
When markets resolve, computes accuracy metrics and builds calibration history.

Usage:
  from src.track_record import TrackRecord
  tr = TrackRecord()
  tr.log_prediction(question, swarm_p, market_price, source, ticker)
  tr.resolve_predictions()  # check if any logged predictions have resolved
  tr.print_summary()
"""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime
from pathlib import Path


TRACK_RECORD_PATH = "results/track_record.json"


class TrackRecord:
    def __init__(self, path: str = TRACK_RECORD_PATH):
        self.path = path
        self.predictions = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.predictions, f, indent=2)

    def log_prediction(
        self,
        question: str,
        swarm_probability: float,
        market_price: float,
        source: str = "unknown",
        ticker: str = "",
        confidence_interval: list | None = None,
        n_agents: int = 0,
        n_rounds: int = 0,
        edge: float | None = None,
    ) -> dict:
        """Log a new swarm prediction."""
        pred = {
            "id": len(self.predictions),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "swarm_probability": round(swarm_probability, 4),
            "market_price": round(market_price, 4),
            "edge": round(edge if edge is not None else swarm_probability - market_price, 4),
            "source": source,
            "ticker": ticker,
            "confidence_interval": confidence_interval or [],
            "n_agents": n_agents,
            "n_rounds": n_rounds,
            "resolution": None,
            "resolved_at": None,
            "swarm_brier": None,
            "market_brier": None,
            "swarm_beat_market": None,
        }
        self.predictions.append(pred)
        self._save()
        return pred

    def resolve(self, pred_id: int, resolution: float):
        """Manually resolve a prediction (1.0 = YES, 0.0 = NO)."""
        for p in self.predictions:
            if p["id"] == pred_id:
                p["resolution"] = resolution
                p["resolved_at"] = datetime.utcnow().isoformat() + "Z"
                p["swarm_brier"] = round((p["swarm_probability"] - resolution) ** 2, 4)
                p["market_brier"] = round((p["market_price"] - resolution) ** 2, 4)
                p["swarm_beat_market"] = p["swarm_brier"] < p["market_brier"]
                self._save()
                return p
        return None

    def resolve_from_kalshi(self):
        """Auto-resolve predictions by checking Kalshi for settled markets."""
        from src.markets.kalshi_client import get_market

        unresolved = [p for p in self.predictions if p["resolution"] is None and p["source"] == "kalshi"]
        resolved_count = 0

        for p in unresolved:
            if not p.get("ticker"):
                continue
            try:
                m = get_market(p["ticker"])
                result = m.get("result", "")
                if result == "yes":
                    self.resolve(p["id"], 1.0)
                    resolved_count += 1
                elif result == "no":
                    self.resolve(p["id"], 0.0)
                    resolved_count += 1
            except Exception:
                continue

        return resolved_count

    def get_resolved(self) -> list[dict]:
        """Get all resolved predictions."""
        return [p for p in self.predictions if p["resolution"] is not None]

    def get_unresolved(self) -> list[dict]:
        """Get all pending predictions."""
        return [p for p in self.predictions if p["resolution"] is None]

    def compute_metrics(self) -> dict:
        """Compute overall accuracy metrics from resolved predictions."""
        resolved = self.get_resolved()
        if not resolved:
            return {"n_resolved": 0, "message": "No resolved predictions yet."}

        swarm_briers = [p["swarm_brier"] for p in resolved]
        market_briers = [p["market_brier"] for p in resolved]
        wins = sum(1 for p in resolved if p["swarm_beat_market"])

        # Calibration curve
        buckets = {}
        for bstart in [i / 10 for i in range(10)]:
            bend = bstart + 0.1
            label = f"{bstart:.1f}-{bend:.1f}"
            items = [p for p in resolved if bstart <= p["swarm_probability"] < bend]
            if bend == 1.0:
                items += [p for p in resolved if p["swarm_probability"] == 1.0]
            if items:
                buckets[label] = {
                    "count": len(items),
                    "mean_predicted": round(statistics.mean([p["swarm_probability"] for p in items]), 4),
                    "actual_rate": round(statistics.mean([p["resolution"] for p in items]), 4),
                }

        return {
            "n_total": len(self.predictions),
            "n_resolved": len(resolved),
            "n_unresolved": len(self.get_unresolved()),
            "overall_swarm_brier": round(statistics.mean(swarm_briers), 4),
            "overall_market_brier": round(statistics.mean(market_briers), 4),
            "win_rate": round(wins / len(resolved), 4) if len(resolved) > 0 else 0,
            "wins": wins,
            "calibration_curve": buckets,
            "best_calls": sorted(resolved, key=lambda p: p["swarm_brier"])[:5],
            "worst_calls": sorted(resolved, key=lambda p: p["swarm_brier"], reverse=True)[:5],
        }

    def print_summary(self):
        """Print a human-readable summary."""
        metrics = self.compute_metrics()

        print(f"\n{'=' * 60}")
        print(f"MiniSim Track Record")
        print(f"{'=' * 60}")

        if metrics.get("n_resolved", 0) == 0:
            print(f"  Total predictions: {metrics.get('n_total', 0)}")
            print(f"  Resolved: 0 (no accuracy data yet)")
            print(f"  Pending: {metrics.get('n_unresolved', 0)}")
            return

        print(f"  Total predictions: {metrics['n_total']}")
        print(f"  Resolved: {metrics['n_resolved']} | Pending: {metrics['n_unresolved']}")
        print(f"  Swarm Brier:  {metrics['overall_swarm_brier']:.4f}")
        print(f"  Market Brier: {metrics['overall_market_brier']:.4f}")
        print(f"  Win rate:     {metrics['wins']}/{metrics['n_resolved']} ({metrics['win_rate']*100:.0f}%)")

        if metrics.get("calibration_curve"):
            print(f"\n  --- Calibration ---")
            for label, d in sorted(metrics["calibration_curve"].items()):
                print(f"  {label}: predicted={d['mean_predicted']:.2f} actual={d['actual_rate']:.2f} (n={d['count']})")
