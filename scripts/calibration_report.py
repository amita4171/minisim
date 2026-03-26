"""
Calibration Report — Generate comprehensive calibration metrics from resolved predictions.

Loads resolved predictions from the SQLite database and/or the 544-question eval
dataset, then computes Brier score, ECE, calibration curve, win rate vs market,
sharpness, and resolution rate. Outputs an ASCII calibration plot and saves
a full JSON report.

Usage:
    python scripts/calibration_report.py --source db
    python scripts/calibration_report.py --source eval-dataset
    python scripts/calibration_report.py --source both
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import statistics
from datetime import datetime

from src.core.calibration import CalibrationTransformer
from src.db.database import Database

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_db() -> list[dict]:
    """Load all resolved predictions from the SQLite database."""
    db = Database()
    rows = db.get_predictions(resolved_only=True, limit=100_000)
    db.close()
    records = []
    for r in rows:
        records.append({
            "question": r["question"],
            "swarm_p": r["swarm_probability"],
            "market_price": r.get("market_price"),
            "resolution": r["resolution"],
            "category": r.get("category") or "unknown",
            "source": r.get("source") or "db",
            "origin": "db",
        })
    return records


def load_from_eval_dataset() -> list[dict]:
    """Load predictions from the 544-question eval dataset results file.

    We first try the calibration results file (which has swarm_p), and fall
    back to the raw eval dataset (which only has market_price — no swarm
    prediction).
    """
    cal_path = os.path.join("results", "calibration_offline_544q.json")
    eval_path = os.path.join("results", "eval_dataset_500.json")

    # Prefer the calibration results — it has swarm predictions
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            data = json.load(f)
        results = data.get("all_results", [])
        records = []
        for r in results:
            records.append({
                "question": r["question"],
                "swarm_p": r["swarm_p"],
                "market_price": r.get("market_price"),
                "resolution": r["resolution"],
                "category": r.get("category", "other"),
                "source": r.get("source", "eval"),
                "origin": "eval-dataset",
            })
        return records

    # Fallback: raw eval dataset (no swarm predictions — use market_price as proxy)
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data = json.load(f)
        questions = data.get("questions", [])
        records = []
        for q in questions:
            if q.get("resolution") is not None:
                records.append({
                    "question": q["question"],
                    "swarm_p": q["market_price"],  # no swarm data available
                    "market_price": q["market_price"],
                    "resolution": q["resolution"],
                    "category": q.get("category", "other"),
                    "source": q.get("source", "eval"),
                    "origin": "eval-dataset",
                })
        return records

    return []


def load_records(source: str) -> list[dict]:
    """Load records based on --source flag."""
    if source == "db":
        records = load_from_db()
    elif source == "eval-dataset":
        records = load_from_eval_dataset()
    elif source == "both":
        db_records = load_from_db()
        eval_records = load_from_eval_dataset()
        # De-duplicate by question text (prefer db record)
        seen = {r["question"] for r in db_records}
        for r in eval_records:
            if r["question"] not in seen:
                db_records.append(r)
                seen.add(r["question"])
        records = db_records
    else:
        raise ValueError(f"Unknown source: {source}")
    return records


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_brier(records: list[dict]) -> tuple[float, dict[str, float]]:
    """Compute overall Brier score and Brier score by category."""
    briers = [(r["swarm_p"] - r["resolution"]) ** 2 for r in records]
    overall = statistics.mean(briers)

    by_category: dict[str, list[float]] = {}
    for r in records:
        cat = r["category"] or "unknown"
        by_category.setdefault(cat, []).append((r["swarm_p"] - r["resolution"]) ** 2)

    cat_brier = {cat: statistics.mean(bs) for cat, bs in sorted(by_category.items())}
    return overall, cat_brier


def compute_ece(predictions: list[float], outcomes: list[float]) -> float:
    """Compute ECE using CalibrationTransformer."""
    ct = CalibrationTransformer(method="platt", n_buckets=10)
    ct.fit(predictions, outcomes)
    return ct.ece if ct.ece is not None else 0.0


def compute_calibration_curve(predictions: list[float], outcomes: list[float]) -> list[dict]:
    """Compute decile-bucket calibration curve."""
    n_buckets = 10
    curve = []
    for i in range(n_buckets):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        bucket_preds = []
        bucket_outcomes = []
        for p, y in zip(predictions, outcomes):
            if lo <= p < hi or (i == n_buckets - 1 and p >= hi):
                bucket_preds.append(p)
                bucket_outcomes.append(y)
        if bucket_preds:
            mean_pred = statistics.mean(bucket_preds)
            actual_rate = statistics.mean(bucket_outcomes)
            curve.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "count": len(bucket_preds),
                "mean_predicted": round(mean_pred, 4),
                "actual_rate": round(actual_rate, 4),
                "gap": round(actual_rate - mean_pred, 4),
            })
        else:
            curve.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "count": 0,
                "mean_predicted": round((lo + hi) / 2, 4),
                "actual_rate": None,
                "gap": None,
            })
    return curve


def compute_win_rate(records: list[dict]) -> tuple[int, int, float]:
    """Compute how often swarm beats market price (lower Brier = win)."""
    wins = 0
    eligible = 0
    for r in records:
        if r["market_price"] is not None:
            swarm_brier = (r["swarm_p"] - r["resolution"]) ** 2
            market_brier = (r["market_price"] - r["resolution"]) ** 2
            eligible += 1
            if swarm_brier < market_brier:
                wins += 1
    rate = wins / eligible if eligible > 0 else 0.0
    return wins, eligible, rate


def compute_sharpness(records: list[dict]) -> float:
    """Mean absolute deviation from 0.5 — more extreme predictions = sharper."""
    return statistics.mean([abs(r["swarm_p"] - 0.5) for r in records])


# ── ASCII calibration plot ────────────────────────────────────────────────────

def ascii_calibration_plot(curve: list[dict]) -> str:
    """Render an ASCII art calibration plot."""
    height = 20  # rows for y-axis (0.0 to 1.0)
    width = 10   # one column per bucket

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height + 1)]

    # Place actual resolution rate markers ('o') and perfect calibration ('*')
    for col, bucket in enumerate(curve):
        # Perfect calibration point: midpoint of bucket
        lo = col / 10
        perfect_y = lo + 0.05
        perfect_row = height - int(round(perfect_y * height))
        perfect_row = max(0, min(height, perfect_row))
        grid[perfect_row][col] = "*"

        # Actual resolution rate
        if bucket["actual_rate"] is not None and bucket["count"] > 0:
            actual_row = height - int(round(bucket["actual_rate"] * height))
            actual_row = max(0, min(height, actual_row))
            if grid[actual_row][col] == "*":
                grid[actual_row][col] = "@"  # overlap
            else:
                grid[actual_row][col] = "o"

    lines = []
    lines.append("Calibration Plot (10 buckets)")
    for row in range(height + 1):
        y_val = (height - row) / height
        if row % 4 == 0:
            label = f"{y_val:.1f} |"
        else:
            label = "     |"
        row_str = "  ".join(grid[row][col] for col in range(width))
        lines.append(f"  {label} {row_str}")

    lines.append("       +" + "----+" * 10)
    lines.append("       0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0")
    lines.append("       o = actual resolution rate   * = perfect calibration")
    lines.append("       @ = actual matches perfect (overlap)")
    return "\n".join(lines)


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(source: str) -> dict:
    """Generate the full calibration report."""
    records = load_records(source)

    if not records:
        print(f"No resolved predictions found for source='{source}'.")
        print("Hint: resolve predictions first, or try --source eval-dataset")
        return {}

    total_in_db = 0
    if source in ("db", "both"):
        db = Database()
        total_in_db = len(db.get_predictions(resolved_only=False, limit=100_000))
        db.close()

    predictions = [r["swarm_p"] for r in records]
    outcomes = [r["resolution"] for r in records]
    n_resolved = len(records)

    # ── Compute all metrics ──
    brier_overall, brier_by_category = compute_brier(records)
    ece = compute_ece(predictions, outcomes)
    curve = compute_calibration_curve(predictions, outcomes)
    wins, eligible, win_rate = compute_win_rate(records)
    sharpness = compute_sharpness(records)

    resolution_rate = None
    if source == "db" and total_in_db > 0:
        resolution_rate = n_resolved / total_in_db
    elif source == "eval-dataset":
        resolution_rate = 1.0  # eval dataset is all resolved
    elif source == "both" and total_in_db > 0:
        resolution_rate = n_resolved / max(total_in_db, n_resolved)

    # ── Print report ──
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  CALIBRATION REPORT")
    print(f"  Source: {source} | N={n_resolved} resolved predictions")
    print(f"  Generated: {datetime.utcnow().isoformat()}Z")
    print(sep)

    print(f"\n  Overall Metrics")
    print(f"  {'-' * 40}")
    print(f"  Brier Score (overall):   {brier_overall:.4f}")
    print(f"  ECE (10 buckets):        {ece:.4f}")
    print(f"  Win Rate vs Market:      {wins}/{eligible} ({win_rate * 100:.0f}%)")
    print(f"  Sharpness (MAD from .5): {sharpness:.4f}")
    if resolution_rate is not None:
        print(f"  Resolution Rate:         {resolution_rate * 100:.1f}%")

    # Brier by category
    print(f"\n  Brier Score by Category")
    print(f"  {'-' * 40}")
    print(f"  {'Category':<16} {'N':>5}  {'Brier':>8}")
    cat_counts: dict[str, int] = {}
    for r in records:
        cat = r["category"] or "unknown"
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, b in sorted(brier_by_category.items(), key=lambda x: x[1]):
        print(f"  {cat:<16} {cat_counts[cat]:>5}  {b:>8.4f}")

    # Calibration curve table
    print(f"\n  Calibration Curve (Predicted vs Actual)")
    print(f"  {'-' * 55}")
    print(f"  {'Bucket':<10} {'N':>5}  {'Predicted':>10}  {'Actual':>10}  {'Gap':>8}")
    for b in curve:
        if b["actual_rate"] is not None:
            print(f"  {b['bucket']:<10} {b['count']:>5}  {b['mean_predicted']:>10.4f}"
                  f"  {b['actual_rate']:>10.4f}  {b['gap']:>+8.4f}")
        else:
            print(f"  {b['bucket']:<10} {b['count']:>5}  {'---':>10}  {'---':>10}  {'---':>8}")

    # ASCII plot
    print()
    print(ascii_calibration_plot(curve))

    # One-line summary for README
    summary_line = (
        f"Brier: {brier_overall:.3f} | ECE: {ece:.3f} | "
        f"Win Rate: {win_rate * 100:.0f}% | N={n_resolved}"
    )
    print(f"\n  README summary: {summary_line}")
    print()

    # ── Build JSON report ──
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "n_resolved": n_resolved,
        "resolution_rate": round(resolution_rate, 4) if resolution_rate is not None else None,
        "brier_overall": round(brier_overall, 4),
        "brier_by_category": {k: round(v, 4) for k, v in brier_by_category.items()},
        "ece": round(ece, 4),
        "win_rate": round(win_rate, 4),
        "wins": wins,
        "eligible_for_win_rate": eligible,
        "sharpness": round(sharpness, 4),
        "calibration_curve": curve,
        "summary_line": summary_line,
    }

    # Save JSON report
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "calibration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to {report_path}")

    # Generate matplotlib calibration plot
    if HAS_MPL:
        _generate_calibration_plot(curve, brier_overall, ece, n_resolved)
    else:
        print("  matplotlib not installed — skipping PNG plot")

    return report


def _generate_calibration_plot(curve: list[dict], brier: float, ece: float, n: int):
    """Save a matplotlib calibration plot to results/calibration_plot.png."""
    predicted = [b["mean_predicted"] for b in curve if b["actual_rate"] is not None and b["count"] > 0]
    actual = [b["actual_rate"] for b in curve if b["actual_rate"] is not None and b["count"] > 0]
    counts = [b["count"] for b in curve if b["actual_rate"] is not None and b["count"] > 0]

    if not predicted:
        print("  No data for plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(predicted, actual, s=[c * 80 for c in counts], alpha=0.7, c="#2563eb", zorder=5)
    ax.plot(predicted, actual, "-o", color="#2563eb", linewidth=2, markersize=6, label="MiniSim")

    for p, a, c in zip(predicted, actual, counts):
        ax.annotate(f"n={c}", (p, a), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title(f"MiniSim Calibration (n={n}, Brier={brier:.3f}, ECE={ece:.3f})", fontsize=13)
    ax.legend(loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join("results", "calibration_plot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Calibration plot saved to {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive calibration report from resolved predictions."
    )
    parser.add_argument(
        "--source",
        choices=["db", "eval-dataset", "both"],
        default="db",
        help="Data source: 'db' (SQLite), 'eval-dataset' (544q results), or 'both'",
    )
    args = parser.parse_args()
    generate_report(args.source)
