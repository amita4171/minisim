#!/usr/bin/env python3
"""
MiniSim Metaculus Tournament Bot — Competes in the Spring 2026 AIB.

Fetches open binary questions from the tournament, runs MiniSim prediction,
submits forecast + private comment with reasoning via API.

Questions are open for ~1.5hrs, so run this every 30min to catch them.

Usage:
  python metaculus_bot.py                          # one-time run
  python metaculus_bot.py --watch --interval 1800   # run every 30min
  python metaculus_bot.py --tournament minibench    # target MiniBench
  python metaculus_bot.py --dry-run                 # preview without submitting
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

import requests

BOT_TOKEN = os.environ.get("METACULUS_BOT_TOKEN", "")
if not BOT_TOKEN:
    logger.warning("METACULUS_BOT_TOKEN not set. Set via environment variable.")
BASE_URL = "https://www.metaculus.com/api"
DEFAULT_TOURNAMENT = "spring-aib-2026"


def get_open_questions(tournament: str = DEFAULT_TOURNAMENT, question_type: str = "binary") -> list[dict]:
    """Fetch open questions from the tournament."""
    resp = requests.get(
        f"{BASE_URL}/posts/",
        params={
            "project": tournament,
            "statuses": "open",
            "forecast_type": question_type,
            "limit": 50,
            "order_by": "-published_at",
        },
        headers={"Authorization": f"Token {BOT_TOKEN}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def submit_forecast(question_id: int, probability: float) -> bool:
    """Submit a binary forecast to Metaculus."""
    probability = max(0.001, min(0.999, probability))

    resp = requests.post(
        f"{BASE_URL}/questions/forecast/",
        json=[{"question": question_id, "probability_yes": round(probability, 4)}],
        headers={
            "Authorization": f"Token {BOT_TOKEN}",
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    return resp.status_code == 200


def submit_comment(post_id: int, comment: str) -> bool:
    """Submit a private comment (reasoning) on a question."""
    resp = requests.post(
        f"{BASE_URL}/comments/create/",
        json={
            "on_post": post_id,
            "text": comment[:4000],
            "is_private": True,
        },
        headers={
            "Authorization": f"Token {BOT_TOKEN}",
            "Content-Type": "application/json",
        },
        timeout=15,
    )
    return resp.status_code in (200, 201)


def run_minisim_prediction(question: str, context: str = "", model: str | None = None) -> dict:
    """Run MiniSim prediction using the router (smart mode)."""
    from src.core.llm_engine import LLMEngine
    engine = LLMEngine(model=model)

    if engine.is_available():
        from src.core.router import routed_predict
        return routed_predict(
            question=question,
            context=context,
            n_agents=10,
            engine=engine,
            max_rounds=2,
        )
    else:
        from src.core.offline_engine import swarm_score_offline
        return swarm_score_offline(question, context, n_agents=15, rounds=2)


def format_reasoning(result: dict) -> str:
    """Format the prediction result as a Metaculus comment."""
    parts = [
        f"**MiniSim Swarm Prediction: P(YES) = {result['swarm_probability_yes']:.3f}**",
        f"95% CI: [{result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f}]",
        f"Diversity: {result.get('diversity_score', 0):.3f}",
    ]

    routing = result.get("routing", {})
    if routing:
        parts.append(f"Route: {routing.get('route', 'unknown')} (initial_std={routing.get('initial_std', 0):.3f})")

    if result.get("top_yes_voices"):
        top_yes = result["top_yes_voices"][0]
        parts.append(f"\n**Top YES voice:** {top_yes['name']} ({top_yes['background']}): {top_yes.get('reasoning', '')[:200]}")

    if result.get("top_no_voices"):
        top_no = result["top_no_voices"][0]
        parts.append(f"\n**Top NO voice:** {top_no['name']} ({top_no['background']}): {top_no.get('reasoning', '')[:200]}")

    if result.get("reasoning_shift_summary"):
        parts.append(f"\n{result['reasoning_shift_summary']}")

    return "\n".join(parts)


def run_bot(
    tournament: str = DEFAULT_TOURNAMENT,
    dry_run: bool = False,
    already_forecasted: set | None = None,
    model: str | None = None,
):
    """Run one cycle of the bot."""
    if already_forecasted is None:
        already_forecasted = set()

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'=' * 60}")
    print(f"MiniSim Bot — {now}")
    print(f"Tournament: {tournament}")
    print(f"{'=' * 60}")

    # Fetch open binary questions
    posts = get_open_questions(tournament, "binary")
    print(f"Open binary questions: {len(posts)}")

    new_questions = []
    for p in posts:
        q = p.get("question", {})
        qid = q.get("id")
        if qid and qid not in already_forecasted:
            new_questions.append(p)

    print(f"New (unforecasted): {len(new_questions)}")

    if not new_questions:
        print("No new questions to forecast.")
        return already_forecasted

    for p in new_questions:
        q = p.get("question", {})
        qid = q["id"]
        post_id = p["id"]
        title = p.get("title", "")

        # Skip moved/restated questions
        skip_kw = ["Moved to", "RESTATED", "moved to", "restated at"]
        if any(kw in title for kw in skip_kw):
            print(f"\n--- Q{qid}: SKIPPED (moved/restated) | {title[:55]} ---")
            already_forecasted.add(qid)
            continue

        print(f"\n--- Q{qid}: {title[:65]} ---")

        # Run prediction
        try:
            result = run_minisim_prediction(title, model=model)
            prob = result["swarm_probability_yes"]
            print(f"  Prediction: P(YES) = {prob:.3f}")
        except Exception as e:
            print(f"  Prediction failed: {e}")
            continue

        if dry_run:
            print(f"  [DRY RUN] Would submit P(YES) = {prob:.3f}")
            continue

        # Submit forecast
        success = submit_forecast(qid, prob)
        if success:
            print(f"  Forecast submitted: P(YES) = {prob:.3f}")
            already_forecasted.add(qid)
        else:
            print(f"  Forecast submission FAILED")
            continue

        # Submit reasoning comment
        reasoning = format_reasoning(result)
        comment_ok = submit_comment(post_id, reasoning)
        print(f"  Comment: {'submitted' if comment_ok else 'failed'}")

        # Log to track record
        try:
            from src.db.database import Database
            db = Database()
            db.log_prediction(
                question=title,
                swarm_probability=prob,
                source="metaculus",
                ticker=str(qid),
                category="tournament",
                n_agents=result.get("config", {}).get("n_agents", 0),
                mode=result.get("config", {}).get("mode", "unknown"),
                confidence_interval=result.get("confidence_interval"),
                diversity_score=result.get("diversity_score", 0),
            )
            db.close()
        except Exception as e:
            logger.warning(f"Failed to save Metaculus prediction to database: {e}")

        time.sleep(2)  # rate limit courtesy

    return already_forecasted


FORECASTED_CACHE = "results/forecasted_questions.json"


def _load_forecasted() -> set:
    """Load previously forecasted question IDs from disk."""
    try:
        with open(FORECASTED_CACHE) as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _save_forecasted(forecasted: set):
    """Save forecasted question IDs to disk."""
    os.makedirs(os.path.dirname(FORECASTED_CACHE) or ".", exist_ok=True)
    with open(FORECASTED_CACHE, "w") as f:
        json.dump(sorted(forecasted), f)


def main():
    parser = argparse.ArgumentParser(description="MiniSim Metaculus Tournament Bot")
    parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT,
                       help="Tournament slug (default: spring-aib-2026)")
    parser.add_argument("--watch", action="store_true", help="Continuous mode")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between runs (default: 1800 = 30min)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    parser.add_argument("--model", type=str, default=None, help="LLM model (e.g., qwen2.5:14b)")
    args = parser.parse_args()

    already_forecasted = _load_forecasted()
    print(f"Loaded {len(already_forecasted)} previously forecasted questions")

    while True:
        already_forecasted = run_bot(
            tournament=args.tournament,
            dry_run=args.dry_run,
            already_forecasted=already_forecasted,
            model=args.model,
        )

        _save_forecasted(already_forecasted)

        if not args.watch:
            break

        print(f"\nNext run in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
