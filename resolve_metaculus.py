#!/usr/bin/env python3
"""
Resolve Metaculus predictions — check which tournament questions have resolved
and update the database with actual outcomes + Brier scores.

Usage: python resolve_metaculus.py
"""
from __future__ import annotations

import json
import os

import requests

from src.database import Database

BOT_TOKEN = os.environ.get("METACULUS_BOT_TOKEN", "")
BASE_URL = "https://www.metaculus.com/api"


def check_resolutions():
    if not BOT_TOKEN:
        print("METACULUS_BOT_TOKEN not set")
        return

    # Load our forecasted question IDs
    try:
        with open("results/forecasted_questions.json") as f:
            qids = json.load(f)
    except FileNotFoundError:
        print("No forecasted_questions.json found")
        return

    print(f"Checking {len(qids)} forecasted questions for resolutions...")

    headers = {"Authorization": f"Token {BOT_TOKEN}"}
    db = Database()

    resolved_count = 0
    still_open = 0
    errors = 0

    for qid in qids:
        try:
            # Fetch the post that contains this question
            resp = requests.get(
                f"{BASE_URL}/posts/",
                params={"has_question_id": qid, "limit": 1},
                headers=headers,
                timeout=15,
            )

            if resp.status_code != 200:
                # Try fetching by searching our DB for the question text
                errors += 1
                continue

            data = resp.json()
            results = data.get("results", [])
            if not results:
                errors += 1
                continue

            post = results[0]
            q = post.get("question", {})
            status = q.get("status", "")
            resolution = q.get("resolution")
            title = post.get("title", "")[:60]

            if status == "resolved" and resolution is not None:
                # Map resolution to 0/1
                if resolution == "yes" or resolution == 1.0 or resolution is True:
                    res_val = 1.0
                elif resolution == "no" or resolution == 0.0 or resolution is False:
                    res_val = 0.0
                else:
                    print(f"  Q{qid}: Unknown resolution '{resolution}' | {title}")
                    continue

                # Find this prediction in our DB and resolve it
                preds = db.get_predictions()
                matched = False
                for p in preds:
                    if p.get("ticker") == str(qid) and p.get("resolution") is None:
                        db.resolve(p["id"], res_val)
                        matched = True
                        print(f"  RESOLVED Q{qid}: {res_val} | {title}")
                        resolved_count += 1
                        break

                if not matched:
                    print(f"  Q{qid}: resolved={res_val} but not found in DB | {title}")
            else:
                still_open += 1

        except Exception as e:
            errors += 1
            continue

        import time
        time.sleep(0.5)  # rate limit

    db.close()

    print(f"\nResults:")
    print(f"  Resolved: {resolved_count}")
    print(f"  Still open: {still_open}")
    print(f"  Errors: {errors}")

    if resolved_count > 0:
        # Print updated metrics
        db = Database()
        m = db.get_metrics()
        print(f"\nUpdated metrics:")
        print(f"  Total predictions: {m['total']}")
        print(f"  Resolved: {m['resolved']}")
        if m.get("swarm_brier"):
            print(f"  Swarm Brier: {m['swarm_brier']}")
        if m.get("win_rate"):
            print(f"  Win rate vs market: {m['win_rate']}")
        db.close()


if __name__ == "__main__":
    check_resolutions()
