"""
Metaculus API client — fetches questions and predictions.
Requires API token (set METACULUS_TOKEN env var or pass directly).

API: https://www.metaculus.com/api/posts/
Auth: Authorization: Token <token>

Note: Resolution values and community predictions require Bot Benchmarking
access tier. Email api-requests@metaculus.com to request access.
Without it, we get: titles, forecaster counts, dates, status.
"""
from __future__ import annotations

import os
import time

import requests

BASE_URL = "https://www.metaculus.com/api"
DEFAULT_TIMEOUT = 15


def _get_token() -> str:
    return os.environ.get("METACULUS_TOKEN", "")


def get_posts(
    statuses: str = "resolved",
    forecast_type: str = "binary",
    order_by: str = "-forecasts_count",
    limit: int = 50,
    token: str | None = None,
    cursor: str | None = None,
) -> dict:
    """Fetch posts from Metaculus API."""
    token = token or _get_token()
    if not token:
        raise ValueError("METACULUS_TOKEN not set. Get yours at metaculus.com/accounts/settings/")

    params = {
        "limit": limit,
        "statuses": statuses,
        "forecast_type": forecast_type,
        "order_by": order_by,
    }
    if cursor:
        params["cursor"] = cursor

    resp = requests.get(
        f"{BASE_URL}/posts/",
        params=params,
        headers={"Authorization": f"Token {token}"},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_post(post_id: int, token: str | None = None) -> dict:
    """Fetch a single post by ID."""
    token = token or _get_token()
    resp = requests.get(
        f"{BASE_URL}/posts/{post_id}/",
        headers={"Authorization": f"Token {token}"},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_resolved_questions(
    limit: int = 200,
    max_pages: int = 10,
    min_forecasters: int = 50,
    token: str | None = None,
) -> list[dict]:
    """Fetch resolved binary questions, sorted by most forecasters.

    Returns parsed question dicts with available fields.
    """
    token = token or _get_token()
    all_questions = []
    next_url = None

    for page in range(max_pages):
        if page == 0:
            data = get_posts(
                statuses="resolved",
                forecast_type="binary",
                order_by="-forecasts_count",
                limit=min(limit, 100),
                token=token,
            )
        else:
            if not next_url:
                break
            resp = requests.get(
                next_url,
                headers={"Authorization": f"Token {token}"},
                timeout=DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        next_url = data.get("next")

        for post in results:
            q = post.get("question", {})
            if not q or q.get("type") != "binary":
                continue

            n_forecasters = post.get("nr_forecasters", 0)
            if n_forecasters < min_forecasters:
                continue

            parsed = {
                "id": q.get("id"),
                "post_id": post.get("id"),
                "title": post.get("title", ""),
                "question": post.get("title", ""),
                "status": q.get("status", ""),
                "resolution": q.get("resolution"),  # None without bot benchmarking tier
                "n_forecasters": n_forecasters,
                "forecasts_count": post.get("forecasts_count", 0),
                "created_at": q.get("created_at"),
                "open_time": q.get("open_time"),
                "scheduled_resolve_time": q.get("scheduled_resolve_time"),
                "actual_resolve_time": q.get("actual_resolve_time"),
                "community_prediction": _extract_community_prediction(q),
                "source": "metaculus",
            }
            all_questions.append(parsed)

        if len(all_questions) >= limit or not next_url:
            break

        time.sleep(0.5)

    return all_questions[:limit]


def get_open_questions(
    limit: int = 50,
    min_forecasters: int = 20,
    token: str | None = None,
) -> list[dict]:
    """Fetch open binary questions for live predictions."""
    token = token or _get_token()
    data = get_posts(
        statuses="open",
        forecast_type="binary",
        order_by="-forecasts_count",
        limit=min(limit, 100),
        token=token,
    )

    questions = []
    for post in data.get("results", []):
        q = post.get("question", {})
        if not q or q.get("type") != "binary":
            continue
        if post.get("nr_forecasters", 0) < min_forecasters:
            continue

        questions.append({
            "id": q.get("id"),
            "post_id": post.get("id"),
            "title": post.get("title", ""),
            "question": post.get("title", ""),
            "n_forecasters": post.get("nr_forecasters", 0),
            "community_prediction": _extract_community_prediction(q),
            "open_time": q.get("open_time"),
            "scheduled_resolve_time": q.get("scheduled_resolve_time"),
            "source": "metaculus",
        })

    return questions


def _extract_community_prediction(q: dict) -> float | None:
    """Extract community prediction if available (requires bot benchmarking tier)."""
    agg = q.get("aggregations", {})
    for method in ["recency_weighted", "unweighted", "metaculus_prediction"]:
        latest = agg.get(method, {}).get("latest")
        if latest and isinstance(latest, dict):
            centers = latest.get("centers")
            if centers and len(centers) > 0:
                return centers[0]
    return None
