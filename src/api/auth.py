"""
API key authentication and rate limiting for the MiniSim API.

Rate limiting uses a simple in-memory sliding window counter.
- Free tier: 100 requests/hour
- Pro tier: 2000 requests/hour
- Tier determined by API key prefix: keys starting with "pro-" get pro tier
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)

# ── API key management ──

_raw_keys = os.environ.get("MINISIM_API_KEYS", "")
if not _raw_keys:
    import secrets as _secrets
    _generated = _secrets.token_urlsafe(32)
    logger.warning(f"MINISIM_API_KEYS not set -- generated ephemeral key: {_generated}")
    _raw_keys = _generated
API_KEYS: set[str] = set(_raw_keys.split(","))


def verify_api_key(authorization: str = Header(None)) -> str:
    """Validate Bearer token against known API keys. Returns the token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token


# ── Rate limiting (sliding window, in-memory) ──

# Tier limits (requests per hour)
_TIER_LIMITS = {
    "free": 100,
    "pro": 2000,
}

# Window duration in seconds
_WINDOW_SECONDS = 3600

# Storage: api_key -> list of request timestamps
_request_timestamps: dict[str, list[float]] = defaultdict(list)


def _get_tier(api_key: str) -> str:
    """Determine rate-limit tier from API key prefix."""
    if api_key.startswith("pro-"):
        return "pro"
    return "free"


def _cleanup_window(timestamps: list[float], now: float) -> list[float]:
    """Remove timestamps outside the sliding window."""
    cutoff = now - _WINDOW_SECONDS
    # Binary search would be optimal but list comprehension is fine for expected load
    return [ts for ts in timestamps if ts > cutoff]


def check_rate_limit(api_key: str) -> Optional[dict]:
    """
    Check rate limit for the given API key.

    Returns None if within limits, or a dict with error info if exceeded:
        {"retry_after": <seconds_until_oldest_expires>}
    """
    now = time.time()
    tier = _get_tier(api_key)
    limit = _TIER_LIMITS[tier]

    # Clean up expired timestamps
    _request_timestamps[api_key] = _cleanup_window(_request_timestamps[api_key], now)
    timestamps = _request_timestamps[api_key]

    if len(timestamps) >= limit:
        # Calculate when the oldest request in window will expire
        retry_after = int(timestamps[0] - (now - _WINDOW_SECONDS)) + 1
        return {"retry_after": max(retry_after, 1)}

    # Record this request
    timestamps.append(now)
    return None


def rate_limit_dependency(authorization: str = Header(None)) -> str:
    """
    Combined auth + rate limit dependency.
    First verifies the API key, then checks rate limits.
    Returns the verified API key.
    """
    token = verify_api_key(authorization)

    exceeded = check_rate_limit(token)
    if exceeded:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(exceeded["retry_after"])},
        )

    return token
