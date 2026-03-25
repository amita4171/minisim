"""
Shared utilities used across multiple modules.
Extracted to eliminate duplication.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Anthropic client (lazy-loaded, thread-safe singleton) ──

_anthropic_client = None


def get_anthropic_client():
    """Get or create a shared Anthropic client instance.

    Lazy-loaded to avoid import-time crashes when credentials aren't available.
    Thread-safe: the Anthropic SDK client supports connection pooling.
    """
    global _anthropic_client
    if _anthropic_client is None:
        try:
            from anthropic import Anthropic
            _anthropic_client = Anthropic()
        except Exception as e:
            logger.debug(f"Anthropic client init failed: {e}")
            raise
    return _anthropic_client


# ── Safe numeric parsing ──

def safe_float(val) -> float | None:
    """Parse a value to float, returning None on failure.

    Handles: str, int, float, None, empty string.
    Used by Kalshi, Polymarket, and other API client parsers.
    """
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
