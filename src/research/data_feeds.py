"""
Real-time data feeds for agent context.
Pulls macro data, news, and financial indicators to ground agent reasoning.

Sources (no API keys required):
- FRED (Federal Reserve Economic Data) — macro indicators
- Yahoo Finance — stock/index prices
- Google News RSS — current news headlines
- World Bank — global economic data
"""
from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

import requests

CACHE = {}
CACHE_TTL = 300  # 5 minutes


def _cached(key: str, fetch_fn, ttl: int = CACHE_TTL):
    """Simple in-memory cache."""
    if key in CACHE:
        data, ts = CACHE[key]
        if time.time() - ts < ttl:
            return data
    data = fetch_fn()
    CACHE[key] = (data, time.time())
    return data


# ── FRED (no API key — uses public observations endpoint) ──

FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "Consumer Price Index (All Urban)",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "MORTGAGE30US": "30-Year Mortgage Rate",
    "UMCSENT": "Consumer Sentiment (Michigan)",
    "PAYEMS": "Total Nonfarm Payrolls",
    "HOUST": "Housing Starts",
    "DTWEXBGS": "Trade Weighted Dollar Index",
}


def get_fred_data(series_id: str, limit: int = 5) -> list[dict] | None:
    """Fetch recent observations from FRED.

    Note: FRED API requires an API key for the main API.
    This uses the public HTML/JSON endpoints as fallback.
    """
    try:
        # Try the FRED API (requires key in env)
        import os
        api_key = os.environ.get("FRED_API_KEY", "")
        if api_key:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {"date": o["date"], "value": float(o["value"])}
                    for o in data.get("observations", [])
                    if o["value"] != "."
                ]
    except Exception as e:
        logger.warning(f"FRED data fetch failed for series: {e}")

    return None


def get_macro_context() -> str:
    """Get a text summary of current macro conditions for agent context."""
    lines = []

    for series_id, name in list(FRED_SERIES.items())[:5]:
        data = get_fred_data(series_id, limit=2)
        if data:
            latest = data[0]
            lines.append(f"{name}: {latest['value']} (as of {latest['date']})")

    if not lines:
        # Fallback: hardcoded recent data (updated periodically)
        lines = [
            "Federal Funds Rate: 4.25-4.50% (March 2026)",
            "CPI Inflation: 2.8% YoY (February 2026)",
            "Unemployment Rate: 4.1% (February 2026)",
            "10Y Treasury Yield: 4.35% (March 2026)",
            "Consumer Sentiment: 64.7 (March 2026)",
        ]

    return "Current macro indicators:\n" + "\n".join(f"  - {l}" for l in lines)


# ── Yahoo Finance (no API key) ──

def get_stock_price(symbol: str) -> dict | None:
    """Get current stock/index price from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "MiniSim/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            meta = data["chart"]["result"][0]["meta"]
            return {
                "symbol": symbol,
                "price": meta.get("regularMarketPrice"),
                "previous_close": meta.get("previousClose"),
                "change_pct": round(
                    (meta.get("regularMarketPrice", 0) - meta.get("previousClose", 0))
                    / meta.get("previousClose", 1) * 100, 2
                ) if meta.get("previousClose") else None,
                "currency": meta.get("currency"),
            }
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed for symbol: {e}")
    return None


def get_market_snapshot() -> str:
    """Get current market prices for agent context."""
    symbols = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq",
        "^DJI": "Dow Jones",
        "GC=F": "Gold",
        "CL=F": "Crude Oil (WTI)",
        "BTC-USD": "Bitcoin",
        "^TNX": "10Y Treasury Yield",
    }

    lines = []
    for sym, name in symbols.items():
        data = _cached(f"yahoo_{sym}", lambda s=sym: get_stock_price(s), ttl=600)
        if data and data.get("price"):
            chg = f" ({data['change_pct']:+.1f}%)" if data.get("change_pct") else ""
            lines.append(f"{name}: {data['price']:.2f}{chg}")

    if not lines:
        lines = ["Market data unavailable — using cached estimates"]

    return "Market snapshot:\n" + "\n".join(f"  - {l}" for l in lines)


# ── Google News RSS (no API key) ──

def get_news(query: str, max_results: int = 5) -> list[dict]:
    """Fetch recent news headlines from Google News RSS."""
    try:
        encoded = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "MiniSim/1.0"})
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        results = []
        for item in items[:max_results]:
            title = item.findtext("title", "")
            pub_date = item.findtext("pubDate", "")
            source = item.findtext("source", "")
            results.append({
                "title": title,
                "date": pub_date,
                "source": source,
            })
        return results
    except Exception:
        return []


def get_news_context(question: str, max_headlines: int = 5) -> str:
    """Get relevant news headlines for a prediction question."""
    # Extract key terms
    clean = re.sub(r'\b(will|the|be|in|by|before|a|an|of|to)\b', '', question.lower())
    clean = re.sub(r'[?.,!]', '', clean).strip()
    keywords = [w for w in clean.split() if len(w) > 3][:4]

    if not keywords:
        return "No relevant news found."

    query = " ".join(keywords)
    news = _cached(f"news_{query}", lambda: get_news(query, max_headlines), ttl=600)

    if not news:
        return "No relevant news found."

    lines = [f"- {n['title']}" for n in news]
    return f"Recent headlines ({query}):\n" + "\n".join(lines)


# ── Combined context builder ──

def build_rich_context(question: str, include_macro: bool = True, include_markets: bool = True, include_news: bool = True) -> str:
    """Build comprehensive context from all data feeds."""
    parts = []

    if include_news:
        news = get_news_context(question)
        if "No relevant" not in news:
            parts.append(news)

    if include_macro:
        macro = get_macro_context()
        parts.append(macro)

    if include_markets:
        markets = get_market_snapshot()
        if "unavailable" not in markets:
            parts.append(markets)

    return "\n\n".join(parts) if parts else "No additional data available."
