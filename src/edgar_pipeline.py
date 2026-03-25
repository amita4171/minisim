"""
SEC EDGAR Earnings Call Pipeline — Extract forward-looking statements
and convert them to resolvable predictions with outcomes.

Strategy (resolution-first):
1. Start from known outcomes (quarterly earnings, FOMC decisions, product launches)
2. Search backward into EDGAR transcripts for forward-looking statements
3. Match statement to outcome = training example with ground truth

Source: SEC EDGAR FULL-TEXT SEARCH API (free, no auth, rate limit: 10 req/sec)
Docs: https://efts.sec.gov/LATEST/search-index?q=...

Usage:
    from src.edgar_pipeline import search_filings, extract_guidance
    filings = search_filings("Apple", form_type="8-K", date_range=("2024-01-01", "2024-12-31"))
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_FULLTEXT_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC requires a User-Agent with contact info
USER_AGENT = os.environ.get(
    "EDGAR_USER_AGENT",
    "MiniSim Research Bot (minisim@example.com)"
)
HEADERS = {"User-Agent": USER_AGENT}


def search_filings(
    company: str,
    form_type: str = "8-K",
    date_start: str | None = None,
    date_end: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Search SEC EDGAR for company filings.

    Args:
        company: Company name or ticker (e.g., "Apple" or "AAPL")
        form_type: SEC form type — "8-K" (current events), "10-K" (annual), "10-Q" (quarterly)
        date_start: "YYYY-MM-DD" start date
        date_end: "YYYY-MM-DD" end date
        max_results: maximum filings to return
    """
    params = {
        "q": company,
        "dateRange": "custom",
        "startdt": date_start or "2023-01-01",
        "enddt": date_end or datetime.now().strftime("%Y-%m-%d"),
        "forms": form_type,
    }

    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"EDGAR search failed: {resp.status_code}")
            return []

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        filings = []
        for hit in hits[:max_results]:
            source = hit.get("_source", {})
            filings.append({
                "company": source.get("display_names", [""])[0] if source.get("display_names") else company,
                "form_type": source.get("form_type", form_type),
                "filed_date": source.get("file_date", ""),
                "description": source.get("display_date_filed", ""),
                "url": f"https://www.sec.gov/Archives/edgar/data/{source.get('entity_id', '')}/{source.get('file_num', '')}",
                "entity_id": source.get("entity_id", ""),
            })

        return filings

    except Exception as e:
        logger.warning(f"EDGAR search error: {e}")
        return []


def search_fulltext(
    query: str,
    date_start: str = "2023-01-01",
    date_end: str | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Full-text search across all SEC filings.

    Great for finding forward-looking statements mentioning specific topics.
    E.g., search for "revenue guidance" or "expects to achieve" or "rate cut"
    """
    params = {
        "q": f'"{query}"',
        "dateRange": "custom",
        "startdt": date_start,
        "enddt": date_end or datetime.now().strftime("%Y-%m-%d"),
    }

    try:
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        results = []
        for hit in hits[:max_results]:
            source = hit.get("_source", {})
            results.append({
                "company": source.get("display_names", ["Unknown"])[0] if source.get("display_names") else "Unknown",
                "form_type": source.get("form_type", ""),
                "filed_date": source.get("file_date", ""),
                "snippet": hit.get("highlight", {}).get("file_description", [""])[0] if hit.get("highlight") else "",
                "entity_id": source.get("entity_id", ""),
            })

        return results

    except Exception as e:
        logger.warning(f"EDGAR fulltext search error: {e}")
        return []


# ── S&P 500 companies for batch processing ──

SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC",
    "XOM", "PFE", "KO", "PEP", "CSCO", "ABT", "CRM", "NFLX", "ADBE",
    "AMD", "INTC", "QCOM",
]


def build_guidance_dataset(
    companies: list[str] | None = None,
    date_start: str = "2023-01-01",
    date_end: str = "2025-12-31",
    output_path: str = "results/edgar_guidance.json",
) -> list[dict]:
    """Build a dataset of forward-looking guidance from earnings filings.

    Searches for revenue guidance, earnings projections, and forward-looking
    statements from S&P 500 company filings.
    """
    if companies is None:
        companies = SP500_SAMPLE[:10]  # start with top 10

    guidance_terms = [
        "revenue guidance",
        "expects revenue",
        "projects earnings",
        "forward looking",
        "fiscal year outlook",
    ]

    all_results = []

    for company in companies:
        print(f"  Searching {company}...")
        for term in guidance_terms[:2]:  # limit queries per company
            results = search_fulltext(
                f"{company} {term}",
                date_start=date_start,
                date_end=date_end,
                max_results=5,
            )
            for r in results:
                r["search_term"] = term
                r["ticker"] = company
                all_results.append(r)

            time.sleep(0.2)  # SEC rate limit: 10 req/sec

    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        key = f"{r['company']}_{r['filed_date']}_{r['form_type']}"
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"count": len(unique), "filings": unique}, f, indent=2)

    print(f"  Found {len(unique)} unique filings with guidance language")
    return unique


if __name__ == "__main__":
    print("Building EDGAR guidance dataset (top 10 S&P 500)...")
    results = build_guidance_dataset()
    print(f"Done: {len(results)} filings saved to results/edgar_guidance.json")
