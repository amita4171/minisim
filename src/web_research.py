"""
Web research module for MiniSim agents.
Fetches real-time information to ground agent reasoning in current facts.

Uses multiple search strategies to provide information asymmetry across agents
(different agents get different search results per arxiv 2510.01171).

This module is designed to be called from the CLI or Streamlit — it uses
requests to fetch public web pages and extracts relevant text.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Optional

import requests

# DuckDuckGo Instant Answer API (no auth needed)
DDG_API = "https://api.duckduckgo.com/"

# News API alternatives (no auth)
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information relevant to a prediction question.

    Uses DuckDuckGo Instant Answer API (free, no auth).
    Returns list of {title, snippet, url} dicts.
    """
    results = []

    # DuckDuckGo Instant Answer
    try:
        resp = requests.get(
            DDG_API,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()

            # Abstract
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", ""),
                    "snippet": data["Abstract"][:500],
                    "url": data.get("AbstractURL", ""),
                    "source": "duckduckgo_abstract",
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", "")[:300],
                        "url": topic.get("FirstURL", ""),
                        "source": "duckduckgo_related",
                    })
    except Exception:
        pass

    return results[:max_results]


def research_question(question: str, n_perspectives: int = 3) -> list[dict]:
    """Research a prediction question from multiple angles.

    Returns a list of research bundles — one per perspective.
    Each bundle has a different search focus to create information asymmetry.

    Args:
        question: The prediction question
        n_perspectives: Number of different research angles

    Returns:
        List of research bundles, each with:
        - perspective: str (e.g., "bull_case", "bear_case", "historical")
        - search_results: list of {title, snippet, url}
        - summary: str
    """
    # Generate different search queries for information asymmetry
    perspectives = _generate_perspectives(question)[:n_perspectives]

    bundles = []
    for persp in perspectives:
        results = search_web(persp["query"], max_results=3)

        bundle = {
            "perspective": persp["label"],
            "query": persp["query"],
            "search_results": results,
            "summary": _summarize_results(results, question),
        }
        bundles.append(bundle)

        # Rate limiting
        time.sleep(0.5)

    return bundles


def _generate_perspectives(question: str) -> list[dict]:
    """Generate different search perspectives for a prediction question."""
    q = question.lower()

    # Core query
    perspectives = [
        {"label": "direct", "query": question},
    ]

    # Extract key entities/topics
    # Remove question words and common filler
    clean = re.sub(r'\b(will|the|be|in|by|before|after|than|more|less|any|a|an)\b', '', q)
    clean = re.sub(r'[?.,!]', '', clean).strip()
    keywords = [w for w in clean.split() if len(w) > 3][:5]

    if keywords:
        # Bull case
        perspectives.append({
            "label": "bull_case",
            "query": f"{' '.join(keywords[:3])} likely why evidence",
        })

        # Bear case / skeptical
        perspectives.append({
            "label": "bear_case",
            "query": f"{' '.join(keywords[:3])} unlikely risks obstacles",
        })

        # Historical precedent
        perspectives.append({
            "label": "historical",
            "query": f"{' '.join(keywords[:3])} historical precedent similar past",
        })

        # Expert analysis
        perspectives.append({
            "label": "expert",
            "query": f"{' '.join(keywords[:3])} expert analysis forecast prediction",
        })

        # Recent news
        perspectives.append({
            "label": "recent_news",
            "query": f"{' '.join(keywords[:3])} latest news 2026",
        })

    return perspectives


def _summarize_results(results: list[dict], question: str) -> str:
    """Create a brief summary of search results relevant to the question."""
    if not results:
        return "No relevant information found."

    snippets = [r["snippet"] for r in results if r.get("snippet")]
    if not snippets:
        return "Search returned results but no text snippets."

    combined = " ".join(snippets)[:800]
    return f"Research findings: {combined}"


def assign_research_to_agents(
    agents: list[dict],
    research_bundles: list[dict],
) -> list[dict]:
    """Assign different research bundles to different agents for information asymmetry.

    Each agent gets a subset of the available research, creating diverse information
    bases (per arxiv 2510.01171 recommendation for ensemble diversity).
    """
    if not research_bundles:
        return agents

    n_bundles = len(research_bundles)

    for i, agent in enumerate(agents):
        # Each agent gets 1-2 bundles, rotated for diversity
        primary = research_bundles[i % n_bundles]
        secondary = research_bundles[(i + 1) % n_bundles] if n_bundles > 1 else None

        agent_research = [primary]
        if secondary and secondary["perspective"] != primary["perspective"]:
            agent_research.append(secondary)

        # Contrarian agents get the bear case if available
        if agent.get("_contrarian_factor", 0) > 0.1:
            bear = next((b for b in research_bundles if b["perspective"] == "bear_case"), None)
            if bear and bear not in agent_research:
                agent_research = [bear] + agent_research[:1]

        # Store on agent
        agent["research"] = agent_research

        # Add to memory stream
        for bundle in agent_research:
            if bundle["summary"] != "No relevant information found.":
                agent["memory_stream"].append(
                    f"Research ({bundle['perspective']}): {bundle['summary'][:200]}"
                )

    return agents
