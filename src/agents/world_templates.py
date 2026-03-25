"""
World-building templates — category detection, world construction, pressures,
timelines, relationships, reasoning generation, evidence, and insight banks.

Extracted from offline_engine.py for modularity.
"""
from __future__ import annotations

import os
import random
import time

import yaml

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_yaml(name: str):
    with open(os.path.join(_DATA_DIR, name)) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Question category detection
# ---------------------------------------------------------------------------

def _detect_category(question: str) -> str:
    """Heuristically classify a question into econ / political / tech."""
    q = question.lower()
    # Use word boundary matching for short keywords to avoid substring false positives
    # e.g., "ai" matching "rain", "gene" matching "general"
    import re

    econ_kw = ["fed", "rate", "inflation", "gdp", "recession", "unemployment", "interest", "economy", "monetary", "fiscal", "stock", "market crash", "s&p", "dow", "treasury"]
    pol_kw = ["election", "president", "congress", "government", "shutdown", "impeach", "vote", "legislation", "bill", "senate", "supreme court", "governor", "partisan", "democrat", "republican", "executive order"]
    tech_kw = ["\\bai\\b", "artificial intelligence", "replace", "automat", "robot", "quantum", "spacex", "launch", "biotech", "\\bgene\\b", "crispr", "autonomous", "self-driving", "chip", "semiconductor", "\\bagi\\b", "openai", "google", "\\bapple\\b", "tesla"]

    def _count(keywords, text):
        count = 0
        for kw in keywords:
            if kw.startswith("\\b"):
                if re.search(kw, text):
                    count += 1
            elif kw in text:
                count += 1
        return count

    scores = {
        "econ": _count(econ_kw, q),
        "political": _count(pol_kw, q),
        "tech": _count(tech_kw, q),
    }
    if max(scores.values()) == 0:
        return "econ"  # default
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# World builder (offline)
# ---------------------------------------------------------------------------

_WORLD_TEMPLATES = _load_yaml("world_templates.yaml")


def build_world_offline(question: str, context: str = "") -> dict:
    """Build a world model without API calls."""
    start = time.time()
    category = _detect_category(question)
    template = _WORLD_TEMPLATES[category]

    # Generate pressures based on question
    pressures = _generate_pressures(question, category)
    timeline = _generate_timeline(question, category)
    relationships = _generate_relationships(template["entities"])

    world = {
        "entities": template["entities"],
        "relationships": relationships,
        "pressures": pressures,
        "timeline": timeline,
        "base_rate_estimate": 0.40,
        "key_uncertainties": template["key_uncertainties"],
        "question_category": category,
    }
    world["_build_time_ms"] = int((time.time() - start) * 1000)
    return world


_PRESSURE_BANK = _load_yaml("pressures.yaml")


def _generate_pressures(question: str, category: str) -> dict:
    return _PRESSURE_BANK.get(category, _PRESSURE_BANK["econ"])


_TIMELINE_BANK = _load_yaml("timelines.yaml")


def _generate_timeline(question: str, category: str) -> list:
    return _TIMELINE_BANK.get(category, _TIMELINE_BANK["econ"])


def _generate_relationships(entities: list) -> list:
    rels = []
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1 : i + 3]:
            rels.append({
                "source": e1["name"],
                "target": e2["name"],
                "relation": "influences",
                "strength": "strong" if e1["relevance"] == "high" and e2["relevance"] == "high" else "moderate",
            })
    return rels


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------

# Reasoning templates keyed by category
_REASONING_TEMPLATES = _load_yaml("reasoning_templates.yaml")

_FACTORS = _load_yaml("factors.yaml")


def _generate_reasoning(background_label: str, score: float, category: str, rng: random.Random) -> tuple[str, list[str]]:
    """Generate reasoning text and key factors for an agent."""
    templates = _REASONING_TEMPLATES[category]
    template = rng.choice(templates)
    fdata = _FACTORS[category]

    f1, f2 = rng.sample(fdata["factors"], 2)
    unc = rng.choice(fdata["uncertainties"])

    direction = "YES" if score > 0.5 else "NO"
    likelihood = "likely" if abs(score - 0.5) > 0.15 else "a toss-up" if abs(score - 0.5) < 0.05 else "moderately probable"
    likelihood_adv = "rapidly" if score > 0.6 else "gradually" if score > 0.4 else "slowly"

    reasoning = template.format(
        background=background_label,
        factor1=f1,
        factor2=f2,
        uncertainty=unc,
        score=score,
        direction=direction,
        likelihood=likelihood,
        likelihood_adv=likelihood_adv,
    )
    key_factors = [f1, f2, unc]
    return reasoning, key_factors


# ---------------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------------

_EVIDENCE_TEMPLATES = _load_yaml("evidence_templates.yaml")


def _generate_evidence(category: str, agent: dict, rng: random.Random) -> list[dict]:
    """Generate 3 evidence items for an agent, reflecting their position."""
    templates = _EVIDENCE_TEMPLATES.get(category, _EVIDENCE_TEMPLATES["econ"])
    score = agent["score_history"][-1]
    evidence = []

    for tmpl in rng.sample(templates, min(3, len(templates))):
        # Evidence values tilted by agent's position
        base_val = 40 + rng.gauss(0, 15)
        if score > 0.5:
            base_val += 10  # YES-leaning agents find YES-supporting data
        else:
            base_val -= 10

        item = {
            "claim": tmpl.format(
                val=base_val,
                direction="above" if base_val > 50 else "below",
                trend="rising" if score > 0.5 else "falling",
                zone="expansion" if base_val > 50 else "contraction",
                signal="steepening (easing signal)" if score > 0.5 else "inverting (caution signal)",
                party="the opposition" if rng.random() > 0.5 else "the majority",
            ),
            "quality": round(rng.uniform(2.5, 5.0), 1),
            "source_type": rng.choice(["official data", "survey", "market data", "expert analysis", "historical analog"]),
        }
        evidence.append(item)
    return evidence


# ---------------------------------------------------------------------------
# Insight bank
# ---------------------------------------------------------------------------

_INSIGHT_BANK = _load_yaml("insights.yaml")
