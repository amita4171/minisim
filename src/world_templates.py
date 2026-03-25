"""
World-building templates — category detection, world construction, pressures,
timelines, relationships, reasoning generation, evidence, and insight banks.

Extracted from offline_engine.py for modularity.
"""
from __future__ import annotations

import random
import statistics
import time


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

_WORLD_TEMPLATES = {
    "econ": {
        "entities": [
            {"name": "Federal Reserve", "type": "org", "description": "US central bank setting monetary policy via the FOMC", "relevance": "high"},
            {"name": "Jerome Powell", "type": "person", "description": "Fed Chair, leads FOMC decisions and communication", "relevance": "high"},
            {"name": "US Labor Market", "type": "metric", "description": "Employment data including NFP, unemployment rate, wage growth", "relevance": "high"},
            {"name": "PCE Inflation Index", "type": "metric", "description": "Fed's preferred inflation measure, target 2%", "relevance": "high"},
            {"name": "US Treasury Market", "type": "metric", "description": "Yield curve dynamics signal rate expectations", "relevance": "medium"},
            {"name": "Consumer Spending", "type": "metric", "description": "Largest GDP component, reflects household confidence", "relevance": "medium"},
            {"name": "Global Trade Tensions", "type": "event", "description": "Tariffs and trade policy affecting economic outlook", "relevance": "medium"},
        ],
        "key_uncertainties": [
            "Whether inflation will sustainably reach 2% target",
            "Labor market resilience vs. softening signals",
            "Geopolitical shocks affecting supply chains and energy prices",
            "Consumer spending trajectory and savings rate trends",
            "Impact of fiscal policy on monetary policy independence",
        ],
    },
    "political": {
        "entities": [
            {"name": "US Congress", "type": "org", "description": "Legislative body — must pass appropriations bills", "relevance": "high"},
            {"name": "White House", "type": "org", "description": "Executive branch — signs or vetoes legislation", "relevance": "high"},
            {"name": "Federal Budget", "type": "policy", "description": "Annual appropriations and continuing resolutions", "relevance": "high"},
            {"name": "Partisan Dynamics", "type": "concept", "description": "Polarization level and willingness to compromise", "relevance": "high"},
            {"name": "Debt Ceiling", "type": "policy", "description": "Statutory limit on federal borrowing", "relevance": "medium"},
            {"name": "Midterm Election Pressure", "type": "event", "description": "Electoral incentives shaping legislative behavior", "relevance": "medium"},
            {"name": "Public Opinion Polls", "type": "metric", "description": "Voter sentiment on key issues driving political calculus", "relevance": "medium"},
        ],
        "key_uncertainties": [
            "Whether bipartisan deal on spending levels can be reached",
            "Impact of external crises on legislative priorities",
            "Individual legislator defections from party leadership",
            "Public opinion backlash risk for each party",
            "Timing of negotiations relative to deadlines",
        ],
    },
    "tech": {
        "entities": [
            {"name": "Large Language Models", "type": "concept", "description": "GPT, Claude, Gemini — frontier AI capabilities", "relevance": "high"},
            {"name": "Automation Technologies", "type": "concept", "description": "AI agents, RPA, autonomous systems replacing tasks", "relevance": "high"},
            {"name": "US Labor Market", "type": "metric", "description": "White-collar employment levels and job displacement data", "relevance": "high"},
            {"name": "Enterprise AI Adoption", "type": "metric", "description": "Rate of corporate AI deployment across industries", "relevance": "high"},
            {"name": "Regulatory Response", "type": "policy", "description": "Government AI regulation, labor protections, retraining programs", "relevance": "medium"},
            {"name": "AI Safety Research", "type": "concept", "description": "Alignment work and capability limitations", "relevance": "medium"},
            {"name": "Historical Technology Adoption", "type": "concept", "description": "Past technology displacement patterns and timelines", "relevance": "medium"},
        ],
        "key_uncertainties": [
            "Speed of AI capability improvement (linear vs exponential)",
            "Enterprise adoption rate vs. organizational inertia",
            "Regulatory intervention slowing deployment",
            "New job creation offsetting displacement",
            "Whether current AI can reliably perform full job roles vs. augmenting tasks",
        ],
    },
}


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


def _generate_pressures(question: str, category: str) -> dict:
    pressure_bank = {
        "econ": {
            "for_yes": [
                "Inflation trending toward 2% target gives Fed room to cut",
                "Slowing labor market — rising unemployment claims suggest cooling",
                "Manufacturing PMI contraction signals economic weakness",
                "Consumer confidence declining, spending pulling back",
                "Global central banks already cutting, creating coordination pressure",
                "Yield curve dynamics pricing in rate cuts",
            ],
            "for_no": [
                "Core inflation remains sticky above target",
                "Strong labor market with low unemployment",
                "Wage growth still elevated, fueling services inflation",
                "Housing market resilient, financial conditions loose",
                "Fed rhetoric emphasizing patience and data-dependence",
                "Geopolitical uncertainty counsels caution on policy changes",
            ],
            "uncertain": [
                "Impact of tariff policy changes on inflation expectations",
                "Consumer savings buffer depletion timeline",
                "Global contagion risk from emerging market stress",
            ],
        },
        "political": {
            "for_yes": [
                "Deep partisan polarization makes compromise difficult",
                "Historical precedent — shutdowns have occurred in 4 of last 10 years",
                "Budget disagreements on defense vs. domestic spending levels",
                "Political incentives to use shutdown as leverage",
                "Narrow Congressional margins amplify individual holdout power",
                "Debt ceiling deadline creating additional fiscal friction",
            ],
            "for_no": [
                "Both parties face electoral backlash from shutdowns",
                "Continuing resolutions provide temporary extension mechanism",
                "Leadership has incentives to avoid disruption before midterms",
                "Past brinkmanship usually resolved at the last minute",
                "Public disapproval of shutdowns is bipartisan",
                "Essential services pressure creates urgency to resolve",
            ],
            "uncertain": [
                "Impact of unexpected external crisis on legislative priorities",
                "Individual legislator defections and caucus dynamics",
                "Timing of critical votes relative to recess schedules",
            ],
        },
        "tech": {
            "for_yes": [
                "AI capabilities improving rapidly — GPT-5, Claude 4 showing agent-level performance",
                "Major enterprises announcing AI-driven headcount reductions",
                "Cost pressure: AI workers cost 1/10th of human equivalents for routine tasks",
                "Historical pattern: technology adoption accelerates once ROI is proven",
                "Coding, customer service, data analysis already seeing significant automation",
                "Venture capital flooding into AI agent startups",
            ],
            "for_no": [
                "10% is an enormous number — millions of jobs in 2-3 years",
                "Historical technology adoption is slower than predicted",
                "Organizational inertia, change management costs, and integration complexity",
                "Regulatory and union resistance to rapid displacement",
                "AI hallucination and reliability issues limit deployment in critical roles",
                "New roles created by AI may offset displaced ones",
            ],
            "uncertain": [
                "Whether AI agents can reliably handle full job functions vs. task augmentation",
                "Speed of regulatory response to AI displacement",
                "Macroeconomic conditions affecting corporate AI investment budgets",
            ],
        },
    }
    return pressure_bank.get(category, pressure_bank["econ"])


def _generate_timeline(question: str, category: str) -> list:
    timeline_bank = {
        "econ": [
            {"date_or_period": "Jan 2026", "event": "Fed holds rates steady, signals data-dependence", "impact": "Markets price in patience"},
            {"date_or_period": "Feb 2026", "event": "January jobs report shows 180K nonfarm payrolls", "impact": "Mixed signal — not weak enough to force cut"},
            {"date_or_period": "Mar 2026", "event": "PCE inflation at 2.4%, above target", "impact": "Reduces probability of near-term cut"},
            {"date_or_period": "Apr 2026", "event": "Q1 GDP growth slows to 1.2%", "impact": "Strengthens case for easing"},
            {"date_or_period": "May 2026", "event": "FOMC meeting — rate decision", "impact": "Decision point for the prediction question"},
        ],
        "political": [
            {"date_or_period": "Jan 2026", "event": "New Congress seated, leadership elections", "impact": "Sets negotiating dynamics"},
            {"date_or_period": "Feb 2026", "event": "President submits FY2027 budget proposal", "impact": "Opens appropriations debate"},
            {"date_or_period": "Mar 2026", "event": "Continuing resolution expires", "impact": "Creates first shutdown risk window"},
            {"date_or_period": "Apr 2026", "event": "Committee markups on spending bills", "impact": "Reveals areas of disagreement"},
            {"date_or_period": "May-Jun 2026", "event": "Floor votes and conference negotiations", "impact": "Peak period for shutdown risk"},
        ],
        "tech": [
            {"date_or_period": "2024", "event": "ChatGPT reaches 200M users, coding agents emerge", "impact": "Proved AI mainstream viability"},
            {"date_or_period": "Early 2025", "event": "Major companies announce AI-driven restructuring", "impact": "White-collar displacement begins"},
            {"date_or_period": "Mid 2025", "event": "AI agent frameworks mature (Claude Agent SDK, AutoGen)", "impact": "Enables autonomous task completion"},
            {"date_or_period": "Late 2025", "event": "First wave of regulatory proposals on AI and labor", "impact": "Creates uncertainty for rapid adoption"},
            {"date_or_period": "2026-2028", "event": "Enterprise AI deployment at scale", "impact": "Determines actual displacement rate"},
        ],
    }
    return timeline_bank.get(category, timeline_bank["econ"])


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
_REASONING_TEMPLATES = {
    "econ": [
        "Based on my analysis of {factor1}, I estimate P(YES) = {score:.2f}. {factor2} supports this view, though {uncertainty} introduces meaningful uncertainty. Historical base rates for similar Fed decisions suggest {direction} outcomes are {likelihood}.",
        "My {background} perspective leads me to weigh {factor1} heavily. Current {factor2} data points toward a {direction} probability. I assign P(YES) = {score:.2f}, acknowledging that {uncertainty} could shift this significantly.",
        "Looking at {factor1} through the lens of {background} expertise, I see {direction} probability at {score:.2f}. The key driver is {factor2}. My main concern is {uncertainty}, which could move this estimate by 10-15 points.",
    ],
    "political": [
        "From my {background} perspective, I assess P(YES) = {score:.2f}. {factor1} is the primary driver. {factor2} adds additional weight. The key wildcard is {uncertainty}, which makes this harder to forecast with precision.",
        "Congressional dynamics around {factor1} suggest {direction} probability. My analysis of {factor2} reinforces this at P(YES) = {score:.2f}. Historical precedent and {uncertainty} create non-trivial forecast risk.",
        "As a {background}, I focus on {factor1} and {factor2}. These suggest P(YES) = {score:.2f}. The {direction} case depends heavily on {uncertainty} resolving favorably.",
    ],
    "tech": [
        "My {background} experience tells me {factor1} is the critical variable. Combined with {factor2}, I estimate P(YES) = {score:.2f}. The {direction} scenario is {likelihood}, but {uncertainty} makes this a wide-confidence forecast.",
        "Analyzing {factor1} alongside {factor2}, I arrive at P(YES) = {score:.2f}. My {background} lens emphasizes that {uncertainty} is often underestimated in technology adoption forecasts. The {direction} case is {likelihood}.",
        "From a {background} standpoint, {factor1} is evolving {likelihood_adv}. {factor2} suggests P(YES) = {score:.2f}. Key risk: {uncertainty} could accelerate or brake the timeline significantly.",
    ],
}

_FACTORS = {
    "econ": {
        "factors": ["monetary policy trajectory", "inflation persistence", "labor market cooling", "yield curve signals", "consumer spending trends", "global growth synchronization", "fiscal policy drag", "housing market dynamics"],
        "uncertainties": ["tariff policy changes", "geopolitical shock risk", "banking sector stress", "oil price volatility", "election-year fiscal dynamics"],
    },
    "political": {
        "factors": ["partisan polarization depth", "legislative calendar pressure", "leadership negotiation leverage", "public opinion polls", "electoral incentive structure", "caucus cohesion levels", "appropriations committee dynamics", "media narrative framing"],
        "uncertainties": ["unexpected crisis diverting attention", "individual legislator defections", "judicial rulings affecting legislative authority", "public protest or activism waves"],
    },
    "tech": {
        "factors": ["AI capability acceleration", "enterprise adoption velocity", "cost reduction from automation", "workforce retraining capacity", "regulatory framework development", "organizational change management", "venture funding into AI agents", "open-source model proliferation"],
        "uncertainties": ["AI reliability and hallucination rates", "regulatory intervention timing", "macroeconomic downturn slowing investment", "public backlash against automation", "unexpected technical breakthroughs or limitations"],
    },
}


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

_EVIDENCE_TEMPLATES = {
    "econ": [
        "PCE inflation trending at {val:.1f}% — {direction} the 2% target",
        "Unemployment claims {trend} over the past 4 weeks",
        "Manufacturing PMI at {val:.0f} — {zone} territory",
        "Fed funds futures imply {val:.0f}% probability of rate action",
        "10Y-2Y spread at {val:+.0f}bps — {signal}",
        "Consumer confidence index {trend} to {val:.0f}",
    ],
    "political": [
        "Congressional approval at {val:.0f}% — {direction} historical average",
        "{val:.0f} appropriations bills passed out of 12 required",
        "Polling shows {val:.0f}% blame {party} for shutdown scenario",
        "Continuing resolution expires in {val:.0f} days",
        "Leadership whip count shows {val:.0f} holdouts in majority caucus",
        "Defense spending gap between chambers: ${val:.0f}B",
    ],
    "tech": [
        "Enterprise AI adoption rate: {val:.0f}% of Fortune 500 deploying agents",
        "AI-related job postings {trend} {val:.0f}% YoY",
        "White-collar unemployment in AI-exposed sectors: {val:.1f}%",
        "Corporate AI spending grew {val:.0f}% in latest quarter",
        "Regulatory AI workforce bills introduced in {val:.0f} state legislatures",
        "Average AI agent handles {val:.0f}x more tasks than 12 months ago",
    ],
}


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

_INSIGHT_BANK = {
    "econ": [
        "The labor market data remains the key swing factor.",
        "Inflation persistence is being underweighted by the consensus.",
        "Historical precedent suggests the Fed moves slower than markets expect.",
        "Global coordination pressure from other central banks is underappreciated.",
        "The yield curve is sending a clearer signal than the headlines.",
        "Consumer spending resilience complicates the rate cut thesis.",
        "The gap between headline and core inflation matters more than most acknowledge.",
        "Real interest rates adjusted for expected inflation tell a different story.",
        "Financial conditions indices show more tightening than the headline rate suggests.",
    ],
    "political": [
        "Electoral incentives often override ideological positions near deadlines.",
        "The narrow majority makes individual holdouts disproportionately powerful.",
        "Media pressure typically forces last-minute compromise.",
        "Historical shutdown probability given current polarization levels is higher than base rate.",
        "The timing relative to recess creates underappreciated pressure.",
        "Public opinion data suggests backlash risk is asymmetric across parties.",
        "Continuing resolution mechanics provide escape valves that reduce shutdown duration.",
        "Cross-party coalitions on specific spending areas can break overall gridlock.",
    ],
    "tech": [
        "Enterprise adoption curves are S-shaped — slow start, then rapid acceleration.",
        "The 10% threshold is much higher than current displacement rates.",
        "AI augmentation ≠ AI replacement — the distinction matters for this forecast.",
        "Regulatory friction will slow deployment more than technological capability.",
        "The open-source model ecosystem is accelerating adoption faster than expected.",
        "Organizational change management is the real bottleneck, not technology.",
        "Job displacement happens at the task level first, then the role level — there's a lag.",
        "Tight labor markets actually accelerate automation adoption.",
    ],
}
