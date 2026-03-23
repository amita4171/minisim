"""
Offline simulation engine — no API calls required.
Generates diverse agents with background-specific probability distributions,
simulates peer influence rounds with personality-driven convergence,
and produces template-based reasoning strings.

This lets the full pipeline run locally (e.g., under Claude Code / Max subscription)
without needing a funded Anthropic API console account.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time

# ---------------------------------------------------------------------------
# BACKGROUNDS — each entry carries a "bias_center" per question *category*
# bias_center is the default P(YES) for that archetype on that topic class.
# Actual initial scores are sampled around this center with jitter.
# ---------------------------------------------------------------------------

BACKGROUNDS = [
    {"label": "Macro Economist", "detail": "PhD economist focused on monetary policy, inflation dynamics, and central bank behavior. Tracks PCE, CPI, employment data, and Fed minutes closely.", "econ": 0.45, "political": 0.40, "tech": 0.35},
    {"label": "Quantitative Trader", "detail": "Builds systematic models, thinks in probabilities and base rates. Uses options-implied probabilities and historical analogs.", "econ": 0.42, "political": 0.38, "tech": 0.40},
    {"label": "Political Analyst", "detail": "Tracks legislative dynamics, lobbying, regulatory shifts, and election cycles. Deep knowledge of Congressional mechanics.", "econ": 0.38, "political": 0.55, "tech": 0.30},
    {"label": "Investigative Journalist", "detail": "Skeptical by nature, hunts for hidden information and contrarian angles. Distrusts consensus narratives.", "econ": 0.35, "political": 0.45, "tech": 0.25},
    {"label": "Venture Capitalist", "detail": "Pattern-matches across industries, optimistic bias toward innovation and disruption. Sees technology as deflationary.", "econ": 0.50, "political": 0.35, "tech": 0.65},
    {"label": "Retired Central Banker", "detail": "Deep institutional knowledge of Fed/ECB decision-making processes. Understands the politics inside the FOMC.", "econ": 0.48, "political": 0.42, "tech": 0.30},
    {"label": "Behavioral Economist", "detail": "Studies cognitive biases, market sentiment, and herd behavior. Believes markets systematically misprice tail events.", "econ": 0.40, "political": 0.45, "tech": 0.45},
    {"label": "Geopolitical Strategist", "detail": "Analyzes international relations, trade wars, sanctions, and their second-order effects on domestic policy.", "econ": 0.43, "political": 0.50, "tech": 0.35},
    {"label": "Data Scientist", "detail": "Relies on statistical models, historical base rates, and ensemble methods. Skeptical of narratives without data.", "econ": 0.40, "political": 0.42, "tech": 0.50},
    {"label": "Hedge Fund PM", "detail": "Risk-adjusted thinker, contrarian when consensus is strong. Looks for asymmetric bets and mispriced risk.", "econ": 0.38, "political": 0.40, "tech": 0.45},
    {"label": "Constitutional Lawyer", "detail": "Analyzes legal frameworks, precedent, regulatory authority, and separation-of-powers dynamics.", "econ": 0.35, "political": 0.55, "tech": 0.28},
    {"label": "Supply Chain Analyst", "detail": "Tracks real-economy signals — shipping data, commodity flows, inventory cycles, and manufacturing PMIs.", "econ": 0.47, "political": 0.35, "tech": 0.40},
    {"label": "Climate Scientist", "detail": "Understands long-term systemic risks, physical economy impacts, and energy transition dynamics.", "econ": 0.42, "political": 0.38, "tech": 0.55},
    {"label": "Tech Executive", "detail": "Silicon Valley perspective, innovation-driven worldview. Believes in exponential curves and platform effects.", "econ": 0.52, "political": 0.30, "tech": 0.70},
    {"label": "Labor Economist", "detail": "Focuses on employment data, wage dynamics, union activity, and labor market tightness indicators.", "econ": 0.44, "political": 0.45, "tech": 0.50},
    {"label": "Emerging Markets Specialist", "detail": "Tracks capital flows, currency dynamics, contagion risks, and dollar-denominated debt stress.", "econ": 0.46, "political": 0.40, "tech": 0.35},
    {"label": "Military Intel Analyst", "detail": "Trained in structured analytic techniques, red-team thinking, and Analysis of Competing Hypotheses.", "econ": 0.40, "political": 0.48, "tech": 0.38},
    {"label": "Academic Historian", "detail": "Draws parallels from historical episodes — 1970s stagflation, 2008 crisis, dot-com bust. Skeptical of 'this time is different.'", "econ": 0.38, "political": 0.50, "tech": 0.30},
    {"label": "Insurance Actuary", "detail": "Probabilistic thinker focused on tail risks, base rates, and mortality/morbidity tables. Conservative estimator.", "econ": 0.36, "political": 0.40, "tech": 0.35},
    {"label": "Retail Investor", "detail": "Follows social media sentiment, Reddit, FinTwit. Reads popular narrative but lacks institutional depth.", "econ": 0.50, "political": 0.45, "tech": 0.60},
    {"label": "Commodity Trader", "detail": "Tracks oil, metals, agricultural futures. Understands physical markets and supply/demand fundamentals.", "econ": 0.44, "political": 0.35, "tech": 0.32},
    {"label": "Bayesian Statistician", "detail": "Updates beliefs incrementally using prior distributions. Resists large jumps without strong evidence.", "econ": 0.41, "political": 0.43, "tech": 0.42},
    {"label": "Pollster", "detail": "Expert in survey methodology, sampling bias, and public opinion trends. Understands how sentiment translates to action.", "econ": 0.43, "political": 0.52, "tech": 0.40},
    {"label": "Pharmaceutical Exec", "detail": "Understands FDA approval cycles, clinical trial outcomes, biotech funding environment, and healthcare policy.", "econ": 0.45, "political": 0.40, "tech": 0.55},
    {"label": "Energy Sector Analyst", "detail": "Tracks oil prices, OPEC dynamics, renewables deployment, and energy policy. Understands energy's macro impact.", "econ": 0.46, "political": 0.38, "tech": 0.45},
    {"label": "Real Estate Developer", "detail": "Sensitive to interest rates, housing supply, zoning, and demographic shifts. Thinks in 5-10 year cycles.", "econ": 0.50, "political": 0.42, "tech": 0.30},
    {"label": "Crypto Analyst", "detail": "Tracks on-chain data, DeFi flows, regulatory developments. Libertarian-leaning, skeptical of central authority.", "econ": 0.55, "political": 0.35, "tech": 0.65},
    {"label": "Union Organizer", "detail": "Understands labor dynamics from the ground up. Tracks wage negotiations, strike activity, and worker sentiment.", "econ": 0.48, "political": 0.55, "tech": 0.52},
    {"label": "Demographer", "detail": "Studies population trends, aging, migration, and their economic consequences. Long-term structural thinker.", "econ": 0.40, "political": 0.45, "tech": 0.48},
    {"label": "Defense Contractor Exec", "detail": "Tracks government spending, defense budgets, geopolitical tensions, and procurement cycles.", "econ": 0.42, "political": 0.48, "tech": 0.40},
    {"label": "Fintech Founder", "detail": "Understands payments, lending, and financial infrastructure disruption. Optimistic about technology solving finance.", "econ": 0.52, "political": 0.32, "tech": 0.68},
    {"label": "Agricultural Economist", "detail": "Tracks food prices, crop yields, weather patterns, and trade policy impacts on farming.", "econ": 0.43, "political": 0.42, "tech": 0.35},
    {"label": "Central Bank Watcher", "detail": "Parses every Fed speech, dot plot, and minutes release. Expert at reading between the lines of central bank communication.", "econ": 0.47, "political": 0.40, "tech": 0.32},
    {"label": "Biotech Researcher", "detail": "Works on drug discovery and understands R&D timelines, regulatory hurdles, and scientific uncertainty.", "econ": 0.40, "political": 0.35, "tech": 0.60},
    {"label": "Sovereign Wealth Fund Analyst", "detail": "Manages long-term national reserves. Thinks in decades, tracks global macro imbalances and structural shifts.", "econ": 0.42, "political": 0.44, "tech": 0.38},
]

PERSONALITIES = [
    {"label": "cautious and evidence-driven", "convergence_rate": 0.15, "contrarian_factor": 0.0},
    {"label": "bold and contrarian", "convergence_rate": 0.05, "contrarian_factor": 0.3},
    {"label": "analytical and methodical", "convergence_rate": 0.20, "contrarian_factor": 0.0},
    {"label": "intuitive and pattern-matching", "convergence_rate": 0.25, "contrarian_factor": 0.05},
    {"label": "skeptical and devil's advocate", "convergence_rate": 0.08, "contrarian_factor": 0.25},
    {"label": "consensus-seeking and diplomatic", "convergence_rate": 0.35, "contrarian_factor": 0.0},
    {"label": "data-obsessed and quantitative", "convergence_rate": 0.18, "contrarian_factor": 0.0},
    {"label": "narrative-driven and qualitative", "convergence_rate": 0.22, "contrarian_factor": 0.05},
    {"label": "risk-averse and conservative", "convergence_rate": 0.12, "contrarian_factor": 0.0},
    {"label": "risk-seeking and opportunistic", "convergence_rate": 0.28, "contrarian_factor": 0.10},
]

# First-name and last-name pools for generating realistic agent names
FIRST_NAMES = [
    "James", "Maria", "David", "Sarah", "Michael", "Jennifer", "Robert", "Lisa",
    "William", "Emily", "Richard", "Jessica", "Thomas", "Amanda", "Charles", "Ashley",
    "Daniel", "Stephanie", "Matthew", "Nicole", "Andrew", "Rachel", "Joseph", "Laura",
    "Christopher", "Megan", "Anthony", "Elizabeth", "Kevin", "Rebecca", "Brian", "Katherine",
    "Steven", "Patricia", "Edward", "Christina", "George", "Michelle", "Kenneth", "Diana",
    "Raj", "Yuki", "Wei", "Fatima", "Carlos", "Olga", "Ahmed", "Priya", "Hans", "Aisha",
]

LAST_NAMES = [
    "Chen", "Patel", "Rodriguez", "Kim", "O'Brien", "Nakamura", "Santos", "Mueller",
    "Williams", "Thompson", "Garcia", "Anderson", "Jackson", "White", "Harris", "Martin",
    "Robinson", "Clark", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez",
    "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez",
    "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell",
    "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Reed",
]

# ---------------------------------------------------------------------------
# Question category detection
# ---------------------------------------------------------------------------

def _detect_category(question: str) -> str:
    """Heuristically classify a question into econ / political / tech."""
    q = question.lower()
    econ_kw = ["fed", "rate", "inflation", "gdp", "recession", "unemployment", "interest", "economy", "monetary", "fiscal", "stock", "market crash", "s&p", "dow", "treasury"]
    pol_kw = ["election", "president", "congress", "government", "shutdown", "impeach", "vote", "legislation", "bill", "senate", "supreme court", "governor", "partisan", "democrat", "republican", "executive order"]
    tech_kw = ["ai", "artificial intelligence", "replace", "automat", "robot", "quantum", "spacex", "launch", "biotech", "gene", "crispr", "autonomous", "self-driving", "chip", "semiconductor", "agi", "openai", "google", "apple", "tesla"]

    scores = {
        "econ": sum(1 for kw in econ_kw if kw in q),
        "political": sum(1 for kw in pol_kw if kw in q),
        "tech": sum(1 for kw in tech_kw if kw in q),
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
# Agent factory (offline)
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


def _make_name(idx: int) -> str:
    """Generate a deterministic but realistic name from index."""
    fi = idx % len(FIRST_NAMES)
    li = (idx * 7 + 3) % len(LAST_NAMES)  # scramble pairing
    return f"{FIRST_NAMES[fi]} {LAST_NAMES[li]}"


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


def generate_population_offline(
    question: str,
    world: dict,
    n_agents: int = 50,
    seed: int | None = None,
) -> tuple[list[dict], int]:
    """Generate N diverse agents with background-specific initial scores."""
    start = time.time()
    category = world.get("question_category", _detect_category(question))

    # Deterministic seed from question if not provided
    if seed is None:
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    agents = []
    for i in range(n_agents):
        bg = BACKGROUNDS[i % len(BACKGROUNDS)]
        pers = PERSONALITIES[i % len(PERSONALITIES)]

        # Sample initial score around archetype center with personality jitter
        center = bg[category]
        jitter = rng.gauss(0, 0.12)
        # Contrarians get pushed away from 0.5
        if pers["contrarian_factor"] > 0.1:
            jitter += rng.choice([-1, 1]) * pers["contrarian_factor"] * 0.15
        initial_score = max(0.02, min(0.98, center + jitter))

        confidence = max(0.2, min(0.95, 0.5 + rng.gauss(0, 0.15)))
        reasoning, key_factors = _generate_reasoning(bg["label"], initial_score, category, rng)

        agent = {
            "id": i,
            "name": _make_name(i),
            "background_category": bg["label"],
            "background_detail": bg["detail"],
            "personality": pers["label"],
            "initial_score": round(initial_score, 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
            "key_factors": key_factors,
            "score_history": [round(initial_score, 4)],
            "memory_stream": [
                f"Initial assessment: P(YES) = {initial_score:.2f}. {reasoning}"
            ],
            "_convergence_rate": pers["convergence_rate"],
            "_contrarian_factor": pers["contrarian_factor"],
        }
        agents.append(agent)

    elapsed_ms = int((time.time() - start) * 1000)
    return agents, elapsed_ms


# ---------------------------------------------------------------------------
# Simulation loop (offline) — mathematical peer influence
# ---------------------------------------------------------------------------

_REFLECTION_TEMPLATES = [
    "After seeing {n_peers} peer opinions (avg {peer_avg:.2f}), I {movement}. {insight}",
    "Peer views ranged from {peer_min:.2f} to {peer_max:.2f}. I adjusted {movement_dir} to {new_score:.2f}. {insight}",
    "Round {round_num} deliberation: peers averaged {peer_avg:.2f}. My {personality} tendency led me to {movement}. {insight}",
]

_INSIGHT_BANK = {
    "econ": [
        "The labor market data remains the key swing factor.",
        "Inflation persistence is being underweighted by the consensus.",
        "Historical precedent suggests the Fed moves slower than markets expect.",
        "Global coordination pressure from other central banks is underappreciated.",
        "The yield curve is sending a clearer signal than the headlines.",
        "Consumer spending resilience complicates the rate cut thesis.",
        "The gap between headline and core inflation matters more than most acknowledge.",
    ],
    "political": [
        "Electoral incentives often override ideological positions near deadlines.",
        "The narrow majority makes individual holdouts disproportionately powerful.",
        "Media pressure typically forces last-minute compromise.",
        "Historical shutdown probability given current polarization levels is higher than base rate.",
        "The timing relative to recess creates underappreciated pressure.",
        "Public opinion data suggests backlash risk is asymmetric across parties.",
    ],
    "tech": [
        "Enterprise adoption curves are S-shaped — slow start, then rapid acceleration.",
        "The 10% threshold is much higher than current displacement rates.",
        "AI augmentation ≠ AI replacement — the distinction matters for this forecast.",
        "Regulatory friction will slow deployment more than technological capability.",
        "The open-source model ecosystem is accelerating adoption faster than expected.",
        "Organizational change management is the real bottleneck, not technology.",
    ],
}


def run_simulation_offline(
    question: str,
    agents: list[dict],
    n_rounds: int = 3,
    peer_sample_size: int = 5,
    seed: int | None = None,
) -> tuple[list[dict], int]:
    """Run K rounds of peer-influence simulation without API calls."""
    start = time.time()
    category = _detect_category(question)

    if seed is None:
        seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16) + 1000
    rng = random.Random(seed)

    insights = _INSIGHT_BANK.get(category, _INSIGHT_BANK["econ"])

    for round_num in range(1, n_rounds + 1):
        for agent in agents:
            # Sample peers
            peers = [a for a in agents if a["id"] != agent["id"]]
            sampled = rng.sample(peers, min(peer_sample_size, len(peers)))

            peer_scores = [p["score_history"][-1] for p in sampled]
            peer_avg = statistics.mean(peer_scores)
            peer_min = min(peer_scores)
            peer_max = max(peer_scores)

            current = agent["score_history"][-1]
            conv_rate = agent["_convergence_rate"]
            contra = agent["_contrarian_factor"]

            # Move toward peer average (convergence) or away (contrarian)
            # Add diminishing noise each round
            noise = rng.gauss(0, 0.03 / round_num)
            if contra > 0.1 and abs(current - peer_avg) < 0.15:
                # Contrarians resist consensus — move away slightly
                delta = (current - peer_avg) * contra * 0.5
            else:
                # Normal convergence toward peer average
                delta = (peer_avg - current) * conv_rate

            new_score = current + delta + noise
            new_score = round(max(0.02, min(0.98, new_score)), 4)
            agent["score_history"].append(new_score)

            # Update confidence (increases slightly with rounds as agents see more data)
            old_conf = agent["confidence"]
            agent["confidence"] = round(min(0.95, old_conf + rng.uniform(0.01, 0.05)), 4)

            # Generate reflection
            shift = new_score - current
            if abs(shift) < 0.02:
                movement = "held my position largely steady"
                movement_dir = "minimally"
            elif shift > 0:
                movement = f"shifted upward by {shift:.2f}"
                movement_dir = "upward"
            else:
                movement = f"shifted downward by {abs(shift):.2f}"
                movement_dir = "downward"

            insight = rng.choice(insights)
            tmpl = rng.choice(_REFLECTION_TEMPLATES)
            reflection = tmpl.format(
                n_peers=len(sampled),
                peer_avg=peer_avg,
                peer_min=peer_min,
                peer_max=peer_max,
                movement=movement,
                movement_dir=movement_dir,
                new_score=new_score,
                round_num=round_num,
                personality=agent["personality"],
                insight=insight,
            )

            agent["memory_stream"].append(
                f"Round {round_num}: Updated to P(YES) = {new_score:.2f}. {reflection}"
            )

    elapsed_ms = int((time.time() - start) * 1000)
    return agents, elapsed_ms


# ---------------------------------------------------------------------------
# Full pipeline (offline)
# ---------------------------------------------------------------------------

def swarm_score_offline(
    question: str,
    context: str = "",
    n_agents: int = 50,
    rounds: int = 3,
    market_price: float | None = None,
    peer_sample_size: int = 5,
) -> dict:
    """Full offline pipeline: world build -> agent gen -> sim -> aggregation."""
    from src.aggregator import aggregate

    world = build_world_offline(question, context)
    agents, agent_gen_ms = generate_population_offline(question, world, n_agents)
    agents, sim_loop_ms = run_simulation_offline(question, agents, rounds, peer_sample_size)
    result = aggregate(agents, market_price)

    result["timing"] = {
        "world_build_ms": world.get("_build_time_ms", 0),
        "agent_gen_ms": agent_gen_ms,
        "sim_loop_ms": sim_loop_ms,
        "total_ms": world.get("_build_time_ms", 0) + agent_gen_ms + sim_loop_ms,
    }

    result["agents"] = [
        {
            "id": a["id"],
            "name": a["name"],
            "background_category": a["background_category"],
            "background_detail": a["background_detail"],
            "personality": a["personality"],
            "initial_score": a["score_history"][0],
            "final_score": a["score_history"][-1],
            "score_history": a["score_history"],
            "confidence": a.get("confidence", 0.5),
            "reasoning": a.get("reasoning", ""),
            "key_factors": a.get("key_factors", []),
            "memory_stream": a["memory_stream"],
        }
        for a in agents
    ]

    result["world_model"] = {k: v for k, v in world.items() if not k.startswith("_")}
    result["question"] = question
    result["config"] = {
        "n_agents": n_agents,
        "rounds": rounds,
        "peer_sample_size": peer_sample_size,
        "mode": "offline",
    }

    return result
