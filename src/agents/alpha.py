"""
Alpha computation — question-specific alpha signals and domain expertise matching.

Extracted from offline_engine.py for modularity.
"""
from __future__ import annotations


def _compute_question_alpha(question: str, category: str) -> float:
    """Compute a question-specific alpha signal independent of market price.

    This is where the swarm adds value OVER the market. It detects signals
    in the question text that historically correlate with mispricings.

    Returns a float in [-0.15, +0.15] that shifts the swarm away from market.
    Positive = swarm thinks market underprices YES.
    Negative = swarm thinks market overprices YES.
    """
    q = question.lower()
    alpha = 0.0

    # --- Base rate anchoring (markets tend to overprice exciting events) ---
    # Questions about rare/dramatic events are typically overpriced by retail traders
    rare_event_kw = ["fail", "crash", "war", "invasion", "default", "collapse",
                     "agi", "breakthrough", "revolution", "ban", "impeach",
                     "resign", "assassin", "pandemic", "nuclear"]
    if any(kw in q for kw in rare_event_kw):
        alpha -= 0.08  # push toward NO — rare events are overpriced

    # Questions with "exceed" or "above" thresholds — markets underestimate inertia
    if any(kw in q for kw in ["exceed", "above", "more than", "over", "surpass"]):
        alpha -= 0.03  # slight NO bias — thresholds are harder to cross

    # Questions about continuation of trends — markets underestimate persistence
    if any(kw in q for kw in ["continue", "remain", "stay", "maintain", "keep"]):
        alpha += 0.05  # trends tend to persist

    # Institutional inertia — government/regulatory actions are slow
    if any(kw in q for kw in ["pass a", "enact", "regulation", "reform", "bill",
                               "legislation", "approve", "confirm"]):
        alpha -= 0.06  # institutional action is slow, markets overestimate speed

    # Technology adoption — markets overestimate speed of deployment
    if category == "tech" and any(kw in q for kw in ["replace", "automat", "adopt",
                                                       "deploy", "commercial", "mainstream"]):
        alpha -= 0.05  # adoption is slower than hype suggests

    # Fed/central bank — markets overreact to rate expectations
    if any(kw in q for kw in ["fed cut", "rate cut", "fed raise", "rate hike"]):
        alpha -= 0.04  # markets consistently over-anticipate Fed moves

    # High-confidence YES signals (well-established trends)
    if any(kw in q for kw in ["ai investment", "ai spending", "coding assistant",
                               "ai adoption", "cloud spending"]):
        alpha += 0.06  # AI growth trends are robust

    # Elections — incumbents have an advantage markets underestimate
    if any(kw in q for kw in ["re-elect", "incumbent", "win re-election"]):
        alpha += 0.05

    return max(-0.15, min(0.15, alpha))


def _compute_domain_expertise(bg_label: str, question: str, category: str) -> float:
    """Compute domain expertise match between an archetype and a question.

    Returns a confidence bonus [0.0, 0.25] — domain experts get higher
    confidence (and thus more weight in aggregation).
    """
    q = question.lower()
    label = bg_label.lower()

    # Direct domain matches
    matches = {
        "macro economist": ["fed", "rate", "inflation", "gdp", "recession", "monetary"],
        "central bank watcher": ["fed", "fomc", "rate cut", "rate hike", "central bank"],
        "retired central banker": ["fed", "rate", "fomc", "monetary"],
        "political analyst": ["election", "congress", "senate", "vote", "shutdown", "legislation"],
        "constitutional lawyer": ["supreme court", "ruling", "constitutional", "legal"],
        "pollster": ["approval", "poll", "voter", "election", "public opinion"],
        "tech executive": ["ai", "tech", "startup", "silicon valley", "software"],
        "venture capitalist": ["startup", "funding", "investment", "unicorn", "ipo"],
        "biotech researcher": ["drug", "fda", "clinical trial", "biotech", "pharma", "medical"],
        "climate scientist": ["climate", "emission", "temperature", "weather", "carbon"],
        "military intel analyst": ["war", "conflict", "military", "defense", "invasion"],
        "geopolitical strategist": ["tariff", "sanction", "trade war", "china", "russia", "nato"],
        "commodity trader": ["oil", "gold", "commodity", "opec", "energy price"],
        "crypto analyst": ["bitcoin", "crypto", "blockchain", "defi", "ethereum"],
        "labor economist": ["job", "employment", "unemployment", "wage", "labor", "worker"],
        "insurance actuary": ["risk", "probability", "fail", "default", "mortality"],
        "real estate developer": ["housing", "home price", "mortgage", "real estate"],
        "energy sector analyst": ["oil", "energy", "renewable", "solar", "opec"],
        "ai safety researcher": ["agi", "ai safety", "alignment", "ai regulation"],
        "data scientist": ["data", "model", "statistical", "prediction", "forecast"],
    }

    for archetype_key, keywords in matches.items():
        if archetype_key in label:
            if any(kw in q for kw in keywords):
                return 0.20  # strong domain match
            break

    # Partial category match
    if category == "econ" and any(kw in label for kw in ["economist", "trader", "banker", "analyst"]):
        return 0.08
    if category == "political" and any(kw in label for kw in ["political", "lawyer", "historian", "pollster"]):
        return 0.08
    if category == "tech" and any(kw in label for kw in ["tech", "data", "researcher", "crypto", "fintech"]):
        return 0.08

    return 0.0
