"""
Agent archetypes — backgrounds, personalities, temperature tiers, and name generation.

Extracted from offline_engine.py for modularity.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BACKGROUNDS — each entry carries a "bias_center" per question *category*
# bias_center is the default P(YES) for that archetype on that topic class.
# Actual initial scores are sampled around this center with jitter.
# ---------------------------------------------------------------------------

# Temperature tiers per research (arxiv 2510.01171, 2510.01218):
#   "analyst"    T=0.3  — deterministic, evidence-driven
#   "calibrator" T=0.5  — focus on base rates
#   "contrarian" T=0.9  — challenge consensus
#   "creative"   T=1.2  — explore unlikely scenarios

BACKGROUNDS = [
    # --- Analysts (T=0.3) — precise, evidence-driven ---
    {"label": "Macro Economist", "detail": "PhD economist focused on monetary policy, inflation dynamics, and central bank behavior. Tracks PCE, CPI, employment data, and Fed minutes closely.", "econ": 0.45, "political": 0.40, "tech": 0.35, "temp_tier": "analyst"},
    {"label": "Quantitative Trader", "detail": "Builds systematic models, thinks in probabilities and base rates. Uses options-implied probabilities and historical analogs.", "econ": 0.42, "political": 0.38, "tech": 0.40, "temp_tier": "analyst"},
    {"label": "Data Scientist", "detail": "Relies on statistical models, historical base rates, and ensemble methods. Skeptical of narratives without data.", "econ": 0.40, "political": 0.42, "tech": 0.50, "temp_tier": "analyst"},
    {"label": "Insurance Actuary", "detail": "Probabilistic thinker focused on tail risks, base rates, and mortality/morbidity tables. Conservative estimator.", "econ": 0.30, "political": 0.38, "tech": 0.28, "temp_tier": "analyst"},
    {"label": "Bayesian Statistician", "detail": "Updates beliefs incrementally using prior distributions. Resists large jumps without strong evidence.", "econ": 0.41, "political": 0.43, "tech": 0.42, "temp_tier": "analyst"},
    {"label": "Supply Chain Analyst", "detail": "Tracks real-economy signals — shipping data, commodity flows, inventory cycles, and manufacturing PMIs.", "econ": 0.47, "political": 0.35, "tech": 0.40, "temp_tier": "analyst"},
    {"label": "Central Bank Watcher", "detail": "Parses every Fed speech, dot plot, and minutes release. Expert at reading between the lines of central bank communication.", "econ": 0.50, "political": 0.40, "tech": 0.32, "temp_tier": "analyst"},
    {"label": "Sovereign Wealth Fund Analyst", "detail": "Manages long-term national reserves. Thinks in decades, tracks global macro imbalances and structural shifts.", "econ": 0.42, "political": 0.44, "tech": 0.38, "temp_tier": "analyst"},
    # --- Calibrators (T=0.5) — base-rate focused, moderate ---
    {"label": "Retired Central Banker", "detail": "Deep institutional knowledge of Fed/ECB decision-making processes. Understands the politics inside the FOMC.", "econ": 0.48, "political": 0.42, "tech": 0.30, "temp_tier": "calibrator"},
    {"label": "Behavioral Economist", "detail": "Studies cognitive biases, market sentiment, and herd behavior. Believes markets systematically misprice tail events.", "econ": 0.40, "political": 0.45, "tech": 0.45, "temp_tier": "calibrator"},
    {"label": "Labor Economist", "detail": "Focuses on employment data, wage dynamics, union activity, and labor market tightness indicators.", "econ": 0.44, "political": 0.45, "tech": 0.50, "temp_tier": "calibrator"},
    {"label": "Pollster", "detail": "Expert in survey methodology, sampling bias, and public opinion trends. Understands how sentiment translates to action.", "econ": 0.43, "political": 0.55, "tech": 0.40, "temp_tier": "calibrator"},
    {"label": "Demographer", "detail": "Studies population trends, aging, migration, and their economic consequences. Long-term structural thinker.", "econ": 0.40, "political": 0.45, "tech": 0.48, "temp_tier": "calibrator"},
    {"label": "Agricultural Economist", "detail": "Tracks food prices, crop yields, weather patterns, and trade policy impacts on farming.", "econ": 0.43, "political": 0.42, "tech": 0.35, "temp_tier": "calibrator"},
    {"label": "Energy Sector Analyst", "detail": "Tracks oil prices, OPEC dynamics, renewables deployment, and energy policy. Understands energy's macro impact.", "econ": 0.46, "political": 0.38, "tech": 0.45, "temp_tier": "calibrator"},
    {"label": "Emerging Markets Specialist", "detail": "Tracks capital flows, currency dynamics, contagion risks, and dollar-denominated debt stress.", "econ": 0.46, "political": 0.40, "tech": 0.35, "temp_tier": "calibrator"},
    {"label": "Pharmaceutical Exec", "detail": "Understands FDA approval cycles, clinical trial outcomes, biotech funding environment, and healthcare policy.", "econ": 0.45, "political": 0.40, "tech": 0.58, "temp_tier": "calibrator"},
    # --- Contrarians (T=0.9) — challenge consensus, wider variance ---
    {"label": "Investigative Journalist", "detail": "Skeptical by nature, hunts for hidden information and contrarian angles. Distrusts consensus narratives.", "econ": 0.28, "political": 0.55, "tech": 0.20, "temp_tier": "contrarian"},
    {"label": "Hedge Fund PM", "detail": "Risk-adjusted thinker, contrarian when consensus is strong. Looks for asymmetric bets and mispriced risk.", "econ": 0.32, "political": 0.48, "tech": 0.52, "temp_tier": "contrarian"},
    {"label": "Academic Historian", "detail": "Draws parallels from historical episodes — 1970s stagflation, 2008 crisis, dot-com bust. Skeptical of 'this time is different.'", "econ": 0.30, "political": 0.55, "tech": 0.22, "temp_tier": "contrarian"},
    {"label": "Military Intel Analyst", "detail": "Trained in structured analytic techniques, red-team thinking, and Analysis of Competing Hypotheses.", "econ": 0.35, "political": 0.52, "tech": 0.35, "temp_tier": "contrarian"},
    {"label": "Constitutional Lawyer", "detail": "Analyzes legal frameworks, precedent, regulatory authority, and separation-of-powers dynamics.", "econ": 0.30, "political": 0.60, "tech": 0.22, "temp_tier": "contrarian"},
    {"label": "Commodity Trader", "detail": "Tracks oil, metals, agricultural futures. Understands physical markets and supply/demand fundamentals.", "econ": 0.48, "political": 0.30, "tech": 0.25, "temp_tier": "contrarian"},
    {"label": "Conspiracy Skeptic", "detail": "Systematically debunks popular narratives. Tests claims against evidence rigorously. Finds hidden assumptions in consensus views.", "econ": 0.25, "political": 0.60, "tech": 0.18, "temp_tier": "contrarian"},
    {"label": "Short Seller", "detail": "Professional skeptic. Hunts for overvalued assets, fraud, and overhyped narratives. Makes money when consensus is wrong.", "econ": 0.28, "political": 0.50, "tech": 0.20, "temp_tier": "contrarian"},
    # --- Creatives (T=1.2) — explore unlikely scenarios, wide range ---
    {"label": "Venture Capitalist", "detail": "Pattern-matches across industries, optimistic bias toward innovation and disruption. Sees technology as deflationary.", "econ": 0.55, "political": 0.30, "tech": 0.72, "temp_tier": "creative"},
    {"label": "Tech Executive", "detail": "Silicon Valley perspective, innovation-driven worldview. Believes in exponential curves and platform effects.", "econ": 0.58, "political": 0.25, "tech": 0.78, "temp_tier": "creative"},
    {"label": "Crypto Analyst", "detail": "Tracks on-chain data, DeFi flows, regulatory developments. Libertarian-leaning, skeptical of central authority.", "econ": 0.60, "political": 0.28, "tech": 0.72, "temp_tier": "creative"},
    {"label": "Fintech Founder", "detail": "Understands payments, lending, and financial infrastructure disruption. Optimistic about technology solving finance.", "econ": 0.55, "political": 0.28, "tech": 0.75, "temp_tier": "creative"},
    {"label": "Retail Investor", "detail": "Follows social media sentiment, Reddit, FinTwit. Reads popular narrative but lacks institutional depth.", "econ": 0.52, "political": 0.50, "tech": 0.65, "temp_tier": "creative"},
    {"label": "Biotech Researcher", "detail": "Works on drug discovery and understands R&D timelines, regulatory hurdles, and scientific uncertainty.", "econ": 0.40, "political": 0.35, "tech": 0.65, "temp_tier": "creative"},
    {"label": "Climate Scientist", "detail": "Understands long-term systemic risks, physical economy impacts, and energy transition dynamics.", "econ": 0.42, "political": 0.38, "tech": 0.60, "temp_tier": "creative"},
    # --- Domain specialists — mixed tiers ---
    {"label": "Political Analyst", "detail": "Tracks legislative dynamics, lobbying, regulatory shifts, and election cycles. Deep knowledge of Congressional mechanics.", "econ": 0.38, "political": 0.62, "tech": 0.30, "temp_tier": "calibrator"},
    {"label": "Geopolitical Strategist", "detail": "Analyzes international relations, trade wars, sanctions, and their second-order effects on domestic policy.", "econ": 0.43, "political": 0.55, "tech": 0.35, "temp_tier": "calibrator"},
    {"label": "Union Organizer", "detail": "Understands labor dynamics from the ground up. Tracks wage negotiations, strike activity, and worker sentiment.", "econ": 0.48, "political": 0.58, "tech": 0.55, "temp_tier": "contrarian"},
    {"label": "Real Estate Developer", "detail": "Sensitive to interest rates, housing supply, zoning, and demographic shifts. Thinks in 5-10 year cycles.", "econ": 0.52, "political": 0.42, "tech": 0.28, "temp_tier": "calibrator"},
    {"label": "Defense Contractor Exec", "detail": "Tracks government spending, defense budgets, geopolitical tensions, and procurement cycles.", "econ": 0.42, "political": 0.52, "tech": 0.40, "temp_tier": "analyst"},
    {"label": "Historical Analogist", "detail": "Maps current events onto historical templates — Weimar, 1930s, 1970s, Japanese lost decade. Strong pattern-matching.", "econ": 0.33, "political": 0.50, "tech": 0.25, "temp_tier": "contrarian"},
    {"label": "AI Safety Researcher", "detail": "Studies alignment, capability elicitation, and societal impact of AI systems. Calibrated about AI limitations.", "econ": 0.38, "political": 0.35, "tech": 0.45, "temp_tier": "analyst"},
    {"label": "Management Consultant", "detail": "Works with Fortune 500 on transformation. Sees enterprise adoption friction firsthand — budgets, change mgmt, politics.", "econ": 0.45, "political": 0.40, "tech": 0.42, "temp_tier": "calibrator"},
]

# Temperature mapping for the stratification tiers
TEMP_TIERS = {
    "analyst": {"temperature": 0.3, "jitter_std": 0.12},
    "calibrator": {"temperature": 0.5, "jitter_std": 0.14},
    "contrarian": {"temperature": 0.9, "jitter_std": 0.22},
    "creative": {"temperature": 1.2, "jitter_std": 0.24},
}

PERSONALITIES = [
    {"label": "cautious and evidence-driven", "convergence_rate": 0.08, "contrarian_factor": 0.0},
    {"label": "bold and contrarian", "convergence_rate": 0.03, "contrarian_factor": 0.40},
    {"label": "analytical and methodical", "convergence_rate": 0.10, "contrarian_factor": 0.0},
    {"label": "intuitive and pattern-matching", "convergence_rate": 0.12, "contrarian_factor": 0.05},
    {"label": "skeptical and devil's advocate", "convergence_rate": 0.04, "contrarian_factor": 0.35},
    {"label": "consensus-seeking and diplomatic", "convergence_rate": 0.18, "contrarian_factor": 0.0},
    {"label": "data-obsessed and quantitative", "convergence_rate": 0.09, "contrarian_factor": 0.0},
    {"label": "narrative-driven and qualitative", "convergence_rate": 0.11, "contrarian_factor": 0.05},
    {"label": "risk-averse and conservative", "convergence_rate": 0.06, "contrarian_factor": 0.10},
    {"label": "risk-seeking and opportunistic", "convergence_rate": 0.14, "contrarian_factor": 0.15},
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


def _make_name(idx: int) -> str:
    """Generate a deterministic but realistic name from index."""
    fi = idx % len(FIRST_NAMES)
    li = (idx * 7 + 3) % len(LAST_NAMES)  # scramble pairing
    return f"{FIRST_NAMES[fi]} {LAST_NAMES[li]}"
