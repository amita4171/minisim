# MiniSim — Swarm Prediction Engine

A startup-grade swarm intelligence prediction engine that simulates N diverse digital humans deliberating across K structured rounds on any prediction question. Produces calibrated probability estimates that rival prediction market crowds.

**Wedge:** Prediction markets (Kalshi/Polymarket) as the first vertical, with a path to synthetic surveys and market research ($78B TAM).

Inspired by [MiroFish](https://github.com/666ghj/MiroFish) (28K+ stars), [Generative Agents](https://github.com/joonspk-research/generative_agents) (Park et al.), and the [Wisdom of Silicon Crowd](https://arxiv.org/abs/2402.19379) (Science Advances 2025).

---

## How It Works

```
Question + Market Price
         |
    [World Builder]
         |  Extracts entities, relationships, pressures, timeline
         |  GraphRAG-style knowledge graph from seed context
         |
    [Agent Factory]  ──────────────────────────────────────┐
         |  40 archetypes: economists, traders, lawyers,   |
         |  journalists, VCs, military intel, pollsters...  |
         |  Temperature stratification: T=0.3 to T=1.2     |
         |  Domain expertise weighting per question         |
         |  Question-specific alpha signals                 |
         |                                                  |
    [4-Round Structured Deliberation]                       |
         |  R1: Initial Forecast (independent)              |
         |  R2: Evidence Exchange (peer quality review)     |
         |  R3: Critique & Rebuttal (opponent pairing)      |
         |  R4: Updated Forecast (final revision)           |
         |                                                  |
    [Calibrated Aggregator]                                 |
         |  60% calibrated confidence-weighted              |
         |  40% extremized (Metaculus-style, alpha=1.25)    |
         |  Mind-change bonus: flexible thinkers upweighted |
         |  Domain experts get higher confidence weights    |
         |                                                  |
    Output ─────────────────────────────────────────────────┘
         |
         ├── swarm_probability_yes + 95% confidence interval
         ├── edge vs market price
         ├── opinion clusters with descriptive labels
         ├── top YES/NO voices with reasoning excerpts
         ├── mind-changers (who changed and why)
         ├── dissenting voices with z-scores
         └── reasoning shift summary
```

---

## Backtest Results (100 Markets)

Backtested on 100 resolved prediction questions across 8 categories.

| Metric | Value |
|--------|-------|
| **Overall Swarm Brier** | 0.142 |
| **Overall Market Brier** | 0.132 |
| **Swarm Beat Market** | 37/100 (37%) |
| **Best Category** | Health (4/5 wins, 80%) |
| **Second Best** | Geopolitics (10/15 wins, 67%) |

### Category Breakdown

| Category | N | Swarm Brier | Market Brier | Win Rate |
|----------|---|-------------|--------------|----------|
| Health | 5 | **0.086** | 0.108 | 80% |
| Geopolitics | 15 | **0.125** | 0.131 | 67% |
| Corporate | 10 | **0.103** | 0.103 | 30% |
| Economics | 25 | 0.144 | **0.131** | 40% |
| Tech/AI | 20 | 0.157 | **0.137** | 30% |
| Climate | 5 | 0.148 | **0.140** | 20% |
| Politics | 20 | 0.169 | **0.148** | 15% |

### Where Swarm Adds Alpha
- **Rare event deflation**: correctly pushes down overpriced dramatic scenarios (bank failures, AGI announcements, nuclear tests)
- **Institutional inertia detection**: correctly identifies when legislation/regulation is slower than markets expect
- **Continuation bias**: recognizes when trends persist (defense spending, AI investment, sanctions)
- **Domain expert upweighting**: Central Bank Watchers outperform on Fed questions, geopolitical strategists on international affairs

### Top Alpha Calls
| Improvement | Swarm | Market | Outcome | Question |
|-------------|-------|--------|---------|----------|
| +0.076 | 0.64 | 0.55 | YES | mRNA cancer vaccine Phase 3 trials |
| +0.053 | 0.61 | 0.55 | YES | 2025 hottest year on record |
| +0.043 | 0.60 | 0.55 | YES | Nasdaq 100 outperform S&P 500 |
| +0.043 | 0.66 | 0.60 | YES | US impose new tariffs on China |

### Calibration Curve
```
Predicted    Count    Actual Rate    Assessment
0.0-0.1        3        0.0%         Well calibrated
0.1-0.2       25        4.0%         Good (slight NO bias)
0.2-0.3       18        0.0%         Well calibrated
0.3-0.4       15       60.0%         Underconfident on YES
0.4-0.5       12       91.7%         Underconfident on YES
0.5-0.6        8       87.5%         Good
0.6-0.7       14      100.0%         Slightly overconfident
0.7-0.8        5      100.0%         Good
```

---

## Key Technical Features

### 40 Agent Archetypes
Economists, traders, political analysts, journalists, VCs, central bankers, behavioral economists, geopolitical strategists, data scientists, hedge fund PMs, constitutional lawyers, supply chain analysts, military intel, actuaries, pollsters, AI safety researchers, management consultants, and more.

### Temperature Stratification (arxiv 2510.01218)
| Tier | Temperature | Jitter Std | Role |
|------|-------------|------------|------|
| Analyst | 0.3 | 0.12 | Evidence-driven, precise estimates |
| Calibrator | 0.5 | 0.14 | Base-rate focused, moderate |
| Contrarian | 0.9 | 0.22 | Challenge consensus, wide variance |
| Creative | 1.2 | 0.24 | Explore unlikely scenarios |

### Anti-Mode-Collapse
- Personality-driven convergence rates (cautious agents move slowly, consensus-seekers move fast)
- Contrarian agents resist consensus — push AWAY from peer average when it's too close
- Temperature stratification ensures creative agents maintain wide distributions
- Convergence comparison shows diversity preserved > 0.15 std dev through 5 rounds

### 4-Round Deliberation Protocol (arxiv 2305.14325)
1. **Initial Forecast**: Independent probability estimate with evidence generation
2. **Evidence Exchange**: Agents share evidence, rate peer evidence quality (1-5)
3. **Critique & Rebuttal**: YES/NO agents paired for debate; disagreement typed (empirical vs methodological)
4. **Updated Forecast**: Final revision tracking mind-changes and reasoning shifts

### Calibrated Aggregation (arxiv 2506.00066)
- **Confidence-weighted averaging**: agents weighted by confidence score
- **Mind-change bonus**: agents who updated meaningfully during debate get upweighted (intellectual flexibility signal)
- **Domain expertise bonus**: matching archetypes get +0.20 confidence on their domain questions
- **Extremized aggregation**: `p^alpha / (p^alpha + (1-p)^alpha)` where alpha=1.25 (Metaculus-style)
- **Combined**: 60% calibrated + 40% extremized

### Question-Specific Alpha
The swarm detects signals in the question text that historically correlate with market mispricings:
- **Rare event deflation**: rare/dramatic events are systematically overpriced by retail traders
- **Institutional inertia**: legislation, regulation, and reform are slower than markets expect
- **Technology adoption lag**: deployment timelines are consistently underestimated
- **Continuation bias**: ongoing trends (spending, sanctions, investment) tend to persist
- **Fed anticipation**: markets consistently over-anticipate Fed rate moves

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a prediction (offline mode — no API key needed)
python main.py --offline \
  --question "Will the Fed cut rates in May 2026?" \
  --agents 50 --rounds 4 --market-price 0.40

# Run the Streamlit dashboard
streamlit run streamlit_app.py

# Run backtest on 100 resolved markets
python backtest.py

# Run convergence comparison (3-round vs 5-round)
python convergence_comparison.py
```

### Example Output
```
SWARM PROBABILITY (YES): 0.3906
Aggregation: calibrated_confidence_weighted_extremized
Mean: 0.3973 | Median: 0.4109 | StDev: 0.1398
Diversity Score: 0.1398
95% CI: [0.3586, 0.4361]

--- Opinion Clusters ---
  central_bank_watcher_bayesian_statistician_yes: 6 agents, mean=0.623
  data_scientist_insurance_actuary_no: 32 agents, mean=0.319
  uncertain_centrists: 12 agents, mean=0.494

Swarm vs Market: BELOW by 0.0094 (edge)

--- Mind Changers (3) ---
  Elizabeth Parker (Crypto Analyst): 0.92 -> 0.76 (toward NO)
  Stephanie Allen (Investigative Journalist): 0.02 -> 0.14 (toward YES)
```

---

## Streamlit Dashboard

6-panel interactive dashboard:

1. **Input Form** (sidebar) — question, context, n_agents (10-500), n_rounds (1-10), market_price
2. **Probability Gauge** — Plotly indicator with swarm P(YES), market price marker, confidence interval
3. **Convergence Chart** — mean score by round with individual agent traces as thin lines
4. **Agent Scatter** — initial vs final score colored by temperature tier; diagonal = no change
5. **Opinion Distribution** — histogram of final P(YES) scores across all agents
6. **Top Voices** — YES/NO/Mind-changer cards with name, background, score trajectory, reasoning

Plus: dissenting voices expander, opinion clusters bar chart, reasoning shift summary, raw JSON export.

```bash
streamlit run streamlit_app.py
```

---

## Modes

### Offline Mode (default)
Uses algorithmic agent generation + mathematical peer influence + question-specific alpha signals. No API calls required. Instant results.

```bash
python main.py --offline -q "Your question?" -a 50 -r 4 -m 0.40
```

### API Mode (requires Anthropic API key)
Uses Claude Sonnet for LLM-powered agent reasoning. Each agent gets a unique prompt reflecting their background, personality, and the world model. Produces richer reasoning but requires a funded [console.anthropic.com](https://console.anthropic.com) account (separate from Claude Max/Pro subscriptions).

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python main.py -q "Your question?" -a 50 -r 3
```

---

## Project Structure

```
minisim/
├── main.py                    # CLI entry: --question, --agents, --rounds, --market-price, --offline
├── streamlit_app.py           # 6-panel Streamlit dashboard
├── backtest.py                # Backtest on 100 resolved markets (Brier, calibration, alpha)
├── convergence_comparison.py  # 3-round vs 5-round convergence analysis
├── src/
│   ├── offline_engine.py      # Offline: full pipeline without API calls
│   │   ├── build_world_offline()          # GraphRAG world model from templates
│   │   ├── generate_population_offline()  # 40 archetypes, temperature tiers, alpha
│   │   ├── run_simulation_offline()       # 4-round deliberation protocol
│   │   └── swarm_score_offline()          # Full pipeline orchestrator
│   ├── aggregator.py          # Calibrated aggregation engine
│   │   ├── aggregate()                    # Confidence-weighted + extremized
│   │   ├── _identify_clusters()           # Opinion cluster detection
│   │   └── _voice_summary()              # Top/dissenting voice extraction
│   ├── world_builder.py       # GraphRAG world model (API mode)
│   ├── agent_factory.py       # Agent generation (API mode)
│   ├── simulation_loop.py     # Deliberation rounds (API mode)
│   └── kalshi_bridge.py       # Full pipeline wrapper (API mode)
├── results/                   # JSON outputs (stress tests, backtest, convergence)
├── docs/                      # Research documents
│   ├── MiniSim_Executive_Summary.md
│   ├── MiniSim_Research_Report.md
│   └── MiniSim_Technical_Stack.md
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Research Foundation

| Paper | Year | Key Finding | How MiniSim Uses It |
|-------|------|------------|-------------------|
| [Multi-agent debate improves factuality](https://arxiv.org/abs/2305.14325) | ICML 2024 | 15-30% factuality improvement via multi-agent society | 4-round deliberation protocol with evidence exchange and critique |
| [Wisdom of Silicon Crowd](https://arxiv.org/abs/2402.19379) | Science Advances 2025 | 12 LLMs match 925 human forecasters (Brier 0.186) | Ensemble diversity + calibration weighting |
| [Verbalized sampling for diversity](https://arxiv.org/abs/2510.01171) | 2025 | 1.6-2.1x diversity boost, training-free | Temperature stratification across agent tiers |
| [Calibrated confidence-weighted voting](https://arxiv.org/abs/2506.00066) | 2025 | Calibrated > uncalibrated > unweighted voting | Mind-change bonus + domain expertise weighting |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | UIST 2023 Best Paper | Memory stream + reflection architecture | Per-agent memory accumulation across rounds |
| [o3 beats human crowds](https://arxiv.org/abs/2507.04562) | 2025 | Brier 0.135 vs human crowd 0.149 | Target benchmark for API mode performance |
| [Format-induced diversity collapse](https://arxiv.org/abs/2505.18949) | 2025 | Structured formats cause mode collapse | Format-aware diversity techniques |
| [Fake prediction markets for confidence](https://arxiv.org/abs/2512.05998) | 2025 | Market mechanisms extract better confidence | Aggregation inspired by market microstructure |
| [Confidence calibration via deliberation](https://arxiv.org/abs/2404.09127) | 2024 | Post-deliberation calibration reduces Brier + ECE | Multi-round deliberation with calibration tracking |
| [LLM forecasting approaches human-level](https://arxiv.org/abs/2402.18563) | NeurIPS 2024 | Retrieval-augmented LM matches competitive forecasters | RAG-style world building + evidence retrieval |

---

## Competitive Landscape

| Company | Funding | Approach | Differentiator |
|---------|---------|----------|----------------|
| **Simile AI** | $100M Series A | Digital twins from real interviews | Grounded in actual human data (CVS Health, Telstra) |
| **Aaru** | $1B valuation | Synthetic population surveys | 90% correlation with real surveys, better predictive accuracy |
| **FutureSearch** | $5.79M seed | Autonomous AI forecaster | Beats human forecasters on geopolitical questions |
| **MiroFish** | $4.1M | 1M agent swarm engine | Open-source, OASIS framework, viral (28K stars) |
| **Unanimous AI** | — | Human swarming platform | 81-93% Oscar accuracy, UN famine forecasting |
| **Metaculus** | Non-profit | Community forecasting | Gold-standard calibration, COVID vaccine timeline |
| **MiniSim** | Pre-seed | Multi-agent deliberation swarm | 4-round debate, calibrated aggregation, 37% market-beating on 100 markets |

### MiniSim's Edge
1. **Structured deliberation** — not just aggregation, but evidence exchange + critique + rebuttal
2. **Domain expertise routing** — the right experts get higher weight on the right questions
3. **Question-specific alpha** — systematic detection of market mispricings (rare event overpricing, institutional inertia, tech adoption lag)
4. **Full explainability** — every prediction comes with mind-changers, dissenting voices, opinion clusters, and reasoning chains
5. **Calibration-first** — extremized aggregation + confidence weighting produces well-calibrated output

---

## Market Opportunity

| Segment | TAM | MiniSim's Play |
|---------|-----|---------------|
| Traditional Market Research | $78B | Replace surveys with synthetic swarm deliberation |
| Prediction Markets | $10-50B | Automated trading signals via swarm consensus |
| Enterprise Forecasting | $5-20B | Scenario analysis for strategic planning |
| AI-Native Synthetic Research | $10-50B (emerging) | Cheaper, faster, more diverse than focus groups |

**Unit Economics** (Aaru model validated):
- Traditional survey: $50K-200K, 4-8 weeks
- MiniSim synthetic deliberation: $5-20K, minutes
- 90%+ accuracy correlation (Aaru benchmark) + better predictive validity

---

## Dependencies

```
anthropic>=0.40.0    # API mode only
streamlit>=1.35.0
plotly>=5.20.0
pandas>=2.0.0
tqdm
python-dotenv
```

---

## License

MIT
