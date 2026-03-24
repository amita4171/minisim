# MiniSim — Swarm Prediction Engine

A startup-grade swarm intelligence prediction engine that simulates N diverse digital humans deliberating across K structured rounds on any prediction question. Produces calibrated probability estimates that rival prediction market crowds.

**Wedge:** Prediction markets (Kalshi/Polymarket) as the first vertical, with a path to synthetic surveys and market research ($78B TAM).

**4 live data sources:** Kalshi, Polymarket, Manifold Markets, PredictIt — with cross-platform arbitrage detection.

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
    [Web Research / RAG]  (optional)
         |  Multi-perspective search: bull/bear/historical/expert/news
         |  Information asymmetry: each agent gets different research
         |
    [Agent Factory]
         |  40 archetypes: economists, traders, lawyers, journalists,
         |  VCs, military intel, pollsters, AI safety researchers...
         |  Temperature stratification: T=0.3 to T=1.2
         |  Domain expertise weighting per question
         |  Question-specific alpha signals
         |
    [4-Round Structured Deliberation]
         |  R1: Initial Forecast (independent)
         |  R2: Evidence Exchange (peer quality review)
         |  R3: Critique & Rebuttal (opponent pairing)
         |  R4: Updated Forecast (final revision)
         |
    [Calibrated Aggregator]
         |  60% calibrated confidence-weighted
         |  40% extremized (Metaculus-style, alpha=1.25)
         |  Mind-change bonus: flexible thinkers upweighted
         |  Domain experts get higher confidence weights
         |
    [Cross-Platform Aggregator]
         |  Compares swarm estimate vs 4 prediction markets
         |  Detects arbitrage when platforms disagree >5%
         |  Liquidity-weighted consensus across platforms
         |
    Output
         ├── swarm_probability_yes + 95% confidence interval
         ├── edge vs market price
         ├── cross-platform consensus + arbitrage signals
         ├── opinion clusters with descriptive labels
         ├── top YES/NO voices with reasoning excerpts
         ├── mind-changers (who changed and why)
         ├── dissenting voices with z-scores
         └── reasoning shift summary
```

---

## Live Data Sources

| Platform | Type | Coverage | Auth | Markets |
|----------|------|----------|------|---------|
| **Kalshi** | Real money (CFTC-regulated) | Economics, politics, crypto, weather | No auth for reads | ~600 active |
| **Polymarket** | Real money (crypto, Polygon) | Politics, geopolitics, tech | No auth for reads | ~100 active |
| **Manifold Markets** | Play money | Broadest: AI, science, culture, politics | No auth for reads | ~10K+ active |
| **PredictIt** | Real money (CFTC-authorized) | US politics: elections, policy, government | No auth for reads | ~250 active |

### Cross-Platform Features
- **Fuzzy question matching** — finds the same question across platforms using text similarity
- **Arbitrage detection** — flags when platforms disagree by >5% on the same question
- **Consensus probability** — liquidity-weighted average across all platforms that list a question
- **607 total markets** scanned in a single run across all 4 sources

### Arbitrage Examples Found
```
Spread=11% | Buy PredictIt@0.24 / Sell Manifold@0.35
  "Will Trump visit Russia during his term?"

Spread=10% | Buy Manifold@0.17 / Sell PredictIt@0.27
  "Will Trump/USA buy or acquire part of Greenland?"
```

---

## Opportunity Scanner

The scanner continuously monitors all 4 platforms for markets where the swarm disagrees with market price.

```bash
# One-time scan across all platforms
python scanner.py

# Continuous monitoring every 5 minutes
python scanner.py --watch --interval 300

# Scan only PredictIt with 3% edge threshold
python scanner.py --source predictit --edge 0.03

# Full scan with more agents
python scanner.py --source all --agents 30 --rounds 3 --max-markets 50
```

### Sample Scanner Output
```
OPPORTUNITIES FOUND: 9

[1] BUY NO | Edge: -10.3%
    Will Fernando Alonso finish on the podium at the 2026 F1 Bahrain Grand Prix?
    Market: 0.47 | Swarm: 0.37 | Source: polymarket

[2] BUY NO | Edge: -9.5%
    Bank of England rate hike in 2026?
    Market: 0.84 | Swarm: 0.75 | Source: polymarket

[3] BUY YES | Edge: +4.3%
    OpenAI receives federal backstop for infrastructure before July?
    Market: 0.05 | Swarm: 0.10 | Source: polymarket
```

---

## Track Record System

Every prediction is persisted with timestamps. When markets resolve, accuracy metrics are auto-computed.

```python
from src.track_record import TrackRecord

tr = TrackRecord()
tr.print_summary()

# Auto-resolve from Kalshi
tr.resolve_from_kalshi()

# Manual resolution
tr.resolve(pred_id=5, resolution=1.0)  # YES outcome

# Get metrics
metrics = tr.compute_metrics()
# Returns: Brier score, win rate, calibration curve, best/worst calls
```

The track record is integrated into the Streamlit dashboard with calibration charts and recent prediction history.

---

## Backtest Results

### Curated Backtest (100 Markets)

| Metric | Value |
|--------|-------|
| **Overall Swarm Brier** | 0.142 |
| **Overall Market Brier** | 0.132 |
| **Swarm Beat Market** | 37/100 (37%) |
| **Best Category** | Health (4/5 wins, 80%) |
| **Second Best** | Geopolitics (10/15 wins, 67%) |

### Live Kalshi Backtest (200 Real Markets)

| Metric | Value |
|--------|-------|
| **Overall Swarm Brier** | 0.054 |
| **Overall Market Brier** | 0.044 |
| **Markets Tested** | 200 real settled Kalshi markets |

### Where Swarm Adds Alpha
- **Rare event deflation**: correctly pushes down overpriced dramatic scenarios
- **Institutional inertia detection**: legislation/regulation slower than markets expect
- **Continuation bias**: trends persist (defense spending, AI investment, sanctions)
- **Domain expert upweighting**: matching archetypes get higher weight on their domain

---

## Key Technical Features

### 40 Agent Archetypes
Economists, traders, political analysts, journalists, VCs, central bankers, behavioral economists, geopolitical strategists, data scientists, hedge fund PMs, constitutional lawyers, supply chain analysts, military intel analysts, actuaries, pollsters, AI safety researchers, management consultants, short sellers, conspiracy skeptics, historical analogists, and more.

### Temperature Stratification ([arxiv 2510.01218](https://arxiv.org/abs/2510.01218))
| Tier | Temperature | Jitter Std | Role |
|------|-------------|------------|------|
| Analyst | 0.3 | 0.12 | Evidence-driven, precise estimates |
| Calibrator | 0.5 | 0.14 | Base-rate focused, moderate |
| Contrarian | 0.9 | 0.22 | Challenge consensus, wide variance |
| Creative | 1.2 | 0.24 | Explore unlikely scenarios |

### 4-Round Deliberation Protocol ([arxiv 2305.14325](https://arxiv.org/abs/2305.14325))
1. **Initial Forecast**: Independent probability estimate with evidence generation
2. **Evidence Exchange**: Agents share evidence, rate peer evidence quality (1-5)
3. **Critique & Rebuttal**: YES/NO agents paired for debate; disagreement typed (empirical vs methodological)
4. **Updated Forecast**: Final revision tracking mind-changes and reasoning shifts

### Calibrated Aggregation ([arxiv 2506.00066](https://arxiv.org/abs/2506.00066))
- **Confidence-weighted averaging** with mind-change bonus
- **Domain expertise bonus**: +0.20 confidence on domain-matched questions
- **Extremized aggregation**: Metaculus-style `p^alpha / (p^alpha + (1-p)^alpha)` where alpha=1.25
- **Combined**: 60% calibrated + 40% extremized

### Question-Specific Alpha Signals
- Rare event deflation (dramatic events overpriced by retail)
- Institutional inertia (legislation slower than markets expect)
- Technology adoption lag (deployment timelines underestimated)
- Fed anticipation (markets over-anticipate rate moves)
- Continuation bias (ongoing trends persist)

### Web Search / RAG Layer
- DuckDuckGo-powered multi-perspective research
- Bull case, bear case, historical, expert, news angles
- Information asymmetry: each agent gets different research bundles
- Contrarian agents get bear case; optimists get bull case

### Anti-Mode-Collapse
- Personality-driven convergence rates
- Contrarian agents resist consensus
- Temperature stratification ensures creative agents maintain wide distributions
- Diversity preserved > 0.15 std dev through 5 rounds

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a prediction (offline mode)
python main.py --offline \
  --question "Will the Fed cut rates in May 2026?" \
  --agents 50 --rounds 4 --market-price 0.40

# Run with web research
python main.py --offline --web-research \
  -q "Will AI replace 10% of white-collar jobs by 2028?" \
  -a 30 -r 4 -m 0.25

# Run the Streamlit dashboard
streamlit run streamlit_app.py

# Scan all 4 prediction platforms for opportunities
python scanner.py --source all --edge 0.03

# Continuous monitoring
python scanner.py --watch --interval 300

# Run backtest on 100 curated markets
python backtest.py

# Run live backtest on real Kalshi markets
python live_backtest.py --agents 30 --rounds 2

# Run convergence comparison
python convergence_comparison.py

# Check cross-platform arbitrage
python -c "from src.cross_platform import find_arbitrage; [print(a) for a in find_arbitrage()]"
```

---

## Streamlit Dashboard

6-panel interactive dashboard + track record:

1. **Input Form** (sidebar) — question, context, n_agents (10-500), n_rounds (1-10), market_price, web research toggle
2. **Probability Gauge** — Plotly indicator with swarm P(YES), market price marker, confidence interval
3. **Convergence Chart** — mean score by round with individual agent traces
4. **Agent Scatter** — initial vs final score colored by temperature tier
5. **Opinion Distribution** — histogram of final P(YES) scores
6. **Top Voices** — YES/NO/Mind-changer cards with reasoning excerpts
7. **Track Record** — running accuracy metrics, calibration chart, recent predictions

```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```
minisim/
├── main.py                    # CLI entry point
├── streamlit_app.py           # 7-panel Streamlit dashboard
├── scanner.py                 # Real-time opportunity scanner (4 platforms)
├── backtest.py                # Curated 100-market backtest
├── live_backtest.py           # Live backtest on real Kalshi data
├── convergence_comparison.py  # 3-round vs 5-round convergence study
├── src/
│   ├── offline_engine.py      # Core: full pipeline without API calls
│   ├── aggregator.py          # Calibrated aggregation engine
│   ├── cross_platform.py      # Cross-platform aggregator + arbitrage
│   ├── kalshi_client.py       # Kalshi API client
│   ├── polymarket_client.py   # Polymarket Gamma API client
│   ├── manifold_client.py     # Manifold Markets API client
│   ├── predictit_client.py    # PredictIt API client
│   ├── track_record.py        # Prediction persistence + accuracy tracking
│   ├── web_research.py        # Web search RAG layer
│   ├── world_builder.py       # GraphRAG world model (API mode)
│   ├── agent_factory.py       # Agent generation (API mode)
│   ├── simulation_loop.py     # Deliberation rounds (API mode)
│   └── kalshi_bridge.py       # Pipeline wrapper (API mode)
├── results/                   # JSON outputs
│   ├── backtest_results.json
│   ├── live_backtest_results.json
│   ├── convergence_comparison.json
│   ├── track_record.json
│   ├── scan_history.json
│   └── stress_test_*.json
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
| [Multi-agent debate improves factuality](https://arxiv.org/abs/2305.14325) | ICML 2024 | 15-30% factuality improvement | 4-round deliberation protocol |
| [Wisdom of Silicon Crowd](https://arxiv.org/abs/2402.19379) | Science Advances 2025 | 12 LLMs match 925 human forecasters | Ensemble diversity + calibration |
| [Verbalized sampling for diversity](https://arxiv.org/abs/2510.01171) | 2025 | 1.6-2.1x diversity boost | Temperature stratification |
| [Calibrated confidence-weighted voting](https://arxiv.org/abs/2506.00066) | 2025 | Calibrated > uncalibrated weighting | Mind-change bonus in aggregation |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | UIST 2023 Best Paper | Memory stream architecture | Per-agent memory accumulation |
| [o3 beats human crowds](https://arxiv.org/abs/2507.04562) | 2025 | Brier 0.135 vs human crowd 0.149 | Target benchmark |
| [Format-induced diversity collapse](https://arxiv.org/abs/2505.18949) | 2025 | Structured formats cause mode collapse | Format-aware diversity |
| [Confidence calibration via deliberation](https://arxiv.org/abs/2404.09127) | 2024 | Post-deliberation calibration reduces Brier | Multi-round calibration tracking |
| [LLM forecasting approaches human-level](https://arxiv.org/abs/2402.18563) | NeurIPS 2024 | RAG LM matches competitive forecasters | World building + evidence retrieval |
| [MA-RAG multi-agent retrieval](https://arxiv.org/abs/2505.20096) | 2025 | Specialized agents outperform standalone LLMs | Information asymmetry across agents |

---

## Competitive Landscape

| Company | Funding | Approach | Differentiator |
|---------|---------|----------|----------------|
| **Simile AI** | $100M Series A | Digital twins from real interviews | Grounded in actual human data |
| **Aaru** | $1B valuation | Synthetic population surveys | 90% correlation with real surveys |
| **FutureSearch** | $5.79M seed | Autonomous AI forecaster | Beats humans on geopolitical Qs |
| **MiroFish** | $4.1M | 1M agent swarm engine | Open-source, OASIS framework |
| **Unanimous AI** | — | Human swarming platform | 81-93% Oscar accuracy |
| **Metaculus** | Non-profit | Community forecasting | Gold-standard calibration |
| **MiniSim** | Pre-seed | Multi-agent deliberation + cross-platform | 4-round debate, 4 data sources, arbitrage |

### MiniSim's Edge
1. **4-platform aggregation** — only system pulling Kalshi + Polymarket + Manifold + PredictIt simultaneously
2. **Cross-platform arbitrage** — detects price discrepancies across platforms
3. **Structured deliberation** — evidence exchange + critique, not just aggregation
4. **Domain expertise routing** — right experts weighted on right questions
5. **Full explainability** — mind-changers, dissenting voices, opinion clusters, reasoning chains
6. **Question-specific alpha** — systematic detection of market mispricings

---

## Market Opportunity

| Segment | TAM | MiniSim's Play |
|---------|-----|---------------|
| Traditional Market Research | $78B | Replace surveys with synthetic deliberation |
| Prediction Markets | $10-50B | Automated trading signals via swarm consensus |
| Enterprise Forecasting | $5-20B | Scenario analysis for strategic planning |
| AI-Native Synthetic Research | $10-50B (emerging) | Cheaper, faster than focus groups |

---

## Modes

### Offline Mode (default)
Algorithmic agent generation + mathematical peer influence + alpha signals. No API calls. Instant results.

```bash
python main.py --offline -q "Your question?" -a 50 -r 4 -m 0.40
```

### API Mode (requires Anthropic API key)
LLM-powered agent reasoning via Claude Sonnet. Requires a funded [console.anthropic.com](https://console.anthropic.com) account (separate from Claude Max/Pro).

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python main.py -q "Your question?" -a 50 -r 3
```

---

## Dependencies

```
anthropic>=0.40.0    # API mode only
streamlit>=1.35.0
plotly>=5.20.0
pandas>=2.0.0
requests>=2.28.0
tqdm
python-dotenv
```

---

## License

MIT
