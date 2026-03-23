# MiniSim — Swarm Prediction Engine

A startup-grade swarm intelligence prediction engine that simulates N diverse digital humans deliberating across K structured rounds on any prediction question. Produces calibrated probability estimates for prediction markets (Kalshi/Polymarket).

Inspired by [MiroFish](https://github.com/666ghj/MiroFish) (28K+ stars), [Generative Agents](https://github.com/joonspk-research/generative_agents) (Park et al.), and the [Wisdom of Silicon Crowd](https://arxiv.org/abs/2402.19379) (Science Advances 2025).

## Architecture

```
Question + Context
       |
  [World Builder] -----> GraphRAG knowledge graph (entities, relationships, pressures)
       |
  [Agent Factory] -----> N diverse agents with backgrounds, personalities, temperature tiers
       |
  [4-Round Deliberation Protocol]
       |  Round 1: Initial Forecast (independent)
       |  Round 2: Evidence Exchange (peer review)
       |  Round 3: Critique & Rebuttal (opponent pairing)
       |  Round 4: Updated Forecast (final revision)
       |
  [Calibrated Aggregator] -> Confidence-weighted + extremized P(YES)
       |
  Output: probability, CI, clusters, top voices, mind-changers
```

## Key Features

- **40 agent archetypes** spanning economics, politics, technology, law, military intelligence, and more
- **Temperature stratification** (arxiv 2510.01218): analysts (T=0.3), calibrators (T=0.5), contrarians (T=0.9), creatives (T=1.2)
- **4-round structured deliberation** based on multi-agent debate research (arxiv 2305.14325)
- **Calibrated aggregation**: confidence-weighted + extremized (Metaculus-style, arxiv 2506.00066)
- **Anti-mode-collapse**: personality-driven convergence rates, contrarian agents resist consensus
- **Full explainability**: mind-changers, dissenting voices (z-scores), opinion clusters, reasoning shift summaries

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

# Run backtest on 30 resolved markets
python backtest.py

# Run convergence comparison (3-round vs 5-round)
python convergence_comparison.py
```

## Modes

### Offline Mode (default for now)
Uses algorithmic agent generation + mathematical peer influence. No API calls required. Runs under Claude Code with a Claude Max subscription.

```bash
python main.py --offline -q "Your question?" -a 50 -r 4
```

### API Mode (requires Anthropic API key)
Uses Claude Sonnet for LLM-powered agent reasoning. Requires a funded [console.anthropic.com](https://console.anthropic.com) account (separate from Claude Max subscription).

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python main.py -q "Your question?" -a 50 -r 3
```

## Streamlit Dashboard

6-panel dashboard with:
1. **Input form** — question, agents, rounds, market price
2. **Probability gauge** — swarm P(YES) with market price reference
3. **Convergence chart** — mean score by round with individual agent traces
4. **Agent scatter** — initial vs final score, colored by temperature tier
5. **Opinion distribution** — histogram of final scores
6. **Top voices** — YES/NO voices + mind-changers + dissenting voices

## Backtest Results

On 30 resolved prediction questions (curated):
- **Swarm Brier Score: 0.246** (target: ≤ 0.25)
- Best on low-probability NO questions (AI-generated movie award, AGI announcement)
- Limitation: offline mode clusters predictions around 0.35-0.45; API mode with real LLM reasoning would enable confident directional calls

## Project Structure

```
minisim/
├── main.py                    # CLI entry point
├── streamlit_app.py           # Streamlit dashboard
├── backtest.py                # Backtest on resolved markets
├── convergence_comparison.py  # 3-round vs 5-round analysis
├── src/
│   ├── world_builder.py       # GraphRAG world model (API mode)
│   ├── agent_factory.py       # Agent generation (API mode)
│   ├── simulation_loop.py     # Deliberation rounds (API mode)
│   ├── aggregator.py          # Calibrated aggregation engine
│   ├── kalshi_bridge.py       # Full pipeline wrapper (API mode)
│   └── offline_engine.py      # Offline mode: full pipeline without API calls
├── results/                   # JSON outputs
├── docs/                      # Research documents
└── requirements.txt
```

## Research Foundation

| Paper | Key Finding | How MiniSim Uses It |
|-------|------------|-------------------|
| [Multi-agent debate](https://arxiv.org/abs/2305.14325) (ICML 2024) | 15-30% factuality improvement | 4-round deliberation protocol |
| [Wisdom of Silicon Crowd](https://arxiv.org/abs/2402.19379) (Science Advances 2025) | 12 LLMs match 925 human forecasters | Ensemble diversity + calibration |
| [Verbalized sampling](https://arxiv.org/abs/2510.01171) | 1.6-2.1x diversity boost | Temperature stratification |
| [Calibrated confidence voting](https://arxiv.org/abs/2506.00066) | Calibrated > uncalibrated weighting | Mind-change bonus in aggregation |
| [Generative Agents](https://arxiv.org/abs/2304.03442) (UIST 2023 Best Paper) | Memory stream architecture | Per-agent memory accumulation |

## Competitive Landscape

- **Simile AI** ($100M Series A) — digital twins from real interviews
- **Aaru** ($1B valuation) — synthetic surveys, 90% correlation with real data
- **FutureSearch** ($5.79M seed) — autonomous AI forecaster
- **MiroFish** (28K GitHub stars) — 1M agent swarm engine

## Dependencies

```
anthropic>=0.40.0    # API mode only
streamlit>=1.35.0
plotly>=5.20.0
pandas>=2.0.0
tqdm
python-dotenv
```

## License

MIT
