---
name: MiniSim Prediction
description: How to run a swarm prediction — from question to calibrated probability
---

# MiniSim Prediction Workflow

## Quick Prediction (Offline, instant)
```bash
python3 main.py --offline -q "Will X happen?" -a 20 -r 2 -m 0.40
```

## Smart Prediction (Router decides single LLM vs swarm)
```bash
python3 main.py --smart --model qwen2.5:14b -q "Will X happen?" -a 15
```

## Full LLM Swarm (always deliberate)
```bash
python3 main.py --llm --model qwen2.5:14b -q "Will X happen?" -a 15 -r 3
```

## Pipeline
1. **World Builder** → extracts entities, pressures, timeline from context
2. **Context-to-Anchor** → LLM reads context, estimates base rate (if no market price)
3. **Agent Factory** → generates N agents with diverse backgrounds + temperature tiers
4. **Router** → measures initial agent variance:
   - stdev < 0.05 → single LLM call (agents agree)
   - stdev 0.05-0.10 → 1 round of light deliberation
   - stdev > 0.10 → full 3-round swarm deliberation
5. **Aggregator** → 40% confidence-weighted + 60% extremized (alpha=1.5)
6. **Calibration** → Platt scaling correction from fitted model
7. **Output** → P(YES), CI, top voices, mind-changers, diversity score

## Key Files
- `src/router.py` — variance-based routing
- `src/llm_engine.py` — LLM interface (Ollama/Anthropic)
- `src/llm_simulation.py` — multi-agent deliberation
- `src/aggregator.py` — calibrated aggregation + Platt correction
- `src/offline_engine.py` — algorithmic fallback (no LLM needed)

## Gotchas
- Ollama concurrency=2 on Apple Silicon (Metal GPU contention at 5+)
- Deliberation HURTS on easy questions — router addresses this
- Calibration model at results/calibration_model_offline.json must exist for Platt correction
- Token cap: 512 for initial, 256 for deliberation (prevents runaway generation)
