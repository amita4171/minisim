---
name: MiniSim Debugging
description: How to debug prediction issues — diversity, routing, calibration, mode collapse
---

# MiniSim Debugging Guide

## Problem: Predictions cluster at 0.50 ("I don't know")
**Cause:** LLM defaulting to safe middle ground.
**Fix:** Check anchor prompt. Ensure context is being passed. The improved prompt in llm_engine.py has explicit probability scale guidance.
```bash
# Test anchor independently
python3 -c "
from src.llm_engine import LLMEngine, ANCHOR_PROMPT
engine = LLMEngine(model='qwen2.5:14b')
r = engine.generate_json(ANCHOR_PROMPT.format(question='Will X?', context='Strong evidence...'))
print(r)
"
```

## Problem: Everything snaps to 5% for unlikely events
**Cause:** Old anchor prompt lacked granularity in low-probability range.
**Fix:** Anchor prompt now has explicit scale: 1-2% (impossible) vs 3-5% (extremely unlikely) vs 8-12% (unlikely but precedented).
**Verify:** Run a question that should be 1-2% vs one that should be 8-12% and check they differ.

## Problem: Mode collapse (low diversity, stdev < 0.05)
**Cause:** Agents converging to same answer despite different personas.
**Check:**
```bash
python3 -c "
from src.offline_engine import swarm_score_offline
r = swarm_score_offline('Your question?', n_agents=20, rounds=2, market_price=0.50)
print(f'Diversity: {r[\"diversity_score\"]:.3f}')
for a in r['agents'][:5]:
    print(f'  {a[\"background_category\"]}: {a[\"final_score\"]:.2f}')
"
```
**Fix options:**
- Increase temperature spread (contrarian T=0.9 → T=1.2)
- Strengthen contrarian prompts ("differ by at least 0.15")
- Reduce deliberation rounds (convergence kills diversity)
- Use router: if initial stdev < 0.05, skip deliberation

## Problem: Single LLM beats swarm
**Cause:** Deliberation dilutes correct extreme estimates toward center.
**Evidence:** Benchmark showed single LLM Brier 0.037 vs swarm 0.041.
**Fix:** Router dispatches easy questions to single_llm. Extremized aggregation (alpha=1.5) amplifies correct directional consensus.

## Problem: Calibration not applied
**Check:**
```bash
python3 -c "
from src.offline_engine import swarm_score_offline
r = swarm_score_offline('Test?', n_agents=10, rounds=1, market_price=0.50)
print(f'Raw: {r[\"swarm_probability_raw\"]:.3f}')
print(f'Calibrated: {r[\"swarm_probability_yes\"]:.3f}')
print(f'Applied: {r[\"calibration_applied\"]}')
"
```
**If False:** Check that results/calibration_model_offline.json exists and is valid.
**Refit:** `python3 -c "from src.calibration import fit_calibration_from_backtest; fit_calibration_from_backtest()"`

## Problem: Ollama errors (500, timeout)
**Cause:** GPU contention from concurrent processes (bot + eval running simultaneously).
**Fix:** Kill one process, or reduce concurrency:
```bash
export MINISIM_CONCURRENCY=1
```
**Check Ollama:**
```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool
ollama ps  # shows GPU usage
```

## Problem: Bot missed tournament questions
**Cause:** Questions only open for ~1.5 hours. Bot interval too long or Mac was sleeping.
**Check:**
```bash
cat results/forecasted_questions.json | python3 -c "import sys,json; print(len(json.load(sys.stdin)))"
# Compare against total questions in tournament
```
**Fix:** Reduce interval to 900 (15min) or deploy to cloud.

## Problem: Metaculus API returns 403
**Cause:** Token not set or expired.
**Check:** `echo $METACULUS_BOT_TOKEN`
**Fix:** Set in environment or .env file.

## Problem: Data leakage in eval
**Cause:** LLM training data includes 2024 outcomes. HIST-001 to HIST-010 are contaminated.
**Accept:** Cannot fix. Real validation = forward-tested Metaculus tournament predictions.
**Workaround:** Use only post-cutoff questions or Metaculus tournament questions for accuracy claims.

## Running Tests
```bash
pytest -m "not slow"     # 74 fast tests, 0.3s
pytest                   # 81 total, needs Ollama, ~4min
pytest tests/test_router.py -v  # just router tests
```
