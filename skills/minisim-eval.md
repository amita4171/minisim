---
name: MiniSim Evaluation
description: How to evaluate accuracy — Brier scores, calibration, gate checks
---

# MiniSim Evaluation Workflow

## Run Eval
```bash
# Offline baseline
python3 eval_runner.py --mode offline --agents 20 --rounds 2

# LLM mode (requires Ollama)
python3 eval_runner.py --mode llm-ollama --model qwen2.5:14b --agents 15 --rounds 2

# Anthropic API mode
python3 eval_runner.py --mode llm-anthropic --agents 15 --rounds 2
```

## Calibration at Scale (544 questions)
```bash
python3 calibration_at_scale.py --mode offline --agents 15 --rounds 2
```

## Head-to-Head Benchmark
```bash
python3 benchmark.py --model qwen2.5:14b --agents 10 --rounds 2
```

## Interpret Results

### Brier Score
- 0.00 = perfect
- 0.10 = excellent
- 0.25 = mediocre
- Lower is better

### ECE (Expected Calibration Error)
- 0.00 = perfectly calibrated
- Our current: 0.059 (beats market at 0.073)

### Gate Check
The eval runner prints "GATE: PASSED" or "GATE: FAILED":
- LLM must beat offline Brier by >= 15%
- If FAILED: fix prompts/aggregation before Phase 2

### Data Leakage Warning
Historical questions (HIST-001 to HIST-010) have known outcomes in the LLM's training data.
The LLM "predicts" 0.98 for Apple Vision Pro not because it's forecasting well, but because
it already knows. Real validation = Metaculus tournament (forward predictions).

## Key Metrics
| Metric | Offline | LLM (Qwen 14B) |
|--------|---------|-----------------|
| Brier (10 resolved) | 0.154 | 0.103 |
| ECE (544 questions) | 0.059 | TBD |
| Mode collapse (8 tests) | 8/8 pass | 2/6 pass |
| In expected range | 41% | 12% |

## Files
- `eval_runner.py` — main eval with --mode flag
- `calibration_at_scale.py` — 544-question calibration run
- `benchmark.py` — head-to-head: swarm vs single LLM vs market
- `results/eval_results_*.json` — saved results per mode
- `results/calibration_model_offline.json` — fitted Platt model
