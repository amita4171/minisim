# MiniSim Refactoring Plan

Baseline: 141 fast tests passing, 38 Python files, ~9,100 lines of source code.

## Issues Found

### Duplication
1. `_get_client()` defined identically in 3 files (agent_factory.py, world_builder.py, simulation_loop.py)
2. `_safe_float()` / `_parse_price()` — same function in kalshi_client.py and polymarket_client.py
3. `_detect_category()` exists only in world_templates.py but is imported by 4+ files — OK, not duplicated

### Responsibility Violations
4. 14 top-level scripts — 5 are NOT in the unified CLI (calibration_at_scale.py, build_eval_dataset.py, resolve_metaculus.py, convergence_comparison.py, live_backtest.py)
5. `src/metaculus_client.py` is ORPHANED — not imported by any file

### Naming
6. Magic numbers in router.py (0.05, 0.10) and aggregator.py (0.05, 0.10, 0.15, 1.5) — should be named constants

### Module Organization
7. 14 top-level scripts clutter the root — could move runner scripts to a `scripts/` directory

### Abstraction
8. `_get_client()` lazy-load pattern repeated 3 times — should be a shared utility

## Workstreams (ordered by safety)

### WS1: Extract shared utilities (duplication fixes)
- Create `src/utils.py` with `safe_float()` and `get_anthropic_client()`
- Replace 3x `_get_client()` with shared version
- Replace 2x `_safe_float()/_parse_price()` with shared version
- **Risk**: Low — pure function extraction. Tests verify behavior.

### WS2: Named constants for magic numbers
- Create constants at top of router.py and aggregator.py
- `LOW_VARIANCE_THRESHOLD`, `HIGH_VARIANCE_THRESHOLD` already exist in router.py — good
- Add `MIND_CHANGE_THRESHOLD`, `EXTREMIZATION_ALPHA`, `SHIFT_THRESHOLD` to aggregator.py
- **Risk**: None — no behavior change.

### WS3: Wire orphaned metaculus_client.py
- `metaculus_bot.py` has its own inline API calls instead of using `src/metaculus_client.py`
- Refactor bot to use the client module
- **Risk**: Medium — bot is live. Must verify submission still works.
- **Decision**: Skip for now. Bot works. Document as follow-up.

### WS4: Add remaining scripts to CLI
- Add `calibrate-scale`, `build-dataset`, `resolve`, `convergence`, `live-backtest` subcommands
- **Risk**: Low — additive, doesn't change existing behavior.
- **Decision**: Skip — these are one-off scripts, not frequent commands. Document as optional.

### WS5: Move runner scripts to scripts/ directory
- **Decision**: Skip — would break imports and is cosmetic. Not worth the risk.

## Execution Plan

1. **WS1**: Extract `src/utils.py` — shared `safe_float()` + `get_anthropic_client()`
2. **WS2**: Named constants in aggregator.py
3. Run tests after each step
4. Commit after each workstream

## Out of Scope
- Moving top-level scripts to scripts/ (import breakage risk)
- Refactoring metaculus_bot.py to use metaculus_client.py (bot is live)
- Any feature additions or bug fixes
