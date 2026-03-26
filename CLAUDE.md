# MiniSim — Claude Configuration

## IDENTITY

MiniSim is a swarm intelligence prediction engine that simulates diverse digital humans deliberating on prediction questions. Python 3.9+, runs locally on Apple M4 Pro with Ollama (qwen2.5:14b). Competes in the Metaculus Spring 2026 AIB tournament ($50K prize pool) with 50 live forecasts.

**Stack**: Python, SQLite, FastAPI, Ollama, Streamlit
**Architecture**: Modular monolith — `src/` with 5 sub-packages (core, agents, markets, research, db)
**Entry points**: api.py, cli.py, main.py, metaculus_bot.py, scanner.py, streamlit_app.py
**Tests**: 148 fast + 9 slow (Ollama-dependent), pytest with `@pytest.mark.slow`

## SUMMARY OF RULES

<!-- Updated automatically as rules are added -->
- **Calibration**: Don't add complexity that pushes predictions away from market consensus. Alpha=1.0 (no extremization). Every calibration layer has historically made things worse.
- **LLM Calls**: Cap per-agent tokens. Long runs are fine if quality is good. Concurrency=2 is optimal for M4 Pro.
- **Workflow**: Every change → commit + push + README update. Frequent commits throughout work.

## SESSION PROTOCOL

### Kick-off
1. Check `~/.claude/projects/-Users-amitagondi-Desktop-projects-minisim/memory/MEMORY.md` for context
2. Check `results/minisim.db` for current prediction count and resolution status
3. Check if Ollama is running: `curl -s localhost:11434/api/tags | head -1`
4. Note any pending blockers from memory

### Wrap-up
1. Run `python -m pytest tests/ -m "not slow" --tb=short -q` — must pass
2. Commit all changes with descriptive messages
3. Push to remote
4. Update README.md if functionality changed
5. Update memory files if project state changed

### Retro
1. Check Brier scores: `python scripts/calibration_report.py --source db`
2. Review any new resolutions available
3. Check if alpha sweep results have changed
4. Review churn hotspots in git log

## CONVENTIONS

### Code Style
- Python 3.9+ with `from __future__ import annotations`
- Type hints on function signatures, not local variables
- No docstrings required on internal helpers; docstrings on public API
- `sys.path.insert(0, ...)` at top of scripts for import resolution

### Imports
- New code: `from src.core.aggregator import aggregate`
- Old code still works: `from src.aggregator import aggregate` (backward compat via sys.modules in src/__init__.py)
- CI checks old-style imports — don't break them

### Error Handling
- External API calls: try/except with retry logic (see metaculus_bot.py)
- Internal functions: let exceptions propagate, catch at entry points
- Database: `Database` class handles its own connection management

### Testing
- Fast tests: `pytest tests/ -m "not slow"` — no external dependencies
- Slow tests: `pytest tests/ -m slow` — require Ollama running
- Test files mirror source: `src/core/aggregator.py` → `tests/test_aggregator.py`
- Use `monkeypatch` for LLM/API mocking, not mock libraries

### File Organization
```
src/
├── core/       (engine, aggregator, calibration, LLM, router)
├── agents/     (archetypes, alpha, world building, simulation)
├── markets/    (API clients, cross-platform, arbitrage)
├── research/   (web research, data feeds, EDGAR)
├── db/         (database, track record)
├── prompts/    (7 text/JSON template files)
├── data/       (7 YAML data files)
└── utils.py
scripts/        (runner scripts — never imported by src/)
tests/          (pytest test files)
```

### Named Constants
- All magic numbers must be named constants at module top
- `aggregator.py`: EXTREMIZATION_ALPHA, CONFIDENCE_WEIGHT, etc.
- `offline_engine.py`: ARCHETYPE_MEAN_BIAS, DEVIATION_AMPLIFIER, etc.
- `router.py`: LOW_VARIANCE_THRESHOLD, HIGH_VARIANCE_THRESHOLD

### Database
- SQLite at `results/minisim.db`
- `Database` class in `src/db/database.py` with `log_prediction()`, `resolve()`, `get_metrics()`
- Brier scores computed on resolution via `db.resolve(pred_id, outcome)`

## PROJECT-SPECIFIC WARNINGS

INITIAL RULES — will be refined as the project evolves.

### Calibration Over-correction (CRITICAL)
NEVER add a calibration layer without testing it on resolved predictions first.
Why: Every calibration layer added so far (isotonic, extremization at 1.5) has
made predictions worse. The model consistently overestimates its information
advantage over the market. Current alpha=1.0 (no extremization) is validated
as optimal on 6 resolved questions + 10 eval questions.

### Tail Risk Floor
NEVER clamp tail-risk probabilities below 10%.
Why: The old 5% floor was too aggressive — it said "almost impossible" for events
that had ~12% base rates. De-extremizing from alpha=1.5 revealed raw probabilities
around 12.3% for these questions.

### Market Price Trust
ALWAYS pull toward market consensus, not away from it.
Why: The shrinkage approach (pull toward market) outperforms the extremization
approach (push away from 50%). When swarm disagrees with market, the market is
usually right.

## DETAILED RULES

### Aggregation & Calibration
<!-- Rules about how predictions are combined and calibrated -->
- EXTREMIZATION_ALPHA = 1.0 (no extremization) — validated March 2026
- Confidence-weighted average (60%) + extremized average (40%) blend
- Mind-change bonus for agents who update based on evidence

### Metaculus Tournament
<!-- Rules about tournament submission and strategy -->
- Bot token in .env as METACULUS_BOT_TOKEN
- submit_forecast() in metaculus_bot.py takes question_id (ticker), not DB id
- Rate limit: 2s between API calls to avoid 429s
- Always clamp probabilities to [0.001, 0.999] before submission

### LLM Integration
<!-- Rules about Ollama/LLM usage -->
- Ollama at localhost:11434, model qwen2.5:14b
- Concurrency=2 optimal for M4 Pro GPU (54 tok/s)
- Cap per-agent tokens to control cost/time
- Offline engine (no LLM) for fast iteration and testing

### Data Pipeline
<!-- Rules about market data, research, external APIs -->
- 4 market sources: Kalshi, Polymarket, Manifold, PredictIt
- Tavily for web research, FRED for economic data, SEC EDGAR for filings
- External API calls always need timeout and retry logic

## META

### How to Write Good Rules
1. Start with ALWAYS, NEVER, PREFER, or AVOID
2. Include WHY — the incident or evidence that motivated the rule
3. Include SCOPE — which files or areas it applies to
4. Keep rules under 3 lines. If you need more, it's a design doc, not a rule.
5. Rules must be falsifiable — "write good code" is not a rule

### Rule Categories for MiniSim
- Aggregation & Calibration (src/core/aggregator.py, src/core/calibration.py)
- Metaculus Tournament (metaculus_bot.py, scripts/resolve_*.py)
- LLM Integration (src/core/llm_engine.py, src/core/llm_simulation.py)
- Data Pipeline (src/markets/*, src/research/*)
- Testing (tests/*)

### Memory Hierarchy
1. CLAUDE.md — conventions and rules (loaded every session)
2. .claude/rules/ — scoped rules (loaded when matching files are touched)
3. MEMORY.md — project-scoped index in ~/.claude/projects/
4. Individual memory files — detailed context per topic

### Self-Improvement Loop
When a mistake is discovered:
1. Identify the root cause (not just the symptom)
2. Abstract to a general pattern
3. Write a rule in CLAUDE.md or .claude/rules/ with evidence
4. If the same mistake class recurs, escalate to a .claude/rules/ file with path scoping
