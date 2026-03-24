# MiniSim Engineering Notes

Everything learned building a swarm prediction engine from scratch in 2 days. Covers architecture decisions, what worked, what failed, technical gotchas, and lessons for anyone building a similar system.

---

## Table of Contents
1. [Architecture Decisions](#architecture-decisions)
2. [What Worked](#what-worked)
3. [What Failed](#what-failed)
4. [LLM Integration Lessons](#llm-integration-lessons)
5. [Aggregation & Calibration](#aggregation--calibration)
6. [Multi-Platform API Integration](#multi-platform-api-integration)
7. [Performance & Concurrency](#performance--concurrency)
8. [Testing & Quality](#testing--quality)
9. [Product Findings](#product-findings)
10. [Security & Operational Issues](#security--operational-issues)
11. [Forward Testing vs Backtesting](#forward-testing-vs-backtesting)
12. [Circuit Breakers & Failure Modes](#circuit-breakers--failure-modes)
13. [Race Conditions](#race-conditions)
14. [Technical Debt Inventory](#technical-debt-inventory)

---

## Architecture Decisions

### Offline-First Design
Built the entire engine to run without any API calls first (`offline_engine.py`). This meant:
- Could iterate on architecture without API costs
- Could run stress tests instantly (544 questions in 1 second)
- Had a baseline to compare LLM mode against
- **Lesson**: Always build the offline/mock version first. It forces you to separate the reasoning logic from the LLM integration.

### Module Split
`offline_engine.py` started at 1,700 lines — way too much. Split into:
- `archetypes.py` (113 lines): Background data, personality traits, name generation
- `world_templates.py` (400 lines): World building, pressures, evidence templates
- `alpha.py` (111 lines): Question-specific alpha signals, domain expertise
- `offline_engine.py` (459 lines): Thin orchestrator

**Key**: kept backward compatibility by re-exporting all symbols from `offline_engine.py` so no other file had to change imports.

### Router Pattern
The variance-based router was the most important architectural decision. Instead of always running the full swarm:
- Low variance (stdev < 0.05): single LLM call — agents agree, deliberation adds noise
- Medium variance (0.05-0.10): 1 round of light deliberation
- High variance (> 0.10): full 3-round swarm

**Problem**: These thresholds were calibrated on only 10 questions. They need validation on 500+ before they're trustworthy.

---

## What Worked

### Context-to-Anchor Mapping
When no market price is given, the LLM reads the seed context and estimates a base rate. Before this fix, the eval score was 18% in-range. After: 41%. The LLM correctly anchored "Apple Vision Pro shipping Feb 2024" to P=0.95 and "China invade Taiwan" to P=0.05.

### Temperature Stratification
Assigning different temperatures to different agent archetypes (analyst=0.3, contrarian=0.9, creative=1.2) produced genuine diversity. Without this, all agents converge to the same answer regardless of persona.

### Extremized Aggregation (alpha=1.5)
The swarm's predictions are directionally correct but too moderate — deliberation pulls everything toward center. Extremizing with alpha=1.5 pushes 0.83→0.90, 0.16→0.08. This single change cut Brier from 0.041 to 0.017 on the benchmark.

### Anti-Mode-Collapse Prompts
Key prompt techniques that actually worked:
- "Do NOT default to 0.50 — take a position based on your background"
- Contrarian agents: "If peers are converging, resist. Your estimate should differ by at least 0.15"
- Per-archetype nudges: analysts demand data, contrarians argue against consensus
- Including agent names in system prompts for stronger persona anchoring

### Fee-Aware Arbitrage
Cross-platform arbitrage is meaningless without accounting for fees. Kalshi (1% taker), Polymarket (2% taker), PredictIt (5% profit + 10% withdrawal) have very different cost structures. A 5% spread on Kalshi-Polymarket is profitable (break-even at 3%), but the same spread on PredictIt-anything is not.

---

## What Failed

### Multi-Round Deliberation (on Easy Questions)
The benchmark proved: **a single LLM call beats a 10-agent swarm on easy questions** (Brier 0.037 vs 0.041). Deliberation dilutes correct extreme estimates toward the center. The fix was the router — but the core finding stands: deliberation hurts when agents already agree.

### Offline Engine Diversity
The offline engine produces predictions clustered around 0.35-0.45 regardless of question. The "alpha signals" (keyword-based heuristics) help a bit but can't replace actual reasoning. The LLM mode is 69% better on Brier.

### Metaculus API Access
Metaculus blocks unauthenticated requests (Cloudflare). Even with a bot account token, resolution values and community predictions are gated behind the "Bot Benchmarking" tier that requires a separate email request. This blocked the most important eval: comparing MiniSim against the Metaculus community prediction.

### Python 3.9 Compatibility
Modern type syntax (`float | None`, `list[dict]`) doesn't work on Python 3.9. Had to add `from __future__ import annotations` to every file. Caught this on the first test run.

### Output Buffering in Background Tasks
Python doesn't flush stdout when running as a background process. Background LLM tasks showed zero output until completion. This made debugging nearly impossible for long-running jobs.

---

## LLM Integration Lessons

### Ollama on Apple Silicon
- M4 Pro runs Llama 3.1 8B at **54 tokens/sec** (100% GPU Metal acceleration)
- Qwen 2.5 14B at **27 tokens/sec** — half the speed but much smarter
- **Concurrency**: 2 parallel calls is optimal for Apple Silicon GPU. 5+ causes Metal contention and actually slows things down.
- A 15-agent x 2-round run takes ~4 min on 8B, ~20 min on 14B

### Client Instantiation Bug
The Anthropic client was created inside `_generate_anthropic()` on every call. This causes:
- Connection churn (new TLS handshake per call)
- Latency spikes
- **Fix**: Initialize once in `__init__`. The SDK client is thread-safe.

### Temperature Passthrough Bug
The temperature parameter was accepted by `_generate_anthropic()` but **never forwarded** to `client.messages.create()`. All API calls ran at default temperature. Analyst agents (T=0.3) and contrarian agents (T=0.9) were identical. Silent bug — nothing crashes, just bad diversity.

### Retry Logic
Without retries, a single 429 or timeout kills the entire simulation. Added exponential backoff with jitter: 1s, 2s, 4s. Retryable errors: 429, timeout, 500, 502, 503, "overloaded".

### JSON Parsing
LLMs don't reliably produce valid JSON, especially smaller models. The parsing pipeline:
1. Try `json.loads(text)` directly
2. Strip markdown fences (` ```json...``` `)
3. Extract first `{...}` substring and parse that
4. Fall back to offline engine values

Ollama's `format: "json"` flag helps but doesn't guarantee valid output.

### Fallback Transparency
When an LLM call fails, the agent silently falls back to offline engine distributions. Added `from_llm` boolean to each agent and `agents_from_llm`/`agents_from_fallback` counts. A customer paying for LLM-quality predictions must know which agents used real reasoning.

---

## Aggregation & Calibration

### Confidence-Weighted vs Extremized
Tested 7 aggregation strategies on 10-question benchmark:
1. Single LLM only: Brier 0.037
2. Current swarm (deliberation): 0.041
3. Single + Swarm average: 0.037
4. **Extremized swarm (alpha=1.5): 0.017** ← WINNER
5. 70% single + 30% swarm: 0.036
6. Single + swarm adjust ±5%: 0.037
7. Pick more extreme: 0.028

The extremized swarm beat everything by a wide margin. The insight: the swarm IS directionally correct, it's just too moderate.

### Platt Scaling Calibration
Fitted on 544-question dataset:
- Platt parameters: a=1.47, b=0.09
- Pre-correction ECE: 0.259 (poor)
- Post-correction ECE: 0.059 (beats market at 0.073)
- Key bias: MiniSim under-predicts YES for mid/high probability events (says 0.50, actual resolves at 0.67)

**Critical gap**: The calibration model is fitted but never applied at inference time. Neither `api.py` nor `router.py` calls `CalibrationTransformer.transform()` before returning predictions.

### Floating-Point Histogram Bug
Histogram bucketing used `i/10` for bucket boundaries. Due to floating-point precision, `3/10 = 0.30000000000000004`, so a score of exactly 0.3 fell into TWO buckets. Fixed with `round(s, 6)` comparisons.

---

## Multi-Platform API Integration

### Kalshi
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`
- No auth needed for reads
- Rate limiting: strict, needs 1s delay between pages
- 429 responses need 2s retry
- Mostly sports and crypto — filter heavily for interesting markets
- Market prices in `yes_bid_dollars`/`yes_ask_dollars` (string format)

### Polymarket
- Gamma API: `https://gamma-api.polymarket.com`
- No auth for reads
- `outcomePrices` is a JSON string inside JSON — needs double parsing
- Prices are probability-as-price (0.65 = 65% implied probability)
- `closed: true` with near-0 or near-1 price indicates resolution

### Manifold Markets
- Base URL: `https://api.manifold.markets`
- Best coverage: 10K+ markets, great for resolved binary questions
- `probability` field is directly the probability (no parsing needed)
- `resolution`: "YES", "NO", "MKT", "CANCEL"
- Play money — not valid for real arbitrage

### PredictIt
- Single endpoint: `https://www.predictit.org/api/marketdata/all/`
- Returns ALL markets in one call (no pagination)
- Nested structure: markets → contracts
- High fees: 5% on profits + 10% withdrawal
- Limited to US politics

### Metaculus
- Base URL: `https://www.metaculus.com/api`
- Requires auth: `Authorization: Token <token>`
- New API uses `/api/posts/` not `/api2/questions/`
- Resolution values and community predictions **gated** behind Bot Benchmarking tier
- Bot accounts created in Settings > My Forecasting Bots > Create a Bot
- Tournament project IDs: `spring-aib-2026`, `minibench`

### Cross-Platform Question Matching
Used `difflib.SequenceMatcher` for fuzzy matching with threshold 0.55. Works well for exact rephrasing but misses semantic equivalence ("Will Trump visit Russia?" vs "Trump-Russia meeting"). A sentence embedding approach would be better but adds a dependency.

---

## Performance & Concurrency

### Ollama Concurrency on Apple Silicon
- Metal GPU handles 1-2 concurrent requests well
- 5+ concurrent requests causes GPU contention — each request slows down
- Optimal: `concurrency=2` for Apple Silicon, `concurrency=5` for CPU-only
- `ThreadPoolExecutor` works fine — Ollama requests are I/O bound

### Token Limits for Speed
Deliberation responses don't need 1024 tokens. Capping at 256 tokens for deliberation rounds prevents runaway generation where one agent produces a 2000-token response and blocks the queue.

### Offline Engine Speed
544 questions x 15 agents x 2 rounds = **1 second** total. The offline engine is useful for rapid iteration on aggregation logic.

---

## Testing & Quality

### Test Coverage
52 tests across 7 modules:
- `test_archetypes.py`: Background data integrity, name generation
- `test_aggregator.py`: Probability bounds, clustering, extremization, histogram
- `test_alpha.py`: Rare event deflation, domain expertise matching
- `test_cross_platform.py`: Fuzzy matching, dedup, cross-listing
- `test_database.py`: CRUD, Brier computation, resolution
- `test_offline_engine.py`: World building, anchor effect, diversity preservation
- `test_calibration.py`: Platt fitting, ECE, save/load

### What's NOT Tested
- API endpoints (no integration tests for FastAPI)
- LLM simulation (would need Ollama running)
- Platform API clients (would need live network)
- End-to-end pipeline (prediction → resolution → calibration update)
- Router thresholds (validated on 10 questions only)

### CI Pipeline
GitHub Actions on push/PR, Python 3.9/3.11/3.12 matrix. Import checks verify all modules load without errors.

---

## Product Findings

### Single LLM Beats Swarm on Easy Questions
The most important finding. On questions with clear answers (Apple headset release, China invasion), one LLM call at T=0.3 gives a better prediction than 10 agents deliberating. The swarm regresses extreme (correct) estimates toward the center.

### Swarm Beats Single LLM on Hard Questions
On genuinely uncertain questions (OpenAI video model release, S&P 5000 threshold), the swarm adds value through diversity of perspective. The router should dispatch between these modes.

### SOTA Model > Infrastructure
Metaculus's own research confirms: "Using a SOTA model matters more than investing in bot infrastructure." The Q2 2025 winner used o3 with basic scaffolding. Fancy multi-agent debate didn't beat a good single model with good research.

### Extremization is the Highest-ROI Intervention
Logit-space extremization (alpha 1.3-1.5) is "the highest-ROI single intervention for LLM forecasting calibration" per the hugo0-bot description. Our benchmark confirmed: alpha=1.5 extremization turned a losing swarm (Brier 0.041) into a winner (0.017).

---

## Security & Operational Issues

### Exposed Tokens
API tokens were shared in conversation and could end up in git history. Mitigation:
- `.env` is in `.gitignore`
- Tokens should be rotated after any exposure
- In production: use environment variables, never hardcode

### Hardcoded Demo Key
`api.py` defaulted to `"demo-key-12345"` for API auth. In production, the server should refuse to start without an explicit `MINISIM_API_KEYS` environment variable.

### In-Memory Prediction Store
`_predictions` is a Python dict in `api.py`. Server restart = all data lost. Need Redis or PostgreSQL for persistence. The SQLite logging is best-effort (wrapped in try/except).

---

## Forward Testing vs Backtesting

### Backtesting Pitfalls
- **Data leakage**: The LLM may have seen the resolution in its training data. A question about "Will Apple release Vision Pro in 2024?" is in every LLM's training set. The model "predicts" 0.95 not because it's a good forecaster but because it already knows the answer.
- **Selection bias**: We curated 100 questions with known outcomes. This isn't representative of the questions MiniSim would face in production.
- **Price at prediction time**: We used the market price at settlement, not at prediction time. A market at 0.99 right before resolution is easy to "predict."

### Forward Testing (What We're Doing with Metaculus)
The Metaculus tournament is genuine forward testing:
- Questions are NEW — no one knows the answer yet
- Scoring is against other bots (peer score, not Brier)
- Results come in over weeks/months
- This is the only valid way to measure forecasting ability

### Pastcasting
An intermediate approach: forecast on past events that happened AFTER the model's knowledge cutoff. Metaculus warns about pitfalls — LLMs can sometimes infer outcomes from adjacent knowledge even post-cutoff.

---

## Circuit Breakers & Failure Modes

### LLM Failure
- If Ollama is down: falls back to offline engine (silent degradation)
- If a single agent call fails: that agent uses offline fallback, `from_llm=False`
- If ALL agent calls fail: entire prediction uses offline engine
- **Need**: A threshold — if >50% of agents fall back, flag the prediction as degraded

### API Rate Limiting
- Kalshi: 429 responses with 2s retry
- Metaculus: Bot token has rate limits (not documented)
- Polymarket: 429 with 2s retry
- **Need**: Per-platform rate limit tracking and backoff

### Arbitrage Execution Risk
Fee-aware arbitrage calculation is theoretical. Actual execution risks:
- Slippage: price moves between calculation and execution
- Partial fills: can't always get full position size at displayed price
- Settlement timing: platforms settle at different speeds
- Counterparty risk: platform could freeze withdrawals

### Database Failure
All database writes are wrapped in try/except with logging. If SQLite file is locked or corrupted, predictions still work — just aren't persisted. **Need**: Health check that verifies DB is writable.

---

## Race Conditions

### Concurrent Predictions in API
`_predictions` dict is not thread-safe. Two concurrent POST /v1/predict requests could:
- Overwrite each other's entries
- Read partial state during background task execution
- **Fix**: Use `threading.Lock` or move to Redis

### Scanner + API Conflict
Both the scanner and the API write to the same SQLite database. SQLite supports concurrent reads but only one writer at a time. Under load, one will get `sqlite3.OperationalError: database is locked`.
- **Fix**: PostgreSQL or WAL mode for SQLite

### Metaculus Bot Timing
Tournament questions are open for ~1.5 hours. If the bot takes 60 min for 20 questions on Qwen 14B, it might miss some questions. Need:
- Prioritize new questions (closest to closing first)
- Use faster model for tournament (8B instead of 14B)
- Or submit initial forecast quickly, then update with better one

### Background Task Completion
FastAPI `BackgroundTasks` runs after response. If the server crashes mid-prediction, the prediction is lost with status "processing" forever. Need:
- Persistent task queue (Redis + arq/rq)
- Timeout on stale predictions
- Client-side polling with max attempts

---

## Technical Debt Inventory

1. **Router thresholds unvalidated** — 0.05/0.10 set from 10-question benchmark
2. **Calibration not applied at inference** — CalibrationTransformer fitted but never called in production pipeline
3. **In-memory API state** — `_predictions` dict lost on restart
4. **SQLite for production** — needs PostgreSQL migration
5. **No API endpoint tests** — FastAPI endpoints untested
6. **Survey engine may be dead weight** — 657 lines across 2 files, unclear product fit vs Simile
7. **11 files import from offline_engine** — backward compat re-exports add complexity
8. **No structured logging** — using print() for progress, logger for errors, inconsistent
9. **Hardcoded Metaculus bot token** in `metaculus_bot.py` default env var
10. **No monitoring/alerting** — no Sentry, no uptime checks, no latency tracking
11. **Metaculus Bot Benchmarking access not granted** — need to email api-requests@metaculus.com to unlock resolution + community prediction data on 250+ questions

---

## Process Lessons

### Always Commit + Push + Update Docs
The most consistent feedback: every code change must be immediately followed by:
1. `git add` + `git commit` with descriptive message
2. `git push`
3. README update if features/metrics changed
4. ENGINEERING_NOTES update if technical lesson learned
5. Memory file update if new context for future sessions

Uncommitted work is invisible to reviewers. Outdated README is worse than no README.

### Background Process Management
Multiple issues with long-running background tasks:
- Python output buffering means no visible progress (fix: `python3 -u` for unbuffered)
- `| head -N` kills the process after N lines (don't use for long-running commands)
- Always log to a file: `> results/output.log 2>&1 &`
- Track PIDs for clean shutdown
- Don't confuse "process exited" with "task completed" — head/pipe can cause early exit

### Validate Claims with Data
Multiple findings were stated confidently but validated on only 10 questions:
- Alpha=1.5 extremization "53% better" — 10 questions
- Router thresholds 0.05/0.10 — 10 questions
- "Single LLM beats swarm" — 10 questions

10 questions is noise, not signal. The plan's hard gate (validate on 500+) exists for a reason. Don't proceed to fine-tuning until these are validated at scale.
