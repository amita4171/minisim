---
name: MiniSim Tournament Bot
description: How to manage the Metaculus tournament bot — submit, watch, resolve
---

# Metaculus Tournament Bot

## Submit Forecasts
```bash
# Dry run first
METACULUS_BOT_TOKEN=<token> python3 -u metaculus_bot.py --dry-run --tournament spring-aib-2026 --model qwen2.5:14b

# Real submission (reuse dry run predictions)
# Extract predictions from dry run log and submit directly via API

# Or run live (computes + submits)
METACULUS_BOT_TOKEN=<token> python3 -u metaculus_bot.py --tournament spring-aib-2026 --model qwen2.5:14b
```

## Watch Mode (continuous)
```bash
METACULUS_BOT_TOKEN=<token> nohup python3 -u metaculus_bot.py \
  --watch --interval 1800 \
  --tournament spring-aib-2026 \
  --model qwen2.5:14b \
  >> results/bot_watch.log 2>&1 &
```
- Runs every 30 minutes
- Dies if Mac sleeps — restart after wake
- Persistent cache: results/forecasted_questions.json

## Check Resolution Status
```bash
METACULUS_BOT_TOKEN=<token> python3 resolve_metaculus.py
```
Note: Resolution values gated behind Bot Benchmarking tier.

## Monitor
```bash
tail -f results/bot_watch.log                    # live output
cat results/forecasted_questions.json | python3 -m json.tool  # submitted questions
grep "Forecast submitted" results/bot_watch.log  # count submissions
```

## Tournament Details
- **Spring 2026 AIB** — project slug: spring-aib-2026 (ID: 32916)
- **MiniBench** — project slug: minibench (bi-weekly, $1K)
- $50K prize pool, 300-500 questions, scored by spot peer score
- Questions open for ~1.5 hours only
- ONE forecast per question (no updates in bot-only tournaments)
- New questions drop randomly throughout the week
- Most questions resolve at end of season (Jan)

## Bot Behavior
- Skips questions with "Moved to" or "RESTATED" in title
- Uses variance-based router (single_llm / light_deliberation / full_swarm)
- Submits private comment with reasoning
- Logs to SQLite database with diversity_score + confidence_interval

## API Endpoints
- Submit: POST /api/questions/forecast/ with [{"question": <id>, "probability_yes": <float>}]
- Comment: POST /api/comments/create/ with {"on_post": <id>, "text": "...", "is_private": true}
- Auth: Authorization: Token <bot_token>

## Free Credits
- Applied via Google Form: https://forms.gle/aQdYMq9Pisrf1v7d8
- Provides OpenRouter key with Claude Sonnet + o3 credits
- 0-5 business days response
