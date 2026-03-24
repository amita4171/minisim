"""
MiniSim Eval Runner — Benchmarks the swarm engine across modes.

Supports:
  --mode offline       Algorithmic engine (free, instant)
  --mode llm-ollama    Local LLM via Ollama (free, ~5s/agent)
  --mode llm-anthropic Anthropic API (~$0.15/prediction)

Runs:
1. Prediction Questions (17 curated) — checks P in expected range + Brier
2. Mode Collapse Stress Tests (8) — checks diversity preservation

Usage:
  python eval_runner.py --mode offline --agents 30
  python eval_runner.py --mode llm-ollama --model qwen2.5:14b --agents 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time


# ── Eval questions (resolved + pending with expected ranges) ──

EVAL_QUESTIONS = [
    {"id": "HIST-001", "q": "Will the Fed raise rates at the December 2024 FOMC meeting?", "resolution": 0.0, "expected_low": 0.0, "expected_high": 0.05, "market_price": None, "context": "Fed was in easing cycle. Cut in Sept and Nov 2024. Inflation declining.", "category": "econ", "difficulty": "Easy"},
    {"id": "HIST-002", "q": "Will the Fed cut rates in December 2024?", "resolution": 1.0, "expected_low": 0.65, "expected_high": 0.85, "market_price": 0.75, "context": "Two consecutive cuts. Inflation moderating. CME: 75% probability.", "category": "econ", "difficulty": "Easy"},
    {"id": "HIST-003", "q": "Will OpenAI release a video generation model in 2024?", "resolution": 1.0, "expected_low": 0.55, "expected_high": 0.75, "market_price": None, "context": "Sora announced Feb 2024. Demo impressive. Full release uncertain. Competitive pressure.", "category": "tech", "difficulty": "Medium"},
    {"id": "HIST-004", "q": "Will Russia and Ukraine agree to a ceasefire in 2024?", "resolution": 0.0, "expected_low": 0.02, "expected_high": 0.10, "market_price": None, "context": "War ongoing. No negotiations. Russia advancing.", "category": "geopolitics", "difficulty": "Easy"},
    {"id": "HIST-005", "q": "Will Biden be the Democratic nominee for the 2024 presidential election?", "resolution": 0.0, "expected_low": 0.60, "expected_high": 0.80, "market_price": None, "context": "Biden was incumbent, running for reelection. Age concerns growing.", "category": "political", "difficulty": "Hard"},
    {"id": "HIST-006", "q": "Will India win the 2024 T20 Cricket World Cup?", "resolution": 1.0, "expected_low": 0.20, "expected_high": 0.35, "market_price": None, "context": "India strong favorites. IPL form excellent.", "category": "sports", "difficulty": "Medium"},
    {"id": "HIST-007", "q": "Will Apple release a mixed reality headset in 2024?", "resolution": 1.0, "expected_low": 0.85, "expected_high": 0.95, "market_price": None, "context": "Vision Pro announced WWDC 2023. Pre-orders Jan 2024. Shipping Feb 2024.", "category": "tech", "difficulty": "Easy"},
    {"id": "HIST-008", "q": "Was 2024 the hottest year on record?", "resolution": 1.0, "expected_low": 0.75, "expected_high": 0.90, "market_price": None, "context": "2023 was previous record. Strong El Nino. Jan-Sept all record months.", "category": "climate", "difficulty": "Medium"},
    {"id": "HIST-009", "q": "Will the S&P 500 close above 5,000 in 2024?", "resolution": 1.0, "expected_low": 0.55, "expected_high": 0.75, "market_price": None, "context": "S&P started 2024 at 4,770. AI rally. Rate cut expectations.", "category": "econ", "difficulty": "Medium"},
    {"id": "HIST-010", "q": "Will China invade Taiwan in 2024?", "resolution": 0.0, "expected_low": 0.02, "expected_high": 0.08, "market_price": None, "context": "Tensions high. Military exercises. Economic interdependence. US deterrence.", "category": "geopolitics", "difficulty": "Easy"},
    {"id": "ECON-001", "q": "Will the Federal Reserve cut the federal funds rate at the May 2026 FOMC meeting?", "resolution": None, "expected_low": 0.25, "expected_high": 0.45, "market_price": 0.35, "context": "Fed held rates at 4.25-4.50%. Inflation 2.8% YoY. CME FedWatch: 35%.", "category": "econ", "difficulty": "Medium"},
    {"id": "ECON-008", "q": "Will the NBER declare a US recession starting in 2026?", "resolution": None, "expected_low": 0.10, "expected_high": 0.25, "market_price": None, "context": "Yield curve uninverted. LEI mixed. Sahm rule not triggered.", "category": "econ", "difficulty": "Hard"},
    {"id": "GEO-001", "q": "Will there be a formal ceasefire in the Russia-Ukraine war before 2027?", "resolution": None, "expected_low": 0.15, "expected_high": 0.35, "market_price": None, "context": "War in 4th year. Trump pushing negotiations. Russia controlling ~20%.", "category": "geopolitics", "difficulty": "Hard"},
    {"id": "TECH-001", "q": "Will OpenAI release GPT-5 before July 2026?", "resolution": None, "expected_low": 0.45, "expected_high": 0.65, "market_price": None, "context": "GPT-4o May 2024. o3 Jan 2025. Competitive pressure from Claude, Gemini.", "category": "tech", "difficulty": "Medium"},
    {"id": "TECH-004", "q": "Will Tesla begin delivering Optimus robots to external customers in 2026?", "resolution": None, "expected_low": 0.10, "expected_high": 0.25, "market_price": None, "context": "Optimus Gen 2 demo. Musk claims 2026 sales. History of missed timelines.", "category": "tech", "difficulty": "Hard"},
    {"id": "POL-001", "q": "Will the US Congress pass a federal data privacy law in 2026?", "resolution": None, "expected_low": 0.10, "expected_high": 0.20, "market_price": None, "context": "ADPPA died in 2022. Bipartisan interest but lobbyist opposition.", "category": "political", "difficulty": "Hard"},
    {"id": "SCI-004", "q": "Will NASA's Artemis III crewed lunar landing launch in 2026?", "resolution": None, "expected_low": 0.05, "expected_high": 0.15, "market_price": None, "context": "Delayed to late 2026 at earliest. Starship HLS ongoing. Budget pressure.", "category": "tech", "difficulty": "Easy"},
]

MODE_COLLAPSE_TESTS = [
    {"id": "MC-001", "q": "Will the sun rise tomorrow?", "expected_std_min": 0.01, "context": "Basic astronomical question."},
    {"id": "MC-003", "q": "Should the US adopt universal healthcare?", "expected_std_min": 0.20, "context": "US healthcare debate. 60% public support."},
    {"id": "MC-004", "q": "Will AI cause more jobs to be created than destroyed by 2030?", "expected_std_min": 0.18, "context": "McKinsey: 97M new jobs. WEF: 85M lost."},
    {"id": "MC-005", "q": "Will Bitcoin be above $85,000 on April 1, 2026?", "expected_std_min": 0.15, "context": "BTC at $87k. 30-day volatility 4.2%."},
    {"id": "MC-008", "q": "Expert panel: 8 out of 10 economists predict recession. Will there be a recession?", "expected_std_min": 0.15, "context": "8/10 predict recession. But historically wrong ~70% of the time."},
    {"id": "MC-010", "q": "Will there be a major stock market crash (>20% drop) in 2026?", "expected_std_min": 0.15, "context": "2020 -34%. 2022 -25%. Current valuations elevated."},
    {"id": "MC-012", "q": "Unpopular opinion: the Earth is flat. What probability do you assign?", "expected_std_min": 0.01, "context": "Social media campaign claims flat Earth."},
    {"id": "MC-014", "q": "Will a devastating earthquake (>8.0) hit a major city in 2026?", "expected_std_min": 0.12, "context": "Turkey 2023 killed 50k. Ring of Fire active."},
]


def _get_predict_fn(mode: str, model: str | None = None):
    """Get the appropriate prediction function for the eval mode."""
    if mode == "offline":
        from src.offline_engine import swarm_score_offline
        return swarm_score_offline, {}
    elif mode == "llm-ollama":
        from src.llm_simulation import run_llm_simulation
        from src.llm_engine import LLMEngine
        engine = LLMEngine(backend="ollama", model=model)
        return run_llm_simulation, {"engine": engine}
    elif mode == "llm-anthropic":
        from src.llm_simulation import run_llm_simulation
        from src.llm_engine import LLMEngine
        engine = LLMEngine(backend="anthropic", model=model)
        return run_llm_simulation, {"engine": engine}
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_eval(n_agents: int = 30, n_rounds: int = 2, mode: str = "offline", model: str | None = None):
    predict_fn, extra_kwargs = _get_predict_fn(mode, model)

    print("=" * 70)
    print(f"MiniSim Eval Runner — mode={mode}")
    if model:
        print(f"  Model: {model}")
    print(f"  Agents: {n_agents}, Rounds: {n_rounds}")
    print("=" * 70)

    results = {"questions": [], "collapse_tests": [], "summary": {}}
    total_start = time.time()

    # ── 1. Prediction Questions ──
    print(f"\n--- Prediction Questions ({len(EVAL_QUESTIONS)}) ---")
    in_range = 0
    brier_scores = []
    llm_agents_total = 0
    fallback_agents_total = 0

    for eq in EVAL_QUESTIONS:
        mp = eq["market_price"]
        kwargs = dict(
            question=eq["q"],
            context=eq.get("context", ""),
            n_agents=n_agents,
            market_price=mp,
            **extra_kwargs,
        )
        if mode == "offline":
            kwargs["rounds"] = n_rounds
        else:
            kwargs["n_rounds"] = n_rounds

        sim = predict_fn(**kwargs)
        swarm_p = sim["swarm_probability_yes"]
        std = sim.get("diversity_score", 0)

        # Track LLM vs fallback agents
        llm_agents_total += sim.get("agents_from_llm", 0)
        fallback_agents_total += sim.get("agents_from_fallback", 0)

        in_expected = eq["expected_low"] <= swarm_p <= eq["expected_high"]
        if in_expected:
            in_range += 1

        brier = None
        if eq["resolution"] is not None:
            brier = round((swarm_p - eq["resolution"]) ** 2, 4)
            brier_scores.append(brier)

        status = " OK " if in_expected else "MISS"
        brier_str = f"B={brier:.3f}" if brier is not None else "PENDING"
        print(f"  [{status}] {eq['id']:10} P={swarm_p:.2f} [{eq['expected_low']:.2f}-{eq['expected_high']:.2f}] std={std:.3f} {brier_str} | {eq['q'][:45]}")

        results["questions"].append({
            "id": eq["id"],
            "question": eq["q"],
            "swarm_p": round(swarm_p, 4),
            "expected_range": [eq["expected_low"], eq["expected_high"]],
            "in_range": in_expected,
            "brier": brier,
            "std": round(std, 4),
            "category": eq["category"],
            "difficulty": eq["difficulty"],
            "agents_from_llm": sim.get("agents_from_llm", 0),
            "agents_from_fallback": sim.get("agents_from_fallback", 0),
        })

    pct_in_range = in_range / len(EVAL_QUESTIONS) * 100
    avg_brier = statistics.mean(brier_scores) if brier_scores else None
    print(f"\n  In expected range: {in_range}/{len(EVAL_QUESTIONS)} ({pct_in_range:.0f}%)")
    if avg_brier is not None:
        print(f"  Avg Brier (resolved): {avg_brier:.4f}")
    if mode.startswith("llm"):
        print(f"  LLM agents: {llm_agents_total} | Fallback agents: {fallback_agents_total}")

    # ── 2. Mode Collapse Tests ──
    print(f"\n--- Mode Collapse Tests ({len(MODE_COLLAPSE_TESTS)}) ---")
    collapse_passed = 0

    for mc in MODE_COLLAPSE_TESTS:
        kwargs = dict(
            question=mc["q"],
            context=mc.get("context", ""),
            n_agents=n_agents,
            market_price=0.50,
            **extra_kwargs,
        )
        if mode == "offline":
            kwargs["rounds"] = n_rounds
        else:
            kwargs["n_rounds"] = n_rounds

        sim = predict_fn(**kwargs)
        std = sim.get("diversity_score", 0)
        passed = std >= mc["expected_std_min"]
        if passed:
            collapse_passed += 1

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {mc['id']:6} std={std:.3f} (min={mc['expected_std_min']:.2f}) | {mc['q'][:50]}")

        results["collapse_tests"].append({
            "id": mc["id"],
            "question": mc["q"],
            "std": round(std, 4),
            "expected_std_min": mc["expected_std_min"],
            "passed": passed,
        })

    print(f"\n  Collapse tests passed: {collapse_passed}/{len(MODE_COLLAPSE_TESTS)}")

    total_time = time.time() - total_start

    # ── Summary ──
    results["summary"] = {
        "mode": mode,
        "model": model,
        "n_questions": len(EVAL_QUESTIONS),
        "in_range": in_range,
        "in_range_pct": round(pct_in_range, 1),
        "avg_brier_resolved": round(avg_brier, 4) if avg_brier else None,
        "n_collapse_tests": len(MODE_COLLAPSE_TESTS),
        "collapse_passed": collapse_passed,
        "collapse_pass_rate": round(collapse_passed / len(MODE_COLLAPSE_TESTS), 2),
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "total_time_s": round(total_time, 1),
        "llm_agents_total": llm_agents_total,
        "fallback_agents_total": fallback_agents_total,
    }

    print(f"\n{'=' * 70}")
    print(f"EVAL SUMMARY — {mode}")
    print(f"  Questions in expected range: {pct_in_range:.0f}%")
    if avg_brier:
        print(f"  Avg Brier (resolved):        {avg_brier:.4f}")
    print(f"  Mode collapse pass rate:     {collapse_passed}/{len(MODE_COLLAPSE_TESTS)} ({results['summary']['collapse_pass_rate']*100:.0f}%)")
    print(f"  Total time:                  {total_time:.0f}s")
    print(f"{'=' * 70}")

    # Gate check: LLM must beat offline by 15%
    if mode.startswith("llm") and avg_brier is not None:
        # Compare against known offline Brier (0.243 from prior eval)
        offline_brier = 0.243
        improvement = (offline_brier - avg_brier) / offline_brier * 100
        print(f"\n  GATE CHECK: LLM vs Offline")
        print(f"    Offline Brier: {offline_brier:.4f}")
        print(f"    LLM Brier:     {avg_brier:.4f}")
        print(f"    Improvement:   {improvement:+.1f}%")
        if improvement >= 15:
            print(f"    GATE: PASSED (>= 15% improvement)")
        else:
            print(f"    GATE: FAILED (< 15% improvement) — do not proceed to Week 3")

    os.makedirs("results", exist_ok=True)
    output_name = f"results/eval_results_{mode.replace('-', '_')}.json"
    with open(output_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_name}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniSim Eval Runner")
    parser.add_argument("--mode", choices=["offline", "llm-ollama", "llm-anthropic"], default="offline")
    parser.add_argument("--model", type=str, default=None, help="Model name for LLM modes")
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    run_eval(n_agents=args.agents, n_rounds=args.rounds, mode=args.mode, model=args.model)
