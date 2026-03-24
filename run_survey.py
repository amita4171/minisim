"""
MiniSim Survey Runner — Synthetic population survey simulation.

Replaces traditional $50K-200K, 4-8 week surveys with synthetic
respondents in minutes. The Aaru competitor feature.

Usage:
    python run_survey.py --demo              # Run demo survey
    python run_survey.py --file survey.json  # Run custom survey from file
    python run_survey.py --respondents 200   # Customize respondent count
"""
from __future__ import annotations

import argparse
import json
import os
import time

from src.survey_engine import SurveyEngine
from src.llm_engine import LLMEngine


# Demo surveys for different verticals
DEMO_SURVEYS = {
    "consumer_product": {
        "title": "AI Personal Assistant Product Research",
        "description": "Consumer attitudes toward a new AI-powered personal assistant device ($299)",
        "questions": [
            {
                "text": "How interested are you in purchasing an AI-powered personal assistant device for your home?",
                "type": "likert",
                "scale_min": 1,
                "scale_max": 5,
                "anchor_low": "Not at all interested",
                "anchor_high": "Extremely interested",
            },
            {
                "text": "Which feature matters most to you in an AI assistant?",
                "type": "multiple_choice",
                "options": [
                    "Voice control for smart home devices",
                    "Calendar and task management",
                    "Health and wellness tracking",
                    "Entertainment and music",
                    "Shopping and ordering",
                    "Privacy and data security",
                ],
            },
            {
                "text": "What is the maximum price you would pay for an AI personal assistant device?",
                "type": "willingness_to_pay",
                "product": "an AI-powered personal assistant device that manages your calendar, controls smart home devices, tracks health, and answers questions",
            },
            {
                "text": "How likely are you to recommend an AI personal assistant to a friend?",
                "type": "nps",
            },
            {
                "text": "Would you trust an AI assistant with your personal health data?",
                "type": "binary",
            },
            {
                "text": "What concerns do you have about AI assistants in the home?",
                "type": "open_ended",
            },
        ],
    },
    "brand_perception": {
        "title": "Electric Vehicle Brand Perception Study",
        "description": "Consumer perception of leading EV brands",
        "questions": [
            {
                "text": "How likely are you to consider purchasing an electric vehicle in the next 3 years?",
                "type": "likert",
                "scale_min": 1,
                "scale_max": 5,
                "anchor_low": "Very unlikely",
                "anchor_high": "Very likely",
            },
            {
                "text": "Which EV brand do you trust most?",
                "type": "multiple_choice",
                "options": ["Tesla", "Ford", "GM/Chevrolet", "Hyundai/Kia", "BMW", "Rivian", "Other/None"],
            },
            {
                "text": "What is the maximum price you would pay for an electric vehicle?",
                "type": "willingness_to_pay",
                "product": "a new electric vehicle with 300-mile range, fast charging, and full self-driving capability",
            },
            {
                "text": "What is your biggest concern about owning an electric vehicle?",
                "type": "multiple_choice",
                "options": [
                    "Charging infrastructure / range anxiety",
                    "Purchase price too high",
                    "Battery degradation over time",
                    "Limited model/style options",
                    "Environmental impact of battery production",
                    "No concerns — I'm ready to buy",
                ],
            },
        ],
    },
    "policy": {
        "title": "AI Regulation Public Opinion Survey",
        "description": "Public attitudes toward government regulation of AI",
        "questions": [
            {
                "text": "Do you believe the government should regulate artificial intelligence?",
                "type": "binary",
            },
            {
                "text": "How concerned are you about AI replacing human jobs?",
                "type": "likert",
                "scale_min": 1,
                "scale_max": 5,
                "anchor_low": "Not concerned at all",
                "anchor_high": "Extremely concerned",
            },
            {
                "text": "Which area of AI regulation is most important?",
                "type": "multiple_choice",
                "options": [
                    "Privacy and data protection",
                    "Job displacement and worker protection",
                    "AI safety and alignment",
                    "Bias and discrimination in AI systems",
                    "Military and weapons use of AI",
                    "AI-generated misinformation",
                ],
            },
            {
                "text": "What are your thoughts on AI's impact on society over the next 10 years?",
                "type": "open_ended",
            },
        ],
    },
}


def print_results(results: dict):
    """Pretty-print survey results."""
    print(f"\n{'=' * 70}")
    print(f"SURVEY RESULTS")
    print(f"Respondents: {results['n_respondents']} | Questions: {results['n_questions']} | Time: {results['total_time_s']}s")
    print(f"{'=' * 70}")

    # Demographics
    print(f"\n--- Respondent Demographics ---")
    for dim, counts in results.get("demographics", {}).items():
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"  {dim}: {dist}")

    # Per-question results
    for qi, q in enumerate(results.get("questions", [])):
        print(f"\n{'─' * 60}")
        print(f"  Q{qi+1}: {q['question'][:65]}")
        print(f"  Type: {q['type']} | Responses: {q['n_responses']}")

        if q["type"] == "likert":
            print(f"  Mean: {q.get('mean', 'N/A')} | Median: {q.get('median', 'N/A')} | StdDev: {q.get('stdev', 'N/A')}")
            if q.get("distribution"):
                for rating, count in sorted(q["distribution"].items()):
                    bar = "#" * count
                    print(f"    {rating}: {bar} ({count})")

        elif q["type"] == "multiple_choice":
            if q.get("distribution"):
                for option, data in q["distribution"].items():
                    pct = data["pct"] if isinstance(data, dict) else data
                    count = data["count"] if isinstance(data, dict) else data
                    print(f"    {pct:>5.1f}% | {option}")

        elif q["type"] == "nps":
            print(f"  NPS Score: {q.get('nps_score', 'N/A')}")
            print(f"  Promoters: {q.get('promoters_pct', 0)}% | Passives: {q.get('passives_pct', 0)}% | Detractors: {q.get('detractors_pct', 0)}%")
            print(f"  Mean: {q.get('mean', 'N/A')}")

        elif q["type"] == "willingness_to_pay":
            print(f"  Mean max price: ${q.get('mean_max_price', 'N/A')}")
            print(f"  Median: ${q.get('median_max_price', 'N/A')}")
            print(f"  25th percentile: ${q.get('p25', 'N/A')} | 75th: ${q.get('p75', 'N/A')}")
            if q.get("cross_tabs", {}).get("income"):
                print(f"  By income:")
                for inc, data in q["cross_tabs"]["income"].items():
                    print(f"    {inc}: ${data['mean']:.0f} (n={data['n']})")

        elif q["type"] == "binary":
            print(f"  Yes: {q.get('yes_pct', 0)}% | No: {q.get('no_pct', 0)}%")
            if q.get("cross_tabs", {}).get("age"):
                print(f"  By age:")
                for age, data in q["cross_tabs"]["age"].items():
                    dist = data.get("distribution", {})
                    yes = dist.get("yes", 0)
                    total = data.get("n", 1)
                    print(f"    {age}: {yes}/{total} ({yes/total*100:.0f}% yes)")

        elif q["type"] == "open_ended":
            sent = q.get("sentiment_distribution", {})
            print(f"  Sentiment: +{sent.get('positive', 0)} / ={sent.get('neutral', 0)} / -{sent.get('negative', 0)}")
            if q.get("sample_responses"):
                print(f"  Sample responses:")
                for resp in q["sample_responses"][:3]:
                    print(f"    \"{resp[:80]}\"")


def main():
    parser = argparse.ArgumentParser(description="MiniSim Survey Simulator")
    parser.add_argument("--demo", type=str, choices=list(DEMO_SURVEYS.keys()),
                       default="consumer_product", help="Run a demo survey")
    parser.add_argument("--file", type=str, default=None, help="Run survey from JSON file")
    parser.add_argument("--respondents", type=int, default=30, help="Number of respondents")
    parser.add_argument("--model", type=str, default=None, help="Ollama model")
    args = parser.parse_args()

    engine = LLMEngine(model=args.model)

    if args.file:
        with open(args.file) as f:
            survey = json.load(f)
        questions = survey.get("questions", [])
        title = survey.get("title", "Custom Survey")
    else:
        survey = DEMO_SURVEYS[args.demo]
        questions = survey["questions"]
        title = survey["title"]

    print(f"\n{'#' * 70}")
    print(f"  {title}")
    print(f"  {survey.get('description', '')}")
    print(f"  Engine: {engine.backend}/{engine.model}")
    print(f"{'#' * 70}")

    se = SurveyEngine(n_respondents=args.respondents, engine=engine)
    results = se.run_survey(questions)
    results["title"] = title

    print_results(results)

    # Save
    os.makedirs("results", exist_ok=True)
    safe_name = title.lower().replace(" ", "_")[:40]
    path = f"results/survey_{safe_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
