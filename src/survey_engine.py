"""
Survey Simulation Engine — Synthetic population survey research.

Simulates N respondents with demographic profiles answering survey questions.
This is the Aaru competitor: replace $50K-200K, 4-8 week surveys with
synthetic responses in minutes.

Supports:
- Likert scale (1-5, 1-7, 1-10)
- Multiple choice (select one or many)
- Open-ended (free text)
- Willingness to pay (price sensitivity)
- Net Promoter Score (0-10)
- Binary (yes/no)

Each respondent has a demographic profile that influences their responses:
age, gender, income, education, geography, ethnicity.

Usage:
    from src.survey_engine import SurveyEngine
    engine = SurveyEngine(n_respondents=100)
    results = engine.run_survey(questions)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.llm_engine import LLMEngine


# ── Demographic distributions (US Census-approximate) ──

AGE_GROUPS = [
    {"label": "18-24", "weight": 0.12, "traits": {"tech_savvy": 0.9, "price_sensitive": 0.8, "brand_loyal": 0.3}},
    {"label": "25-34", "weight": 0.18, "traits": {"tech_savvy": 0.85, "price_sensitive": 0.6, "brand_loyal": 0.4}},
    {"label": "35-44", "weight": 0.16, "traits": {"tech_savvy": 0.7, "price_sensitive": 0.5, "brand_loyal": 0.5}},
    {"label": "45-54", "weight": 0.15, "traits": {"tech_savvy": 0.5, "price_sensitive": 0.4, "brand_loyal": 0.6}},
    {"label": "55-64", "weight": 0.17, "traits": {"tech_savvy": 0.35, "price_sensitive": 0.4, "brand_loyal": 0.7}},
    {"label": "65+", "weight": 0.22, "traits": {"tech_savvy": 0.2, "price_sensitive": 0.5, "brand_loyal": 0.8}},
]

GENDERS = [
    {"label": "Male", "weight": 0.49},
    {"label": "Female", "weight": 0.49},
    {"label": "Non-binary", "weight": 0.02},
]

INCOME_LEVELS = [
    {"label": "Under $30K", "weight": 0.20, "spending_power": 0.2},
    {"label": "$30K-$60K", "weight": 0.25, "spending_power": 0.4},
    {"label": "$60K-$100K", "weight": 0.25, "spending_power": 0.6},
    {"label": "$100K-$150K", "weight": 0.15, "spending_power": 0.8},
    {"label": "Over $150K", "weight": 0.15, "spending_power": 1.0},
]

EDUCATION = [
    {"label": "High School", "weight": 0.27},
    {"label": "Some College", "weight": 0.20},
    {"label": "Bachelor's", "weight": 0.33},
    {"label": "Master's+", "weight": 0.20},
]

REGIONS = [
    {"label": "Northeast", "weight": 0.17, "urban": 0.85},
    {"label": "Midwest", "weight": 0.21, "urban": 0.65},
    {"label": "South", "weight": 0.38, "urban": 0.70},
    {"label": "West", "weight": 0.24, "urban": 0.80},
]

# LLM prompt for survey response generation
SURVEY_SYSTEM_PROMPT = """You are simulating a survey respondent with this demographic profile:
- Age: {age}
- Gender: {gender}
- Income: {income}
- Education: {education}
- Region: {region}

Answer the survey question authentically from this person's perspective.
Consider how their demographics would influence their views, preferences, and experiences.
Respond with ONLY a valid JSON object."""

SURVEY_QUESTION_PROMPTS = {
    "likert": """Survey question: "{question}"

Rate on a scale of {scale_min} to {scale_max}, where {scale_min} = "{anchor_low}" and {scale_max} = "{anchor_high}".

Return JSON: {{"rating": <integer {scale_min}-{scale_max}>, "reasoning": "<one sentence>"}}""",

    "multiple_choice": """Survey question: "{question}"

Options:
{options_text}

Select the option that best matches your perspective.

Return JSON: {{"choice": "<exact text of chosen option>", "reasoning": "<one sentence>"}}""",

    "open_ended": """Survey question: "{question}"

Provide a brief, authentic response (2-3 sentences) reflecting your demographic perspective.

Return JSON: {{"response": "<your answer>", "sentiment": "<positive/neutral/negative>"}}""",

    "willingness_to_pay": """Survey question: "What is the maximum price you would pay for: {product}?"

Consider your income level ({income}) and how much you value this product.

Return JSON: {{"max_price": <number in dollars>, "would_buy_at": <price you'd comfortably pay>, "reasoning": "<one sentence>"}}""",

    "nps": """Survey question: "{question}"

On a scale of 0-10, how likely are you to recommend this to a friend or colleague?
0 = Not at all likely, 10 = Extremely likely

Return JSON: {{"score": <integer 0-10>, "reasoning": "<one sentence>"}}""",

    "binary": """Survey question: "{question}"

Answer yes or no from your demographic perspective.

Return JSON: {{"answer": "<yes or no>", "confidence": <float 0-1>, "reasoning": "<one sentence>"}}""",
}


def _weighted_choice(options: list[dict], rng: random.Random) -> dict:
    """Pick from weighted options."""
    weights = [o["weight"] for o in options]
    return rng.choices(options, weights=weights, k=1)[0]


class SurveyEngine:
    """Runs synthetic surveys with demographically diverse respondents."""

    def __init__(
        self,
        n_respondents: int = 100,
        engine: LLMEngine | None = None,
        concurrency: int = 2,
        seed: int | None = None,
    ):
        self.n_respondents = n_respondents
        self.engine = engine or LLMEngine()
        self.concurrency = concurrency
        self.rng = random.Random(seed or 42)
        self.respondents = self._generate_respondents()

    def _generate_respondents(self) -> list[dict]:
        """Generate demographically diverse respondent profiles."""
        respondents = []
        for i in range(self.n_respondents):
            age = _weighted_choice(AGE_GROUPS, self.rng)
            gender = _weighted_choice(GENDERS, self.rng)
            income = _weighted_choice(INCOME_LEVELS, self.rng)
            education = _weighted_choice(EDUCATION, self.rng)
            region = _weighted_choice(REGIONS, self.rng)

            respondents.append({
                "id": i,
                "age": age["label"],
                "age_traits": age["traits"],
                "gender": gender["label"],
                "income": income["label"],
                "spending_power": income["spending_power"],
                "education": education["label"],
                "region": region["label"],
                "urban": region["urban"],
            })
        return respondents

    def run_survey(self, questions: list[dict]) -> dict:
        """Run a complete survey across all respondents.

        Args:
            questions: List of question dicts, each with:
                - text: str (the question)
                - type: str (likert, multiple_choice, open_ended, willingness_to_pay, nps, binary)
                - options: list[str] (for multiple_choice)
                - scale_min/scale_max: int (for likert)
                - anchor_low/anchor_high: str (for likert)
                - product: str (for willingness_to_pay)

        Returns:
            Dict with per-question results including distributions and cross-tabs
        """
        print(f"Running survey: {len(questions)} questions x {self.n_respondents} respondents")
        start = time.time()

        all_results = []
        for qi, question in enumerate(questions):
            print(f"  Q{qi+1}/{len(questions)}: {question['text'][:50]}...")
            q_results = self._run_question(question)
            all_results.append(q_results)

        elapsed = time.time() - start
        print(f"  Survey complete: {elapsed:.0f}s")

        return {
            "n_respondents": self.n_respondents,
            "n_questions": len(questions),
            "total_time_s": round(elapsed, 1),
            "questions": all_results,
            "demographics": self._demographic_summary(),
        }

    def _run_question(self, question: dict) -> dict:
        """Run a single question across all respondents."""
        q_type = question.get("type", "likert")
        responses = []

        def _ask_respondent(respondent):
            system = SURVEY_SYSTEM_PROMPT.format(
                age=respondent["age"],
                gender=respondent["gender"],
                income=respondent["income"],
                education=respondent["education"],
                region=respondent["region"],
            )

            if q_type == "likert":
                prompt = SURVEY_QUESTION_PROMPTS["likert"].format(
                    question=question["text"],
                    scale_min=question.get("scale_min", 1),
                    scale_max=question.get("scale_max", 5),
                    anchor_low=question.get("anchor_low", "Strongly Disagree"),
                    anchor_high=question.get("anchor_high", "Strongly Agree"),
                )
            elif q_type == "multiple_choice":
                options = question.get("options", [])
                options_text = "\n".join(f"  - {opt}" for opt in options)
                prompt = SURVEY_QUESTION_PROMPTS["multiple_choice"].format(
                    question=question["text"], options_text=options_text,
                )
            elif q_type == "willingness_to_pay":
                prompt = SURVEY_QUESTION_PROMPTS["willingness_to_pay"].format(
                    product=question.get("product", question["text"]),
                    income=respondent["income"],
                )
            elif q_type == "nps":
                prompt = SURVEY_QUESTION_PROMPTS["nps"].format(question=question["text"])
            elif q_type == "binary":
                prompt = SURVEY_QUESTION_PROMPTS["binary"].format(question=question["text"])
            else:  # open_ended
                prompt = SURVEY_QUESTION_PROMPTS["open_ended"].format(question=question["text"])

            parsed = self.engine.generate_json(prompt, system=system, temperature=0.7, max_tokens=200)
            return {"respondent": respondent, "response": parsed}

        # Run concurrently
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(_ask_respondent, r): r for r in self.respondents}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                if result["response"]:
                    responses.append(result)
                done += 1
                if done % 20 == 0 or done == self.n_respondents:
                    print(f"    [{done}/{self.n_respondents}]")

        return self._analyze_responses(question, responses)

    def _analyze_responses(self, question: dict, responses: list[dict]) -> dict:
        """Analyze responses and compute statistics + cross-tabs."""
        q_type = question.get("type", "likert")
        analysis = {
            "question": question["text"],
            "type": q_type,
            "n_responses": len(responses),
        }

        if q_type == "likert":
            ratings = [r["response"].get("rating") for r in responses if r["response"] and r["response"].get("rating")]
            ratings = [int(r) for r in ratings if r is not None]
            if ratings:
                analysis["mean"] = round(statistics.mean(ratings), 2)
                analysis["median"] = statistics.median(ratings)
                analysis["stdev"] = round(statistics.stdev(ratings), 2) if len(ratings) > 1 else 0
                analysis["distribution"] = {str(i): ratings.count(i) for i in range(question.get("scale_min", 1), question.get("scale_max", 5) + 1)}
                analysis["cross_tabs"] = self._cross_tabs(responses, lambda r: r["response"].get("rating"))

        elif q_type == "multiple_choice":
            choices = [r["response"].get("choice", "") for r in responses if r["response"]]
            choice_counts = {}
            for c in choices:
                # Fuzzy match to options
                best = c
                for opt in question.get("options", []):
                    if opt.lower() in c.lower() or c.lower() in opt.lower():
                        best = opt
                        break
                choice_counts[best] = choice_counts.get(best, 0) + 1
            total = sum(choice_counts.values())
            analysis["distribution"] = {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in sorted(choice_counts.items(), key=lambda x: -x[1])}

        elif q_type == "nps":
            scores = [r["response"].get("score") for r in responses if r["response"] and r["response"].get("score") is not None]
            scores = [int(s) for s in scores]
            if scores:
                promoters = sum(1 for s in scores if s >= 9)
                detractors = sum(1 for s in scores if s <= 6)
                n = len(scores)
                nps = round((promoters - detractors) / n * 100, 1) if n > 0 else 0
                analysis["nps_score"] = nps
                analysis["mean"] = round(statistics.mean(scores), 2)
                analysis["promoters_pct"] = round(promoters / n * 100, 1)
                analysis["passives_pct"] = round((n - promoters - detractors) / n * 100, 1)
                analysis["detractors_pct"] = round(detractors / n * 100, 1)
                analysis["distribution"] = {str(i): scores.count(i) for i in range(11)}

        elif q_type == "willingness_to_pay":
            prices = [r["response"].get("max_price") for r in responses if r["response"] and r["response"].get("max_price")]
            prices = [float(p) for p in prices if p is not None and float(p) > 0]
            if prices:
                analysis["mean_max_price"] = round(statistics.mean(prices), 2)
                analysis["median_max_price"] = round(statistics.median(prices), 2)
                analysis["p25"] = round(sorted(prices)[len(prices) // 4], 2)
                analysis["p75"] = round(sorted(prices)[3 * len(prices) // 4], 2)
                analysis["cross_tabs"] = self._cross_tabs_numeric(responses, lambda r: r["response"].get("max_price"))

        elif q_type == "binary":
            answers = [r["response"].get("answer", "").lower() for r in responses if r["response"]]
            yes = sum(1 for a in answers if a.startswith("y"))
            no = sum(1 for a in answers if a.startswith("n"))
            total = yes + no
            analysis["yes_pct"] = round(yes / total * 100, 1) if total > 0 else 0
            analysis["no_pct"] = round(no / total * 100, 1) if total > 0 else 0
            analysis["cross_tabs"] = self._cross_tabs(responses, lambda r: "yes" if r["response"].get("answer", "").lower().startswith("y") else "no")

        elif q_type == "open_ended":
            texts = [r["response"].get("response", "") for r in responses if r["response"]]
            sentiments = [r["response"].get("sentiment", "neutral") for r in responses if r["response"]]
            analysis["sample_responses"] = texts[:10]
            analysis["sentiment_distribution"] = {
                "positive": sum(1 for s in sentiments if s == "positive"),
                "neutral": sum(1 for s in sentiments if s == "neutral"),
                "negative": sum(1 for s in sentiments if s == "negative"),
            }

        return analysis

    def _cross_tabs(self, responses: list[dict], value_fn) -> dict:
        """Compute cross-tabs by key demographics."""
        tabs = {}
        for dim_name, dim_key in [("age", "age"), ("gender", "gender"), ("income", "income"), ("region", "region")]:
            groups = {}
            for r in responses:
                val = value_fn(r)
                if val is None:
                    continue
                group = r["respondent"][dim_key]
                if group not in groups:
                    groups[group] = []
                groups[group].append(val)

            tabs[dim_name] = {}
            for group, values in sorted(groups.items()):
                if isinstance(values[0], (int, float)):
                    tabs[dim_name][group] = {"mean": round(statistics.mean(values), 2), "n": len(values)}
                else:
                    from collections import Counter
                    counts = Counter(values)
                    tabs[dim_name][group] = {"distribution": dict(counts), "n": len(values)}

        return tabs

    def _cross_tabs_numeric(self, responses: list[dict], value_fn) -> dict:
        """Cross-tabs for numeric values."""
        tabs = {}
        for dim_name, dim_key in [("income", "income"), ("age", "age"), ("region", "region")]:
            groups = {}
            for r in responses:
                val = value_fn(r)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
                group = r["respondent"][dim_key]
                if group not in groups:
                    groups[group] = []
                groups[group].append(val)

            tabs[dim_name] = {}
            for group, values in sorted(groups.items()):
                if values:
                    tabs[dim_name][group] = {"mean": round(statistics.mean(values), 2), "n": len(values)}

        return tabs

    def _demographic_summary(self) -> dict:
        """Summarize the respondent pool demographics."""
        from collections import Counter
        return {
            "age": dict(Counter(r["age"] for r in self.respondents)),
            "gender": dict(Counter(r["gender"] for r in self.respondents)),
            "income": dict(Counter(r["income"] for r in self.respondents)),
            "education": dict(Counter(r["education"] for r in self.respondents)),
            "region": dict(Counter(r["region"] for r in self.respondents)),
        }
