"""
LLM Engine — Unified interface for local (Ollama) and API-based models.

Supports:
- Ollama (local, free): Llama 3 8B, Mistral 7B, Qwen 2.5, Gemma 2
- Anthropic API (paid): Claude Sonnet
- Fallback to offline engine if no LLM available

Usage:
    from src.llm_engine import LLMEngine
    engine = LLMEngine()  # auto-detects Ollama or falls back
    response = engine.generate("Your prompt", json_mode=True)
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LLMEngine:
    """Unified LLM interface with auto-detection and fallback."""

    def __init__(
        self,
        backend: str = "auto",
        model: str | None = None,
        ollama_url: str = "http://localhost:11434",
        max_retries: int = 3,
    ):
        """
        Args:
            backend: "ollama", "anthropic", or "auto" (tries ollama first)
            model: Model name. If None, auto-selects best available.
            ollama_url: Ollama server URL
            max_retries: Retry count for transient failures
        """
        self.ollama_url = ollama_url
        self.backend = backend
        self.model = model
        self.max_retries = max_retries
        self.stats = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0,
                      "total_ms": 0, "retries": 0}

        if backend == "auto":
            self.backend = self._detect_backend()

        if self.model is None:
            self.model = self._select_model()

        # Bug fix #1: Initialize Anthropic client once (connection pooling, thread-safe)
        self._anthropic_client = None
        if self.backend == "anthropic":
            try:
                from anthropic import Anthropic
                self._anthropic_client = Anthropic()
            except Exception as e:
                logger.debug(f"Anthropic client initialization failed: {e}")

    def _detect_backend(self) -> str:
        """Auto-detect available LLM backend."""
        # Try Ollama first (free, local)
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                if models:
                    return "ollama"
                # Ollama running but no models — still use it, will pull
                return "ollama"
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.debug(f"Ollama not available: {e}")

        # Try Anthropic API
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key and len(api_key) > 10:
            return "anthropic"

        return "offline"

    def _select_model(self) -> str:
        """Select the best available model for the backend."""
        if self.backend == "ollama":
            # Check what's installed
            try:
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                models = [m["name"] for m in resp.json().get("models", [])]
            except Exception:
                models = []

            # Preference order
            preferred = [
                "llama3.1:8b", "llama3:8b", "llama3.2:3b",
                "mistral:7b", "mistral:latest",
                "qwen2.5:7b", "qwen2.5:3b",
                "gemma2:9b", "gemma2:2b",
                "phi3:mini",
            ]
            for m in preferred:
                if m in models:
                    return m

            # If nothing installed, default to llama3.1:8b (will need to pull)
            if models:
                return models[0]
            return "llama3.1:8b"

        elif self.backend == "anthropic":
            return "claude-sonnet-4-6"

        return "offline"

    def is_available(self) -> bool:
        """Check if the LLM backend is ready."""
        return self.backend != "offline"

    def generate(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> dict:
        """Generate a response from the LLM.

        Returns:
            dict with keys: text, tokens_in, tokens_out, time_ms, model, backend
        """
        start = time.time()
        self.stats["calls"] += 1

        if self.backend not in ("ollama", "anthropic"):
            return {"text": "", "error": "No LLM backend available", "backend": "offline",
                    "model": "none", "tokens_in": 0, "tokens_out": 0, "time_ms": 0}

        # Bug fix #3: Retry with exponential backoff on transient failures
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.backend == "ollama":
                    result = self._generate_ollama(prompt, system, json_mode, max_tokens, temperature)
                else:
                    result = self._generate_anthropic(prompt, system, max_tokens, temperature)

                elapsed = int((time.time() - start) * 1000)
                result["time_ms"] = elapsed
                self.stats["tokens_in"] += result.get("tokens_in", 0)
                self.stats["tokens_out"] += result.get("tokens_out", 0)
                self.stats["total_ms"] += elapsed
                return result

            except Exception as e:
                last_error = e
                is_retryable = any(kw in str(e).lower() for kw in
                                   ["429", "rate", "timeout", "500", "502", "503", "overloaded"])
                if is_retryable and attempt < self.max_retries - 1:
                    wait = (2 ** attempt) + (time.time() % 1)  # exp backoff + jitter
                    logger.warning(f"Retry {attempt+1}/{self.max_retries} after {wait:.1f}s: {e}")
                    self.stats["retries"] += 1
                    time.sleep(wait)
                else:
                    break

        self.stats["errors"] += 1
        return {"text": "", "error": str(last_error), "backend": self.backend,
                "model": self.model, "tokens_in": 0, "tokens_out": 0,
                "time_ms": int((time.time() - start) * 1000)}

    def _generate_ollama(
        self, prompt: str, system: str, json_mode: bool,
        max_tokens: int, temperature: float,
    ) -> dict:
        """Generate via Ollama REST API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        return {
            "text": data.get("response", ""),
            "backend": "ollama",
            "model": self.model,
            "tokens_in": data.get("prompt_eval_count", 0),
            "tokens_out": data.get("eval_count", 0),
        }

    def _generate_anthropic(
        self, prompt: str, system: str, max_tokens: int, temperature: float,
    ) -> dict:
        """Generate via Anthropic API."""
        # Bug fix #1: reuse client (initialized in __init__)
        if self._anthropic_client is None:
            from anthropic import Anthropic
            self._anthropic_client = Anthropic()

        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,  # Bug fix #2: actually pass temperature
        }
        if system:
            kwargs["system"] = system

        response = self._anthropic_client.messages.create(**kwargs)
        text = response.content[0].text

        return {
            "text": text,
            "backend": "anthropic",
            "model": self.model,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }

    def generate_json(
        self, prompt: str, system: str = "", temperature: float = 0.5,
        max_tokens: int = 512,
    ) -> dict | None:
        """Generate and parse JSON response. Returns parsed dict or None."""
        result = self.generate(prompt, system=system, json_mode=True, temperature=temperature, max_tokens=max_tokens)
        text = result.get("text", "")
        if not text:
            return None

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
        return None

    def pull_model(self, model: str | None = None) -> bool:
        """Pull a model in Ollama. Returns True if successful."""
        if self.backend != "ollama":
            return False

        model = model or self.model
        print(f"Pulling {model}... (this may take a few minutes)")
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=600,
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"Pull failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Return usage statistics."""
        avg_ms = self.stats["total_ms"] / max(self.stats["calls"], 1)
        return {
            **self.stats,
            "avg_ms_per_call": round(avg_ms),
            "backend": self.backend,
            "model": self.model,
        }


# ── Agent prompt templates for LLM-powered simulation ──
# Designed to maximize diversity and prevent mode collapse.
# Key techniques:
# - Explicit persona anchoring ("as a ___, you would...")
# - Contrarian instructions for contrarian archetypes
# - Prohibition against defaulting to 0.50
# - Requirement to differ from peers

AGENT_SYSTEM_PROMPT = """You are {name}, a {background}

Your personality: {personality}

CRITICAL RULES:
- You MUST answer from your specific professional perspective, not as a generic analyst
- Your probability estimate should reflect YOUR expertise and biases, not a balanced view
- Do NOT default to 0.50 — take a position based on your background
- {diversity_instruction}
- Respond with ONLY a valid JSON object, no other text"""

AGENT_INITIAL_PROMPT = """As a {background_short}, estimate the probability of this question resolving YES:

Question: {question}

Context:
{context}

Pressures FOR YES: {pressures_yes}
Pressures FOR NO: {pressures_no}
Key uncertainties: {pressures_uncertain}

{persona_nudge}

You MUST give a specific probability reflecting your professional view. Avoid 0.45-0.55 unless you have strong reasons — most questions have directional evidence.

Return JSON:
{{
  "initial_score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<3-5 sentences drawing on your SPECIFIC expertise, not generic analysis>",
  "key_factors": ["factor1", "factor2", "factor3"]
}}"""

AGENT_DELIBERATION_PROMPT = """You are {name}, a {background_short}. Your personality: {personality}.

Question: {question}
Your current estimate: P(YES) = {current_score:.2f}

Peer opinions this round:
{peer_opinions}

Your prior reasoning:
{memory}

{deliberation_nudge}

Update your estimate. You may move toward peers if their evidence is compelling, but do NOT simply average. Maintain your professional perspective.

Return JSON:
{{
  "updated_score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reflection": "<2-3 sentences on what specifically changed or didn't change your mind>",
  "new_insight": "<one concrete insight from this round>"
}}"""

# Per-archetype diversity instructions
DIVERSITY_INSTRUCTIONS = {
    "analyst": "You are evidence-driven. Only move your estimate if you see hard data.",
    "calibrator": "You focus on base rates. Anchor to historical frequency of similar events.",
    "contrarian": "You ACTIVELY challenge consensus. If peers cluster around a number, explain why they might be wrong. Your estimate should often differ from the group mean by at least 0.15.",
    "creative": "You consider unlikely scenarios others miss. Think about tail risks, black swans, and second-order effects. Your estimate can be more extreme than others.",
}

# Persona-specific nudges for initial forecast
PERSONA_NUDGES = {
    "analyst": "As an analyst, focus on the quantitative evidence. What do the numbers say?",
    "calibrator": "As a calibrator, what is the historical base rate for this type of event? Start there and adjust.",
    "contrarian": "As a contrarian, what is the consensus view? Now argue against it. Your estimate should challenge the obvious answer.",
    "creative": "As a creative thinker, what are the scenarios that would surprise everyone? Consider both extreme YES and extreme NO outcomes.",
}

# Deliberation nudges
DELIBERATION_NUDGES = {
    "analyst": "Evaluate the quality of peer evidence. Only shift if they cite data you haven't considered.",
    "calibrator": "Check: are peers anchoring to narratives instead of base rates? Correct for that.",
    "contrarian": "If peers are converging, resist. The value you add is maintaining an independent view. Only converge if they present evidence that specifically refutes your position.",
    "creative": "What are peers missing? Is there a scenario none of them considered?",
}

# Context-to-anchor: when no market price is available, LLM estimates base rate
ANCHOR_PROMPT = """You are a calibrated forecaster. Given the question and context below, estimate the probability of YES.

Question: {question}

Context: {context}

Consider:
1. Historical base rates for this type of event
2. The specific evidence in the context
3. Known biases (people overestimate dramatic events, underestimate inertia)

IMPORTANT: Be precise with low probabilities. Do NOT round everything to 5%.
Use the full range:
- 1-2%: virtually impossible (violates physics, no precedent ever)
- 3-5%: extremely unlikely (would require multiple unprecedented events)
- 8-12%: unlikely but has some precedent or plausible pathway
- 15-25%: possible but against base rates
- 30-45%: could go either way, leaning NO
- 50%: true coin flip (avoid unless genuinely uncertain)
- 55-70%: could go either way, leaning YES
- 75-90%: likely, strong evidence supports it
- 95-99%: near certain, would require extraordinary circumstances to fail

Respond with ONLY a JSON object:
{{"probability": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""
