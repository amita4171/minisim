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
import os
import time
from typing import Optional

import requests


class LLMEngine:
    """Unified LLM interface with auto-detection and fallback."""

    def __init__(
        self,
        backend: str = "auto",
        model: str | None = None,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Args:
            backend: "ollama", "anthropic", or "auto" (tries ollama first)
            model: Model name. If None, auto-selects best available.
            ollama_url: Ollama server URL
        """
        self.ollama_url = ollama_url
        self.backend = backend
        self.model = model
        self.stats = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0, "total_ms": 0}

        if backend == "auto":
            self.backend = self._detect_backend()

        if self.model is None:
            self.model = self._select_model()

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
        except (requests.ConnectionError, requests.Timeout):
            pass

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

        try:
            if self.backend == "ollama":
                result = self._generate_ollama(prompt, system, json_mode, max_tokens, temperature)
            elif self.backend == "anthropic":
                result = self._generate_anthropic(prompt, system, max_tokens, temperature)
            else:
                return {"text": "", "error": "No LLM backend available", "backend": "offline",
                        "model": "none", "tokens_in": 0, "tokens_out": 0, "time_ms": 0}

            elapsed = int((time.time() - start) * 1000)
            result["time_ms"] = elapsed
            self.stats["tokens_in"] += result.get("tokens_in", 0)
            self.stats["tokens_out"] += result.get("tokens_out", 0)
            self.stats["total_ms"] += elapsed
            return result

        except Exception as e:
            self.stats["errors"] += 1
            return {"text": "", "error": str(e), "backend": self.backend,
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
            timeout=120,
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
        from anthropic import Anthropic
        client = Anthropic()

        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
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
    ) -> dict | None:
        """Generate and parse JSON response. Returns parsed dict or None."""
        result = self.generate(prompt, system=system, json_mode=True, temperature=temperature)
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

AGENT_SYSTEM_PROMPT = """You are a forecasting agent with this profile:
- Background: {background}
- Personality: {personality}
- Temperature tier: {temp_tier}

You make calibrated probability estimates on prediction questions.
You must respond with ONLY a JSON object, no other text."""

AGENT_INITIAL_PROMPT = """Question: {question}

Context:
{context}

World model pressures:
FOR YES: {pressures_yes}
FOR NO: {pressures_no}
UNCERTAIN: {pressures_uncertain}

Provide your initial probability estimate for this question resolving YES.

Return JSON:
{{
  "initial_score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<3-5 sentences from your specific expertise>",
  "key_factors": ["factor1", "factor2", "factor3"]
}}"""

AGENT_DELIBERATION_PROMPT = """Question: {question}

Your current estimate: P(YES) = {current_score:.2f}
Your background: {background}

Peer opinions this round:
{peer_opinions}

Your memory of prior reasoning:
{memory}

Reflect on peer opinions and update your estimate.

Return JSON:
{{
  "updated_score": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reflection": "<2-3 sentences on how peers influenced your view>",
  "new_insight": "<one new insight from this round>"
}}"""
