"""
GraphRAG-style knowledge graph builder from seed context.
Extracts entities, relationships, pressures, and timeline using Claude.
"""

import json
import time
from src.utils import get_anthropic_client as _get_client

WORLD_BUILD_PROMPT = """You are a knowledge graph builder for prediction market analysis.

Given a prediction question and optional context, extract a structured world model.

Question: {question}
Context: {context}

Return a JSON object with these fields:
{{
  "entities": [
    {{"name": "...", "type": "person|org|policy|event|metric|concept", "description": "...", "relevance": "high|medium|low"}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "relation": "...", "strength": "strong|moderate|weak"}}
  ],
  "pressures": {{
    "for_yes": ["pressure 1", "pressure 2", ...],
    "for_no": ["pressure 1", "pressure 2", ...],
    "uncertain": ["factor 1", "factor 2", ...]
  }},
  "timeline": [
    {{"date_or_period": "...", "event": "...", "impact": "..."}}
  ],
  "base_rate_estimate": 0.5,
  "key_uncertainties": ["uncertainty 1", "uncertainty 2", ...]
}}

Be thorough. Include at least 5 entities, 5 relationships, 3+ pressures per side, and 3+ timeline events.
Return ONLY valid JSON, no markdown."""


def build_world(question: str, context: str = "") -> dict:
    """Build a GraphRAG-style world model from a prediction question."""
    start = time.time()

    response = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": WORLD_BUILD_PROMPT.format(
                    question=question,
                    context=context or "No additional context provided. Use your knowledge."
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    world = json.loads(raw)
    elapsed_ms = int((time.time() - start) * 1000)
    world["_build_time_ms"] = elapsed_ms

    return world
