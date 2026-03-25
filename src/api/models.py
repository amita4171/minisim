"""
Pydantic request/response models for the MiniSim API.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Request models ──


class PredictConfig(BaseModel):
    n_agents: int = Field(default=20, ge=5, le=200)
    n_rounds: int = Field(default=3, ge=1, le=10)
    mode: str = Field(default="smart", pattern="^(offline|llm-ollama|llm-anthropic|smart)$")
    model: Optional[str] = None
    peer_sample_size: int = Field(default=5, ge=3, le=15)


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500)
    context: str = Field(default="", max_length=2000)
    market_price: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    config: PredictConfig = Field(default_factory=PredictConfig)


class ResolveRequest(BaseModel):
    resolution: float = Field(..., ge=0.0, le=1.0)


# ── Response models ──


class PredictStatus(str, Enum):
    processing = "processing"
    completed = "completed"
    failed = "failed"


class PredictResponse(BaseModel):
    prediction_id: str
    status: PredictStatus
    question: str
    swarm_probability: Optional[float] = None
    confidence_interval: Optional[list[float]] = None
    market_price: Optional[float] = None
    edge: Optional[float] = None
    top_yes_voices: Optional[list[dict]] = None
    top_no_voices: Optional[list[dict]] = None
    mind_changers: Optional[list[dict]] = None
    dissenting_voices: Optional[list[dict]] = None
    opinion_clusters: Optional[list[dict]] = None
    reasoning_shift_summary: Optional[str] = None
    diversity_score: Optional[float] = None
    agents_from_llm: Optional[int] = None
    agents_from_fallback: Optional[int] = None
    timing: Optional[dict] = None
    config: Optional[dict] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ResolveResponse(BaseModel):
    prediction_id: str
    swarm_brier: float
    market_brier: Optional[float] = None
    swarm_beat_market: Optional[bool] = None


class MetricsResponse(BaseModel):
    total_predictions: int
    resolved: int
    pending: int
    avg_brier: Optional[float] = None
    market_brier: Optional[float] = None
    win_rate: Optional[float] = None
    calibration_ece: Optional[float] = None
    accuracy_by_category: Optional[dict] = None
