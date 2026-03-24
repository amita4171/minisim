"""
MiniSim Production API — FastAPI B2B forecasting service.

Endpoints:
  POST /v1/predict          — Submit a prediction request (async)
  GET  /v1/predict/{id}     — Poll for result
  POST /v1/predict/{id}/resolve — Submit ground truth
  GET  /v1/metrics          — Aggregate accuracy dashboard
  GET  /v1/health           — Health check

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import secrets
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from src.database import Database

app = FastAPI(
    title="MiniSim Forecasting API",
    description="Swarm intelligence prediction engine — multi-agent deliberation for calibrated probability estimates.",
    version="0.1.0",
)

# In-memory prediction store (replace with Redis for production persistence)
_predictions: dict[str, dict] = {}

# API key auth (simple bearer token — replace with DB-backed keys for production)
API_KEYS = set(os.environ.get("MINISIM_API_KEYS", "demo-key-12345").split(","))


def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token


# ── Request/Response models ──

class PredictConfig(BaseModel):
    n_agents: int = Field(default=20, ge=5, le=200)
    n_rounds: int = Field(default=3, ge=1, le=10)
    mode: str = Field(default="offline", pattern="^(offline|llm-ollama|llm-anthropic)$")
    model: Optional[str] = None
    peer_sample_size: int = Field(default=5, ge=3, le=15)


class PredictRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500)
    context: str = Field(default="", max_length=2000)
    market_price: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    config: PredictConfig = Field(default_factory=PredictConfig)


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


class ResolveRequest(BaseModel):
    resolution: float = Field(..., ge=0.0, le=1.0)


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


# ── Background task: run prediction ──

def _run_prediction(pred_id: str, request: PredictRequest):
    """Run swarm prediction in background."""
    try:
        if request.config.mode == "offline":
            from src.offline_engine import swarm_score_offline
            result = swarm_score_offline(
                question=request.question,
                context=request.context,
                n_agents=request.config.n_agents,
                rounds=request.config.n_rounds,
                market_price=request.market_price,
                peer_sample_size=request.config.peer_sample_size,
            )
        else:
            from src.llm_simulation import run_llm_simulation
            from src.llm_engine import LLMEngine
            backend = "ollama" if request.config.mode == "llm-ollama" else "anthropic"
            engine = LLMEngine(backend=backend, model=request.config.model)
            result = run_llm_simulation(
                question=request.question,
                context=request.context,
                n_agents=request.config.n_agents,
                n_rounds=request.config.n_rounds,
                market_price=request.market_price,
                peer_sample_size=request.config.peer_sample_size,
                engine=engine,
            )

        _predictions[pred_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "swarm_probability": result.get("swarm_probability_yes"),
            "confidence_interval": result.get("confidence_interval"),
            "edge": result.get("edge"),
            "top_yes_voices": result.get("top_yes_voices", [])[:3],
            "top_no_voices": result.get("top_no_voices", [])[:3],
            "mind_changers": result.get("mind_changers", [])[:3],
            "dissenting_voices": result.get("dissenting_voices", [])[:3],
            "opinion_clusters": result.get("opinion_clusters"),
            "reasoning_shift_summary": result.get("reasoning_shift_summary"),
            "diversity_score": result.get("diversity_score"),
            "agents_from_llm": result.get("agents_from_llm", 0),
            "agents_from_fallback": result.get("agents_from_fallback", 0),
            "timing": result.get("timing"),
        })

        # Log to database
        try:
            db = Database()
            db_id = db.log_prediction(
                question=request.question,
                swarm_probability=result["swarm_probability_yes"],
                market_price=request.market_price,
                source="api",
                n_agents=request.config.n_agents,
                n_rounds=request.config.n_rounds,
                mode=request.config.mode,
                confidence_interval=result.get("confidence_interval"),
                diversity_score=result.get("diversity_score", 0),
            )
            _predictions[pred_id]["db_id"] = db_id
            db.close()
        except Exception:
            pass

    except Exception as e:
        _predictions[pred_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        })


# ── Endpoints ──

@app.post("/v1/predict", response_model=PredictResponse)
async def create_prediction(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Submit a prediction request. Returns immediately with a prediction_id."""
    pred_id = f"pred_{uuid.uuid4().hex[:12]}"

    _predictions[pred_id] = {
        "prediction_id": pred_id,
        "status": "processing",
        "question": request.question,
        "market_price": request.market_price,
        "config": request.config.dict(),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    background_tasks.add_task(_run_prediction, pred_id, request)

    return PredictResponse(
        prediction_id=pred_id,
        status=PredictStatus.processing,
        question=request.question,
        market_price=request.market_price,
        config=request.config.dict(),
        created_at=_predictions[pred_id]["created_at"],
    )


@app.get("/v1/predict/{prediction_id}", response_model=PredictResponse)
async def get_prediction(
    prediction_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Poll for prediction result."""
    if prediction_id not in _predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")

    pred = _predictions[prediction_id]
    return PredictResponse(**{k: v for k, v in pred.items() if k != "db_id"})


@app.post("/v1/predict/{prediction_id}/resolve", response_model=ResolveResponse)
async def resolve_prediction(
    prediction_id: str,
    request: ResolveRequest,
    api_key: str = Depends(verify_api_key),
):
    """Submit ground truth resolution for a prediction."""
    if prediction_id not in _predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")

    pred = _predictions[prediction_id]
    if pred["status"] != "completed":
        raise HTTPException(status_code=400, detail="Prediction not yet completed")

    swarm_p = pred.get("swarm_probability", 0.5)
    market_p = pred.get("market_price")

    swarm_brier = round((swarm_p - request.resolution) ** 2, 6)
    market_brier = round((market_p - request.resolution) ** 2, 6) if market_p is not None else None
    beat = swarm_brier < market_brier if market_brier is not None else None

    # Update database
    db_id = pred.get("db_id")
    if db_id:
        try:
            db = Database()
            db.resolve(db_id, request.resolution)
            db.close()
        except Exception:
            pass

    return ResolveResponse(
        prediction_id=prediction_id,
        swarm_brier=swarm_brier,
        market_brier=market_brier,
        swarm_beat_market=beat,
    )


@app.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Aggregate accuracy metrics from all predictions."""
    try:
        db = Database()
        metrics = db.get_metrics()
        cat_accuracy = db.get_category_accuracy()
        db.close()

        return MetricsResponse(
            total_predictions=metrics.get("total", 0),
            resolved=metrics.get("resolved", 0),
            pending=metrics.get("pending", 0),
            avg_brier=metrics.get("swarm_brier"),
            market_brier=metrics.get("market_brier"),
            win_rate=metrics.get("win_rate"),
            accuracy_by_category={r["category"]: {"brier": r["avg_swarm_brier"], "n": r["n"]} for r in cat_accuracy} if cat_accuracy else None,
        )
    except Exception:
        return MetricsResponse(total_predictions=0, resolved=0, pending=0)


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "predictions_in_memory": len(_predictions),
    }
