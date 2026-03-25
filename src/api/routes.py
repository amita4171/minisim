"""
Endpoint handlers for the MiniSim API.

All route functions are registered on an APIRouter and included
by the main app in app.py.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import HTMLResponse

from src.api.auth import rate_limit_dependency, verify_api_key
from src.api.deps import get_database, get_request_log_db
from src.api.models import (
    MetricsResponse,
    PredictConfig,
    PredictRequest,
    PredictResponse,
    PredictStatus,
    ResolveRequest,
    ResolveResponse,
)
from src.db.database import Database

logger = logging.getLogger(__name__)

# In-memory prediction store (shared with root api.py for backward compat)
_predictions: dict[str, dict] = {}


# ── Background task: run prediction ──


def _run_prediction(pred_id: str, request: PredictRequest) -> None:
    """Run swarm prediction in background."""
    try:
        if request.config.mode == "smart":
            from src.core.router import routed_predict
            from src.core.llm_engine import LLMEngine
            engine = LLMEngine(model=request.config.model)
            result = routed_predict(
                question=request.question,
                context=request.context,
                n_agents=request.config.n_agents,
                market_price=request.market_price,
                peer_sample_size=request.config.peer_sample_size,
                engine=engine,
                max_rounds=request.config.n_rounds,
            )
        elif request.config.mode == "offline":
            from src.core.offline_engine import swarm_score_offline
            result = swarm_score_offline(
                question=request.question,
                context=request.context,
                n_agents=request.config.n_agents,
                rounds=request.config.n_rounds,
                market_price=request.market_price,
                peer_sample_size=request.config.peer_sample_size,
            )
        else:
            from src.core.llm_simulation import run_llm_simulation
            from src.core.llm_engine import LLMEngine
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
        except Exception as e:
            logger.warning(f"Failed to save prediction to database: {e}")

    except Exception as e:
        _predictions[pred_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        })


# ── Routers ──

# Authenticated endpoints (v1 prefix)
v1_router = APIRouter(prefix="/v1")

# Public endpoints (no prefix)
public_router = APIRouter()


# ── V1 Endpoints ──


@v1_router.post("/predict", response_model=PredictResponse)
async def create_prediction(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(rate_limit_dependency),
):
    """Submit a prediction request. Returns immediately with a prediction_id."""
    start = time.time()
    pred_id = f"pred_{uuid.uuid4().hex[:12]}"

    _predictions[pred_id] = {
        "prediction_id": pred_id,
        "status": "processing",
        "question": request.question,
        "market_price": request.market_price,
        "config": request.config.model_dump(),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    background_tasks.add_task(_run_prediction, pred_id, request)

    # Log the request
    latency_ms = (time.time() - start) * 1000
    try:
        log_db = get_request_log_db()
        log_db.log_request(
            api_key=api_key,
            question=request.question,
            mode=request.config.mode,
            latency_ms=latency_ms,
            status="accepted",
        )
    except Exception as e:
        logger.warning(f"Failed to log request: {e}")

    return PredictResponse(
        prediction_id=pred_id,
        status=PredictStatus.processing,
        question=request.question,
        market_price=request.market_price,
        config=request.config.model_dump(),
        created_at=_predictions[pred_id]["created_at"],
    )


@v1_router.get("/predict/{prediction_id}", response_model=PredictResponse)
async def get_prediction(
    prediction_id: str,
    api_key: str = Depends(rate_limit_dependency),
):
    """Poll for prediction result."""
    if prediction_id not in _predictions:
        raise HTTPException(status_code=404, detail="Prediction not found")

    pred = _predictions[prediction_id]
    return PredictResponse(**{k: v for k, v in pred.items() if k != "db_id"})


@v1_router.post("/predict/{prediction_id}/resolve", response_model=ResolveResponse)
async def resolve_prediction(
    prediction_id: str,
    request: ResolveRequest,
    api_key: str = Depends(rate_limit_dependency),
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
        except Exception as e:
            logger.warning(f"Failed to update resolution in database: {e}")

    return ResolveResponse(
        prediction_id=prediction_id,
        swarm_brier=swarm_brier,
        market_brier=market_brier,
        swarm_beat_market=beat,
    )


@v1_router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(api_key: str = Depends(rate_limit_dependency)):
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


@v1_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "predictions_in_memory": len(_predictions),
    }


# ── Public Endpoints ──


@public_router.get("/accuracy", response_class=HTMLResponse)
async def accuracy_dashboard():
    """Public-facing accuracy dashboard -- the sales asset."""
    try:
        db = Database()
        metrics = db.get_metrics()
        cat_data = db.get_category_accuracy()
        db.close()
    except Exception:
        metrics = {"total": 0, "resolved": 0, "pending": 0}
        cat_data = []

    # Load calibration if available
    cal_curve = {}
    try:
        with open("results/calibration_model_offline.json") as f:
            cal = json.load(f)
            cal_curve = cal.get("calibration_curve", {})
    except Exception as e:
        logger.debug(f"Calibration file not available: {e}")

    total = metrics.get("total", 0)
    resolved = metrics.get("resolved", 0)
    brier = metrics.get("swarm_brier", "---")
    market_brier = metrics.get("market_brier", "---")
    win_rate = metrics.get("win_rate", 0)
    if isinstance(win_rate, float):
        win_rate = f"{win_rate*100:.0f}%"

    # Build calibration rows
    cal_rows = ""
    for bucket, d in sorted(cal_curve.items()):
        gap = d.get("gap", 0)
        color = "#2ca02c" if abs(gap) < 0.1 else "#ff7f0e" if abs(gap) < 0.2 else "#d62728"
        cal_rows += f'<tr><td>{bucket}</td><td>{d["count"]}</td><td>{d["mean_predicted"]:.3f}</td><td>{d["actual_rate"]:.3f}</td><td style="color:{color}">{gap:+.3f}</td></tr>'

    # Build category rows
    cat_rows = ""
    for c in cat_data:
        cat_rows += f'<tr><td>{c["category"]}</td><td>{c["n"]}</td><td>{c["avg_swarm_brier"]:.4f}</td><td>{c.get("avg_market_brier", "---")}</td><td>{c.get("wins", 0)}/{c["n"]}</td></tr>'

    return f"""<!DOCTYPE html>
<html><head><title>MiniSim Accuracy Dashboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ color: #1f3864; }} h2 {{ color: #2f5496; border-bottom: 2px solid #d6e4f0; padding-bottom: 8px; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
.metric {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; color: #1f3864; }}
.metric .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th {{ background: #2f5496; color: white; padding: 10px; text-align: left; }}
td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f0f4ff; }}
.method {{ background: #e8f0fe; padding: 16px; border-radius: 8px; margin: 16px 0; font-size: 14px; }}
footer {{ margin-top: 40px; padding: 20px 0; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
</style></head><body>
<h1>MiniSim Accuracy Dashboard</h1>
<p>Real-time calibration metrics for the MiniSim swarm prediction engine.</p>

<div class="metrics">
  <div class="metric"><div class="value">{total}</div><div class="label">Total Predictions</div></div>
  <div class="metric"><div class="value">{resolved}</div><div class="label">Resolved</div></div>
  <div class="metric"><div class="value">{brier}</div><div class="label">Swarm Brier Score</div></div>
  <div class="metric"><div class="value">{win_rate}</div><div class="label">Win Rate vs Market</div></div>
</div>

<h2>Calibration Curve (544 questions)</h2>
<p>Predicted probability vs actual resolution rate. Perfect calibration = gap of 0.000.</p>
<table>
<tr><th>Bucket</th><th>N</th><th>Predicted</th><th>Actual</th><th>Gap</th></tr>
{cal_rows}
</table>

<h2>Accuracy by Category</h2>
<table>
<tr><th>Category</th><th>N</th><th>Swarm Brier</th><th>Market Brier</th><th>Wins</th></tr>
{cat_rows}
</table>

<div class="method">
<strong>Method:</strong> {total} questions evaluated using 20-40 diverse agent archetypes deliberating across 2-4 structured rounds (evidence exchange, critique, rebuttal). Aggregation: 60% calibrated confidence-weighted + 40% extremized (Metaculus-style). Temperature stratification: analysts (T=0.3), calibrators (T=0.5), contrarians (T=0.9), creatives (T=1.2).
</div>

<footer>MiniSim Swarm Prediction Engine &middot; Updated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</footer>
</body></html>"""
