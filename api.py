"""
MiniSim Production API -- FastAPI B2B forecasting service.

This module re-exports the app and internal state from the refactored
src.api package, preserving backward compatibility for existing imports:

    from api import app, _predictions
    from api import API_KEYS

Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from src.api.app import app  # noqa: F401
from src.api.routes import _predictions  # noqa: F401
from src.api.auth import API_KEYS, verify_api_key  # noqa: F401
from src.api.models import (  # noqa: F401
    PredictConfig,
    PredictRequest,
    PredictResponse,
    PredictStatus,
    ResolveRequest,
    ResolveResponse,
    MetricsResponse,
)
