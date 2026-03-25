"""
FastAPI application factory for the MiniSim API.

Creates the app, attaches middleware, CORS, and includes all routers.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import public_router, v1_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MiniSim Forecasting API",
    description="Swarm intelligence prediction engine -- multi-agent deliberation for calibrated probability estimates.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ──

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header to every response."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


# ── Include routers ──

app.include_router(v1_router)
app.include_router(public_router)
