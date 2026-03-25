"""
MiniSim -- Agent-Based Prediction Engine

Sub-packages:
    src.core      -- prediction engine (aggregator, calibration, router, LLM engines)
    src.agents    -- agent archetypes, world building, simulation loop
    src.markets   -- API clients (Kalshi, Polymarket, Manifold, PredictIt, Metaculus)
    src.research  -- web search, data feeds, EDGAR pipeline
    src.db        -- persistence (SQLite database, track record)

Backward-compatible re-exports so that old import paths still work:
    from src.offline_engine import swarm_score_offline   # still works
    from src.aggregator import aggregate                 # still works
"""
import sys as _sys

# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------
# Register sub-package modules under their old ``src.<name>`` paths so that
# ``from src.offline_engine import swarm_score_offline`` keeps working.
# New code should use the canonical sub-package path, e.g.:
#   from src.core.offline_engine import swarm_score_offline

def _compat(old_name: str, new_path: str) -> None:
    """Lazily alias ``src.<old_name>`` -> ``<new_path>``."""
    import importlib
    mod = importlib.import_module(new_path)
    _sys.modules[f"src.{old_name}"] = mod
    globals()[old_name] = mod

# core/
_compat("aggregator",      "src.core.aggregator")
_compat("calibration",     "src.core.calibration")
_compat("router",          "src.core.router")
_compat("llm_engine",      "src.core.llm_engine")
_compat("llm_simulation",  "src.core.llm_simulation")
_compat("offline_engine",  "src.core.offline_engine")

# agents/
_compat("archetypes",      "src.agents.archetypes")
_compat("alpha",           "src.agents.alpha")
_compat("world_templates", "src.agents.world_templates")
_compat("agent_factory",   "src.agents.agent_factory")
_compat("world_builder",   "src.agents.world_builder")
_compat("simulation_loop", "src.agents.simulation_loop")

# markets/
_compat("kalshi_client",      "src.markets.kalshi_client")
_compat("polymarket_client",  "src.markets.polymarket_client")
_compat("manifold_client",    "src.markets.manifold_client")
_compat("predictit_client",   "src.markets.predictit_client")
_compat("metaculus_client",   "src.markets.metaculus_client")
_compat("cross_platform",     "src.markets.cross_platform")
_compat("kalshi_bridge",      "src.markets.kalshi_bridge")

# research/
_compat("web_research",    "src.research.web_research")
_compat("data_feeds",      "src.research.data_feeds")
_compat("edgar_pipeline",  "src.research.edgar_pipeline")

# db/
_compat("database",        "src.db.database")
_compat("track_record",    "src.db.track_record")

del _compat  # clean up namespace
