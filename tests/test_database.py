"""Tests for SQLite database layer."""
import os
import tempfile
import pytest
from src.db.database import Database


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path=path)
    yield database
    database.close()
    os.unlink(path)


def test_log_prediction(db):
    pred_id = db.log_prediction(
        question="Will it rain?",
        swarm_probability=0.65,
        market_price=0.50,
        source="test",
    )
    assert pred_id > 0


def test_get_predictions(db):
    db.log_prediction("Q1", 0.3, 0.4, "test")
    db.log_prediction("Q2", 0.7, 0.6, "test")
    preds = db.get_predictions()
    assert len(preds) == 2


def test_resolve_prediction(db):
    pred_id = db.log_prediction("Will X?", 0.7, 0.5, "test")
    result = db.resolve(pred_id, 1.0)
    assert result is True

    preds = db.get_predictions(resolved_only=True)
    assert len(preds) == 1
    assert preds[0]["swarm_brier"] is not None
    assert preds[0]["swarm_brier"] == pytest.approx(0.09, abs=0.01)  # (0.7-1.0)^2


def test_resolve_computes_market_brier(db):
    pred_id = db.log_prediction("Will Y?", 0.8, 0.3, "test")
    db.resolve(pred_id, 1.0)

    preds = db.get_predictions(resolved_only=True)
    assert preds[0]["market_brier"] == pytest.approx(0.49, abs=0.01)  # (0.3-1.0)^2
    assert preds[0]["swarm_beat_market"] == 1  # swarm was closer


def test_get_metrics_empty(db):
    metrics = db.get_metrics()
    assert metrics["total"] == 0
    assert metrics["resolved"] == 0


def test_get_metrics_with_data(db):
    p1 = db.log_prediction("Q1", 0.7, 0.5, "test")
    p2 = db.log_prediction("Q2", 0.3, 0.5, "test")
    db.resolve(p1, 1.0)
    db.resolve(p2, 0.0)

    metrics = db.get_metrics()
    assert metrics["total"] == 2
    assert metrics["resolved"] == 2
    assert metrics["swarm_brier"] is not None


def test_log_with_agents(db):
    agents = [
        {"background_category": "Economist", "initial_score": 0.4, "final_score": 0.6},
        {"background_category": "Trader", "initial_score": 0.3, "final_score": 0.5},
    ]
    pred_id = db.log_prediction("Q", 0.5, 0.5, "test", agents=agents, category="econ")
    db.resolve(pred_id, 1.0)

    accuracy = db.get_archetype_accuracy()
    assert len(accuracy) == 2
