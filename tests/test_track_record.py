"""Tests for the track record persistence system."""
import os
import json
import tempfile
import pytest

from src.track_record import TrackRecord


@pytest.fixture
def tr(tmp_path):
    """Create a TrackRecord with a temp file."""
    path = str(tmp_path / "track_record.json")
    return TrackRecord(path=path)


def test_log_prediction(tr):
    pred = tr.log_prediction(
        question="Will it rain?",
        swarm_probability=0.65,
        market_price=0.50,
        source="test",
    )
    assert pred["id"] == 0
    assert pred["swarm_probability"] == 0.65
    assert pred["resolution"] is None


def test_predictions_persist(tmp_path):
    path = str(tmp_path / "tr.json")
    tr1 = TrackRecord(path=path)
    tr1.log_prediction("Q1", 0.5, 0.5, "test")
    tr1.log_prediction("Q2", 0.7, 0.6, "test")

    # Reload from disk
    tr2 = TrackRecord(path=path)
    assert len(tr2.predictions) == 2


def test_resolve_computes_brier(tr):
    tr.log_prediction("Will X?", 0.80, 0.50, "test")
    result = tr.resolve(0, 1.0)

    assert result is not None
    assert result["resolution"] == 1.0
    assert result["swarm_brier"] == pytest.approx(0.04, abs=0.01)  # (0.8-1.0)^2
    assert result["swarm_beat_market"] is True  # 0.04 < 0.25


def test_resolve_returns_none_for_invalid_id(tr):
    result = tr.resolve(999, 1.0)
    assert result is None


def test_get_resolved_and_unresolved(tr):
    tr.log_prediction("Q1", 0.5, 0.5, "test")
    tr.log_prediction("Q2", 0.7, 0.6, "test")
    tr.resolve(0, 1.0)

    assert len(tr.get_resolved()) == 1
    assert len(tr.get_unresolved()) == 1


def test_compute_metrics_empty(tr):
    metrics = tr.compute_metrics()
    assert metrics["n_resolved"] == 0


def test_compute_metrics_with_data(tr):
    tr.log_prediction("Q1", 0.70, 0.50, "test")
    tr.log_prediction("Q2", 0.30, 0.50, "test")
    tr.resolve(0, 1.0)
    tr.resolve(1, 0.0)

    metrics = tr.compute_metrics()
    assert metrics["n_resolved"] == 2
    assert metrics["overall_swarm_brier"] is not None
    assert metrics["win_rate"] is not None


def test_compute_metrics_no_division_by_zero(tr):
    """Should not crash when no predictions are resolved."""
    tr.log_prediction("Q1", 0.5, 0.5, "test")
    metrics = tr.compute_metrics()
    assert metrics["n_resolved"] == 0


def test_edge_prediction(tr):
    tr.log_prediction("Q1", 0.01, 0.01, "test")
    tr.resolve(0, 0.0)
    metrics = tr.compute_metrics()
    assert metrics["overall_swarm_brier"] < 0.01
