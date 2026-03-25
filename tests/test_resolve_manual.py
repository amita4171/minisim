"""Tests for scripts/resolve_manual.py — manual Metaculus resolution tool."""
import json
import os
import tempfile

import pytest

from src.db.database import Database
from scripts.resolve_manual import (
    fetch_unresolved,
    resolve_batch,
    resolve_interactive,
    print_summary,
)


@pytest.fixture
def db():
    """Create a temporary database seeded with unresolved metaculus predictions."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path=path)

    # Seed 3 unresolved metaculus predictions
    database.log_prediction(
        question="Will AI pass the bar exam by 2026?",
        swarm_probability=0.80,
        market_price=0.60,
        source="metaculus",
        ticker="42684",
    )
    database.log_prediction(
        question="Will BTC exceed $100k by end of 2025?",
        swarm_probability=0.35,
        market_price=0.40,
        source="metaculus",
        ticker="42779",
    )
    database.log_prediction(
        question="Will there be a Mars mission by 2030?",
        swarm_probability=0.55,
        market_price=0.50,
        source="metaculus",
        ticker="42800",
    )

    yield database
    database.close()
    os.unlink(path)


def test_batch_resolve_from_json(db):
    """Batch mode resolves predictions correctly via JSON file."""
    batch_data = {"42684": 1.0, "42779": 0.0}

    fd, batch_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(batch_path, "w") as f:
            json.dump(batch_data, f)

        predictions = fetch_unresolved(db)
        resolved = resolve_batch(db, predictions, batch_path)
    finally:
        os.unlink(batch_path)

    # Should have resolved exactly 2 predictions
    assert len(resolved) == 2

    # Verify the DB was updated
    resolved_preds = db.get_predictions(resolved_only=True)
    assert len(resolved_preds) == 2

    resolutions = {p["ticker"]: p["resolution"] for p in resolved_preds}
    assert resolutions["42684"] == 1.0
    assert resolutions["42779"] == 0.0

    # Brier scores should be computed
    for p in resolved_preds:
        assert p["swarm_brier"] is not None
        assert p["market_brier"] is not None

    # Third prediction should remain unresolved
    unresolved = fetch_unresolved(db)
    assert len(unresolved) == 1
    assert unresolved[0]["ticker"] == "42800"


def test_interactive_skip(db, monkeypatch):
    """Pressing Enter in interactive mode skips without resolving."""
    predictions = fetch_unresolved(db)

    # Simulate: skip, skip, skip (all Enter)
    inputs = iter(["", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    resolved = resolve_interactive(db, predictions)

    assert len(resolved) == 0

    # All predictions should remain unresolved
    unresolved = fetch_unresolved(db)
    assert len(unresolved) == 3


def test_summary_output(db, capsys):
    """Summary output includes resolved count, avg Brier, win rate, and best/worst."""
    # Resolve two predictions first
    predictions = fetch_unresolved(db)
    ticker_to_pred = {p["ticker"]: p for p in predictions}

    db.resolve(ticker_to_pred["42684"]["id"], 1.0)  # P=0.80, res=1.0 => brier=0.04
    db.resolve(ticker_to_pred["42779"]["id"], 0.0)  # P=0.35, res=0.0 => brier=0.1225

    resolved = [
        (ticker_to_pred["42684"]["id"], 1.0),
        (ticker_to_pred["42779"]["id"], 0.0),
    ]

    output = print_summary(db, resolved)

    assert "Resolved: 2" in output
    assert "Avg Brier:" in output
    assert "Win rate vs market:" in output
    assert "Best 5" in output
    assert "Worst 5" in output


def test_interactive_resolve_yes_and_no(db, monkeypatch):
    """Interactive mode correctly resolves YES (1.0) and NO (0.0) inputs."""
    predictions = fetch_unresolved(db)

    # Resolve first as YES, second as NO, skip third
    inputs = iter(["1.0", "0.0", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    resolved = resolve_interactive(db, predictions)

    assert len(resolved) == 2

    # First prediction resolved as 1.0, second as 0.0
    assert resolved[0][1] == 1.0
    assert resolved[1][1] == 0.0


def test_batch_missing_ticker(db, capsys):
    """Batch mode skips tickers not in the DB and warns."""
    batch_data = {"99999": 1.0, "42684": 1.0}

    fd, batch_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(batch_path, "w") as f:
            json.dump(batch_data, f)

        predictions = fetch_unresolved(db)
        resolved = resolve_batch(db, predictions, batch_path)
    finally:
        os.unlink(batch_path)

    # Only 1 resolved (99999 not in DB)
    assert len(resolved) == 1

    captured = capsys.readouterr()
    assert "99999" in captured.out
    assert "not found" in captured.out


def test_fetch_unresolved_excludes_non_metaculus(db):
    """fetch_unresolved only returns metaculus predictions, not other sources."""
    # Add a non-metaculus prediction
    db.log_prediction(
        question="Polymarket question",
        swarm_probability=0.5,
        market_price=0.5,
        source="polymarket",
        ticker="PM001",
    )

    unresolved = fetch_unresolved(db)
    sources = {p.get("source") for p in unresolved}
    # fetch_unresolved doesn't return source column, but all should be metaculus
    # Verify count is still 3 (the original metaculus ones)
    assert len(unresolved) == 3


def test_summary_no_resolutions(db, capsys):
    """Summary with zero resolutions prints appropriate message."""
    output = print_summary(db, [])

    assert "Resolved: 0" in output
    assert "No predictions were resolved." in output
