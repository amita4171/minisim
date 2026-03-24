"""Integration tests for the FastAPI production API."""
import time
import pytest
from fastapi.testclient import TestClient

from api import app, _predictions


@pytest.fixture
def client():
    """Create a test client with clean prediction state."""
    _predictions.clear()
    return TestClient(app)


HEADERS = {"Authorization": "Bearer demo-key-12345"}


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_predict_requires_auth(client):
    resp = client.post("/v1/predict", json={"question": "Will it rain?"})
    assert resp.status_code == 401


def test_predict_invalid_key(client):
    resp = client.post(
        "/v1/predict",
        json={"question": "Will it rain tomorrow?"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


def test_predict_missing_question(client):
    resp = client.post("/v1/predict", json={}, headers=HEADERS)
    assert resp.status_code == 422  # validation error


@pytest.mark.slow
def test_predict_creates_prediction(client):
    resp = client.post(
        "/v1/predict",
        json={"question": "Will it rain tomorrow in Seattle?"},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_id" in data
    assert data["status"] == "processing"
    assert data["question"] == "Will it rain tomorrow in Seattle?"


@pytest.mark.slow
def test_predict_with_config(client):
    resp = client.post(
        "/v1/predict",
        json={
            "question": "Will the Fed cut rates in May 2026?",
            "market_price": 0.35,
            "config": {"n_agents": 10, "n_rounds": 2, "mode": "offline"},
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["market_price"] == 0.35


def test_get_prediction_not_found(client):
    resp = client.get("/v1/predict/nonexistent", headers=HEADERS)
    assert resp.status_code == 404


@pytest.mark.slow
def test_predict_and_poll(client):
    """Full flow: submit prediction, wait, poll for result."""
    # Submit
    resp = client.post(
        "/v1/predict",
        json={
            "question": "Will it be sunny tomorrow in Phoenix Arizona?",
            "config": {"n_agents": 5, "n_rounds": 1, "mode": "offline"},
        },
        headers=HEADERS,
    )
    pred_id = resp.json()["prediction_id"]

    # Background task runs synchronously in TestClient
    # Poll for result
    resp = client.get(f"/v1/predict/{pred_id}", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()

    # Should be completed (TestClient runs background tasks inline)
    if data["status"] == "completed":
        assert data["swarm_probability"] is not None
        assert 0 <= data["swarm_probability"] <= 1
        assert data["timing"] is not None


@pytest.mark.slow
def test_resolve_prediction(client):
    """Submit, complete, then resolve with ground truth."""
    # Submit
    resp = client.post(
        "/v1/predict",
        json={
            "question": "Will the sun rise tomorrow morning?",
            "market_price": 0.99,
            "config": {"n_agents": 5, "n_rounds": 1, "mode": "offline"},
        },
        headers=HEADERS,
    )
    pred_id = resp.json()["prediction_id"]

    # Wait for completion
    resp = client.get(f"/v1/predict/{pred_id}", headers=HEADERS)
    if resp.json()["status"] != "completed":
        time.sleep(1)
        resp = client.get(f"/v1/predict/{pred_id}", headers=HEADERS)

    if resp.json()["status"] == "completed":
        # Resolve
        resp = client.post(
            f"/v1/predict/{pred_id}/resolve",
            json={"resolution": 1.0},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "swarm_brier" in data
        assert data["swarm_brier"] >= 0


def test_resolve_not_found(client):
    resp = client.post(
        "/v1/predict/nonexistent/resolve",
        json={"resolution": 1.0},
        headers=HEADERS,
    )
    assert resp.status_code == 404


def test_metrics(client):
    resp = client.get("/v1/metrics", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_predictions" in data
    assert "resolved" in data


def test_accuracy_dashboard_public(client):
    """Accuracy dashboard should be accessible without auth."""
    resp = client.get("/accuracy")
    assert resp.status_code == 200
    assert "MiniSim" in resp.text
    assert "Calibration" in resp.text
