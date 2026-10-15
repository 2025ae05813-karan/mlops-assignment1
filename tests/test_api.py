"""API endpoint tests (supports Tasks 5 & 6).

Uses FastAPI's TestClient so no live server is required. Assumes a trained
model exists (models/model.joblib); the CI pipeline trains before testing.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

LOW_RISK = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}
HIGH_RISK = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
    "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
    "ca": 3, "thal": 3,
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_prediction_and_confidence():
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease", "no_disease")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


def test_predict_confidence_matches_predicted_class():
    r = client.post("/predict", json=HIGH_RISK)
    body = r.json()
    # confidence is the probability of the predicted class, so it must be >= 0.5
    assert body["confidence"] >= 0.5


@pytest.mark.parametrize("bad", [
    {"age": "not-a-number"},                 # wrong type
    {**LOW_RISK, "sex": 5},                   # out of range
    {k: v for k, v in LOW_RISK.items() if k != "age"},  # missing field
])
def test_predict_rejects_invalid_input(bad):
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_request" in r.text
