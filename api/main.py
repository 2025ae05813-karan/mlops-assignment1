"""FastAPI serving application (Assignment Tasks 6 & 8).

Endpoints:
  GET  /          - service metadata
  GET  /health    - liveness/readiness probe (used by K8s in Task 7)
  POST /predict   - JSON patient features -> prediction + confidence
  GET  /metrics   - Prometheus metrics (added by the instrumentator, Task 8)

Request logging (Task 8) is handled by middleware that records method, path,
status, and latency for every call.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src import predict as predict_mod

# --------------------------------------------------------------------------- #
# Logging (Task 8 — log API requests)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model cache on startup so the first prediction isn't slow, and
    fail fast with a clear log if the model artifact is missing."""
    try:
        predict_mod.load_model()
        meta = predict_mod.load_metadata()
        logger.info("Model loaded: %s", meta.get("selected_model", "unknown"))
    except FileNotFoundError as exc:
        logger.error("Model not available at startup: %s", exc)
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts the risk of heart disease from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status, and latency for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# Prometheus metrics at /metrics (Task 8)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
def root() -> dict:
    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the model is loaded and ready to serve."""
    try:
        predict_mod.load_model()
        meta = predict_mod.load_metadata()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=meta.get("selected_model"),
        )
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    """Predict heart disease risk for a single patient."""
    result = predict_mod.predict_one(features.model_dump())
    logger.info(
        "prediction=%s confidence=%.3f", result["label"], result["confidence"]
    )
    return PredictionResponse(**result)
