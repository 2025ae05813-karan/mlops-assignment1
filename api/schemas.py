"""Pydantic request/response schemas for the serving API (Assignment Task 6).

Validation here means malformed requests are rejected with a clear 422 error
before they ever reach the model — a production-readiness requirement.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """A single patient's measurements — the 13 model input features.

    Field constraints encode the valid ranges / categories documented in the
    UCI dataset so obviously invalid input is rejected early.
    """

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1/0)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1/0)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression vs rest")
    slope: int = Field(..., ge=0, le=3, description="Slope of peak exercise ST (1-3)")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored (0-3)")
    thal: int = Field(..., ge=0, le=7, description="3=normal, 6=fixed, 7=reversible defect")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    """The prediction returned to the client."""

    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float = Field(..., description="Probability of disease (class 1)")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
