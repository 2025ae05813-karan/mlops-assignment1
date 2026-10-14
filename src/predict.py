"""Inference module (Assignment Task 4 — reproducible loading & prediction).

Loads the persisted (preprocessing + model) pipeline and exposes a simple
`predict_one` / `predict_batch` interface used by both the CLI and the serving
API (Task 6). Because the saved object is a full sklearn Pipeline, inference
applies the identical preprocessing that was fit during training.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src import config

MODEL_PATH = config.MODELS_DIR / "model.joblib"
METADATA_PATH = config.MODELS_DIR / "model_metadata.json"


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    """Load and cache the trained pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Train it first with `python -m src.train`."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_metadata(path: Path = METADATA_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _to_frame(records: list[dict]) -> pd.DataFrame:
    """Build a feature DataFrame with columns in the exact training order."""
    df = pd.DataFrame(records)
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    return df[config.FEATURE_COLUMNS]


def predict_batch(records: list[dict]) -> list[dict]:
    """Predict for a list of feature dicts.

    Returns one dict per record with the predicted label, the human-readable
    label, and the model's confidence (probability of the predicted class).
    """
    model = load_model()
    X = _to_frame(records)
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)

    results = []
    for p, pr in zip(preds, proba):
        results.append({
            "prediction": int(p),
            "label": "disease" if int(p) == 1 else "no_disease",
            "confidence": float(pr[int(p)]),
            "probability_disease": float(pr[1]),
        })
    return results


def predict_one(record: dict) -> dict:
    """Convenience wrapper for a single record."""
    return predict_batch([record])[0]


if __name__ == "__main__":
    # Smoke test with a representative patient record.
    sample = {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }
    print(json.dumps(predict_one(sample), indent=2))
