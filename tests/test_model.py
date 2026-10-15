"""Unit tests for training, evaluation, and inference (Assignment Task 5)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src import config
from src.evaluate import compute_metrics
from src.train import candidate_models


# --------------------------------------------------------------------------- #
# candidate_models
# --------------------------------------------------------------------------- #
def test_at_least_two_candidate_models():
    """The rubric requires training at least two classification models."""
    specs = candidate_models()
    assert len(specs) >= 2
    names = {s.name for s in specs}
    assert "logistic_regression" in names
    assert "random_forest" in names


def test_candidate_models_have_param_grids():
    for spec in candidate_models():
        assert spec.param_grid, f"{spec.name} should define a tuning grid"
        # grid keys must target the 'model' step of the pipeline
        assert all(k.startswith("model__") for k in spec.param_grid)


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["roc_auc"] == 1.0


def test_compute_metrics_keys_and_ranges():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.2, 0.7, 0.4, 0.3, 0.9])
    m = compute_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in m
        assert 0.0 <= m[key] <= 1.0


# --------------------------------------------------------------------------- #
# end-to-end estimator behavior (fast, synthetic data)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", candidate_models(), ids=lambda s: s.name)
def test_each_estimator_learns_signal(spec):
    """Each candidate estimator should beat chance on a separable problem."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        random_state=config.RANDOM_STATE,
    )
    est = spec.estimator
    est.fit(X, y)
    acc = est.score(X, y)
    assert acc > 0.7  # clearly better than random on an easy synthetic set


def test_model_predict_proba_available():
    """Confidence reporting in the API depends on predict_proba existing."""
    for spec in candidate_models():
        assert hasattr(spec.estimator, "predict_proba")
