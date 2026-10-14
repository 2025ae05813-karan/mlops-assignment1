"""Model training, tuning, and selection (Assignment Tasks 2 & 4).

Pipeline:
  1. Load the cleaned dataset and split into train/test (stratified).
  2. For each candidate model (Logistic Regression, Random Forest):
       - run GridSearchCV with stratified k-fold cross-validation,
       - record cross-validated ROC-AUC and the best hyperparameters.
  3. Evaluate every tuned model on the held-out test set using accuracy,
     precision, recall, F1, and ROC-AUC.
  4. Select the best model by cross-validated ROC-AUC.
  5. Save the full (preprocessing + model) pipeline to models/model.joblib
     together with a metadata JSON for reproducibility.

MLflow logging (Task 3) is layered on top of this module via `mlflow_tracking`
being passed to `run_training`; when disabled, training still runs fully.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import config
from src.data_loader import load_processed
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.preprocess import build_model_pipeline, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """A candidate model: an estimator plus a hyperparameter search grid.

    Grid keys are prefixed with `model__` because the estimator sits under the
    'model' step of the Pipeline built in preprocess.build_model_pipeline.
    """
    name: str
    estimator: object
    param_grid: dict = field(default_factory=dict)


def candidate_models() -> list[ModelSpec]:
    """Define the models to train. At least two, per the rubric."""
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            param_grid={
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=config.RANDOM_STATE),
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    ]


@dataclass
class TrainedModel:
    name: str
    best_estimator: object
    best_params: dict
    cv_roc_auc: float
    test_metrics: dict


def _tune_one(spec: ModelSpec, X_train, y_train) -> GridSearchCV:
    pipeline = build_model_pipeline(spec.estimator)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def run_training(mlflow_tracking: bool = False) -> TrainedModel:
    """Train, tune, evaluate, select, and persist the best model."""
    df = load_processed()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    logger.info("Train/test split: %d / %d rows", len(X_train), len(X_test))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Optional MLflow setup (Task 3)
    mlflow = None
    if mlflow_tracking:
        import mlflow as _mlflow

        _mlflow.set_tracking_uri(config.MLFLOW_TRACKING_DIR.as_uri())
        _mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        mlflow = _mlflow

    trained: list[TrainedModel] = []

    for spec in candidate_models():
        logger.info("=== Tuning %s ===", spec.name)
        search = _tune_one(spec, X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        logger.info("%s | CV ROC-AUC=%.4f | best params=%s",
                    spec.name, search.best_score_, search.best_params_)
        logger.info("%s | test metrics=%s", spec.name, metrics)

        # Diagnostic plots per model
        cm_path = config.SCREENSHOTS_DIR / f"confusion_matrix_{spec.name}.png"
        roc_path = config.SCREENSHOTS_DIR / f"roc_curve_{spec.name}.png"
        plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix — {spec.name}", cm_path)
        plot_roc_curve(y_test, y_proba, f"ROC Curve — {spec.name}", roc_path)

        if mlflow is not None:
            with mlflow.start_run(run_name=spec.name):
                mlflow.log_params(search.best_params_)
                mlflow.log_param("cv_folds", config.CV_FOLDS)
                mlflow.log_metric("cv_roc_auc", float(search.best_score_))
                for k, v in metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                mlflow.log_artifact(str(cm_path))
                mlflow.log_artifact(str(roc_path))
                mlflow.sklearn.log_model(best, artifact_path="model")

        trained.append(TrainedModel(
            name=spec.name,
            best_estimator=best,
            best_params=search.best_params_,
            cv_roc_auc=float(search.best_score_),
            test_metrics=metrics,
        ))

    # Select best by cross-validated ROC-AUC
    winner = max(trained, key=lambda t: t.cv_roc_auc)
    logger.info("Selected best model: %s (CV ROC-AUC=%.4f)", winner.name, winner.cv_roc_auc)

    # Persist the winning pipeline + metadata (Task 4 — reproducibility)
    model_path = config.MODELS_DIR / "model.joblib"
    joblib.dump(winner.best_estimator, model_path)
    logger.info("Saved model pipeline -> %s", model_path)

    metadata = {
        "selected_model": winner.name,
        "best_params": winner.best_params,
        "cv_roc_auc": winner.cv_roc_auc,
        "test_metrics": winner.test_metrics,
        "feature_columns": config.FEATURE_COLUMNS,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "random_state": config.RANDOM_STATE,
        "all_models": {
            t.name: {"cv_roc_auc": t.cv_roc_auc, "test_metrics": t.test_metrics,
                     "best_params": t.best_params}
            for t in trained
        },
    }
    meta_path = config.MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Saved model metadata -> %s", meta_path)

    return winner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Heart Disease classifiers.")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    args = parser.parse_args()
    run_training(mlflow_tracking=args.mlflow)
