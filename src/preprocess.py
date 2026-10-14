"""Preprocessing pipeline (Assignment Tasks 2 & 4).

Builds a scikit-learn ColumnTransformer that:
  * numeric features  -> median imputation + standard scaling
  * categorical features -> most-frequent imputation + one-hot encoding

Fitting all transformers *inside* a Pipeline (rather than mutating the raw
DataFrame) is what guarantees reproducibility and prevents data leakage: the
imputation/scaling statistics are learned only from the training fold during
cross-validation and from the training split for the final model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the target vector y."""
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Construct the ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(estimator) -> Pipeline:
    """Wrap the preprocessor + an estimator into a single fit/predict Pipeline.

    The returned object takes a raw feature DataFrame (same columns as the
    cleaned CSV) and handles all transformation internally — this is exactly
    what gets saved and served, so training and serving share identical logic.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
