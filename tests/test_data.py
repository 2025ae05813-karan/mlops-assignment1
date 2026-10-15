"""Unit tests for data loading, cleaning, and preprocessing (Assignment Task 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import binarize_target, clean_data, load_raw_file
from src.preprocess import build_model_pipeline, build_preprocessor, split_features_target


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    """A tiny frame mimicking the raw schema (14 cols incl. 'num' target)."""
    data = {
        "age": [63.0, 67.0, 41.0, 63.0],       # last row duplicates row 0
        "sex": [1.0, 1.0, 0.0, 1.0],
        "cp": [1.0, 4.0, 2.0, 1.0],
        "trestbps": [145.0, 160.0, 130.0, 145.0],
        "chol": [233.0, 286.0, 204.0, 233.0],
        "fbs": [1.0, 0.0, 0.0, 1.0],
        "restecg": [2.0, 2.0, 2.0, 2.0],
        "thalach": [150.0, 108.0, 172.0, 150.0],
        "exang": [0.0, 1.0, 0.0, 0.0],
        "oldpeak": [2.3, 1.5, 1.4, 2.3],
        "slope": [3.0, 2.0, 1.0, 3.0],
        "ca": [0.0, 3.0, np.nan, 0.0],          # a missing value
        "thal": [6.0, 3.0, 3.0, 6.0],
        "num": [0, 2, 0, 0],                     # 0 and 2 (severity)
    }
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# load_raw_file
# --------------------------------------------------------------------------- #
def test_load_raw_file_shape_and_columns():
    """The real Cleveland file loads with the expected 14 columns."""
    df = load_raw_file(config.RAW_CLEVELAND)
    assert list(df.columns) == config.COLUMNS
    assert len(df) == 303


def test_load_raw_file_question_marks_become_nan():
    df = load_raw_file(config.RAW_CLEVELAND)
    # ca and thal are known to contain the '?' marker in the Cleveland file.
    assert df[["ca", "thal"]].isna().sum().sum() > 0


def test_load_raw_file_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_raw_file(config.RAW_DIR / "does_not_exist.data")


# --------------------------------------------------------------------------- #
# binarize_target
# --------------------------------------------------------------------------- #
def test_binarize_target_maps_1to4_to_one(raw_like_df):
    out = binarize_target(raw_like_df)
    assert config.TARGET in out.columns
    assert config.RAW_TARGET not in out.columns
    # num=[0,2,0,0] -> target=[0,1,0,0]
    assert out[config.TARGET].tolist() == [0, 1, 0, 0]


def test_binarize_target_is_binary(raw_like_df):
    out = binarize_target(raw_like_df)
    assert set(out[config.TARGET].unique()).issubset({0, 1})


# --------------------------------------------------------------------------- #
# clean_data
# --------------------------------------------------------------------------- #
def test_clean_data_drops_duplicates(raw_like_df):
    out = clean_data(raw_like_df)
    # one duplicate row removed -> 3 unique rows
    assert len(out) == 3


def test_clean_data_produces_binary_target(raw_like_df):
    out = clean_data(raw_like_df)
    assert set(out[config.TARGET].unique()).issubset({0, 1})


# --------------------------------------------------------------------------- #
# split_features_target
# --------------------------------------------------------------------------- #
def test_split_features_target_separates_columns(raw_like_df):
    cleaned = clean_data(raw_like_df)
    X, y = split_features_target(cleaned)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert config.TARGET not in X.columns
    assert len(X) == len(y)


# --------------------------------------------------------------------------- #
# preprocessing pipeline
# --------------------------------------------------------------------------- #
def test_preprocessor_imputes_missing_values(raw_like_df):
    """After transform there must be no NaNs (imputation handled them)."""
    cleaned = clean_data(raw_like_df)
    X, _ = split_features_target(cleaned)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert not np.isnan(transformed).any()


def test_preprocessor_scales_numeric_features(raw_like_df):
    """Numeric columns should be standardized (roughly zero mean after fit)."""
    cleaned = clean_data(raw_like_df)
    X, _ = split_features_target(cleaned)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    n_numeric = len(config.NUMERIC_FEATURES)
    numeric_block = transformed[:, :n_numeric]
    # StandardScaler -> each numeric column mean ~ 0
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-6)


def test_model_pipeline_fits_and_predicts(raw_like_df):
    """A full pipeline must fit on the toy data and emit valid predictions."""
    from sklearn.linear_model import LogisticRegression

    cleaned = clean_data(raw_like_df)
    X, y = split_features_target(cleaned)
    pipe = build_model_pipeline(LogisticRegression(max_iter=200))
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(X)
