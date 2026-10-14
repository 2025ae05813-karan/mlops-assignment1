"""Central configuration: paths, column schema, and feature groupings.

Keeping all constants in one place makes the pipeline reproducible and keeps
the data loader, preprocessing, training, and API in agreement about the schema.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
SCREENSHOTS_DIR = REPORT_DIR / "screenshots"

# Primary raw source (Cleveland processed subset — the canonical 14-attribute file)
RAW_CLEVELAND = RAW_DIR / "processed.cleveland.data"
# Other sites available for an optional multi-site merge
RAW_HUNGARIAN = RAW_DIR / "processed.hungarian.data"
RAW_SWITZERLAND = RAW_DIR / "processed.switzerland.data"
RAW_VA = RAW_DIR / "processed.va.data"

PROCESSED_CSV = PROCESSED_DIR / "heart.csv"

# --------------------------------------------------------------------------- #
# Schema — the 14 attributes used by all published Heart Disease experiments
# --------------------------------------------------------------------------- #
# Raw files have no header; columns are positional. These names follow the
# standard UCI documentation (heart-disease.names).
COLUMNS = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 true, 0 false)
    "restecg",   # resting ECG results (0-2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise-induced angina (1 yes, 0 no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # slope of the peak exercise ST segment (1-3)
    "ca",        # number of major vessels colored by fluoroscopy (0-3)
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # raw target: 0 = no disease, 1-4 = disease severity
]

TARGET = "target"            # final binary target column name
RAW_TARGET = "num"           # original multi-class target column name

# Missing-value marker used throughout the raw UCI files
NA_MARKER = "?"

# Feature typing for the preprocessing pipeline
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Ordered list of model input features (everything except the target)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# MLflow
MLFLOW_EXPERIMENT = "heart-disease-classification"
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
