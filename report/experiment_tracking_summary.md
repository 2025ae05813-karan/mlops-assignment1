# Experiment Tracking Summary (MLflow)

**Experiment:** `heart-disease-classification`  
**Tracking URI:** `file:///home/s2025ae05813/workspace/mlops-assignment1/mlruns`  
**Total runs:** 4

Launch the UI with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Run: `logistic_regression`

- **Run ID:** `3f7f4c54ad1c45ea8ad64c043305d466`
- **Status:** FINISHED

**Parameters**

| param | value |
|-------|-------|
| cv_folds | 5 |
| model__C | 1.0 |
| model__penalty | l2 |
| model__solver | lbfgs |

**Metrics**

| metric | value |
|--------|-------|
| cv_roc_auc | 0.9025 |
| test_accuracy | 0.8852 |
| test_f1 | 0.8814 |
| test_precision | 0.8387 |
| test_recall | 0.9286 |
| test_roc_auc | 0.9665 |

**Logged artifacts:** `confusion_matrix_logistic_regression.png`, `model`, `roc_curve_logistic_regression.png`

## Run: `random_forest`

- **Run ID:** `01b3de1b8dd0436dbcbcaa9c2cee0b01`
- **Status:** FINISHED

**Parameters**

| param | value |
|-------|-------|
| cv_folds | 5 |
| model__max_depth | 5 |
| model__min_samples_leaf | 2 |
| model__n_estimators | 400 |

**Metrics**

| metric | value |
|--------|-------|
| cv_roc_auc | 0.8993 |
| test_accuracy | 0.8525 |
| test_f1 | 0.8475 |
| test_precision | 0.8065 |
| test_recall | 0.8929 |
| test_roc_auc | 0.9545 |

**Logged artifacts:** `confusion_matrix_random_forest.png`, `model`, `roc_curve_random_forest.png`

## Run: `logistic_regression`

- **Run ID:** `03de5a3d8ceb4542ae54d1bb7a87bbff`
- **Status:** FINISHED

**Parameters**

| param | value |
|-------|-------|
| cv_folds | 5 |
| model__C | 1.0 |
| model__penalty | l2 |
| model__solver | lbfgs |

**Metrics**

| metric | value |
|--------|-------|
| cv_roc_auc | 0.9025 |
| test_accuracy | 0.8852 |
| test_f1 | 0.8814 |
| test_precision | 0.8387 |
| test_recall | 0.9286 |
| test_roc_auc | 0.9665 |

**Logged artifacts:** `confusion_matrix_logistic_regression.png`, `model`, `roc_curve_logistic_regression.png`

## Run: `random_forest`

- **Run ID:** `f75799c51f784746b9d78c6da00d8ca6`
- **Status:** FINISHED

**Parameters**

| param | value |
|-------|-------|
| cv_folds | 5 |
| model__max_depth | 5 |
| model__min_samples_leaf | 2 |
| model__n_estimators | 400 |

**Metrics**

| metric | value |
|--------|-------|
| cv_roc_auc | 0.8993 |
| test_accuracy | 0.8525 |
| test_f1 | 0.8475 |
| test_precision | 0.8065 |
| test_recall | 0.8929 |
| test_roc_auc | 0.9545 |

**Logged artifacts:** `confusion_matrix_random_forest.png`, `model`, `roc_curve_random_forest.png`
