# Experiment Tracking Summary (MLflow)

**Experiment:** `heart-disease-classification`  
**Tracking URI:** `file:///Users/auriga/Desktop/new-workspace/task/MLOPS%20ASSIGNMENT%201/heart-disease-mlops/mlruns`  
**Total runs:** 2

Launch the UI with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Run: `logistic_regression`

- **Run ID:** `e1031f5e841c4d879ea404fc6f7f63b1`
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

- **Run ID:** `6a9fe2010ee247e5b16a2f24f1db874f`
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
| cv_roc_auc | 0.8996 |
| test_accuracy | 0.8525 |
| test_f1 | 0.8475 |
| test_precision | 0.8065 |
| test_recall | 0.8929 |
| test_roc_auc | 0.9545 |

**Logged artifacts:** `confusion_matrix_random_forest.png`, `model`, `roc_curve_random_forest.png`
