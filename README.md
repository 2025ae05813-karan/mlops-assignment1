# Heart Disease Risk Prediction — End-to-End MLOps Pipeline

A production-style machine learning solution that predicts the risk of heart
disease from patient health data and serves it as a cloud-ready, monitored API.
Built for the MLOps (AIMLCZG523) Assignment 01.

[![CI/CD Pipeline](https://github.com/<your-user>/<your-repo>/actions/workflows/ci.yml/badge.svg)](../../actions)

---

## What this project demonstrates

| Capability | Implementation |
|------------|----------------|
| Data acquisition & cleaning | Reproducible download + parser for the raw UCI files |
| EDA | Class balance, histograms, categorical breakdowns, correlation heatmap |
| Modeling | Logistic Regression + Random Forest, tuned with GridSearchCV + 5-fold CV |
| Experiment tracking | MLflow (params, metrics, artifacts, model) |
| Reproducibility | Single sklearn Pipeline, pinned deps, fixed seeds, clean-env verified |
| Serving | FastAPI `/predict` (prediction + confidence), `/health`, `/metrics` |
| Containerization | Self-contained Dockerfile (trains at build, non-root, healthcheck) |
| CI/CD | GitHub Actions: lint → test → train, with artifact uploads |
| Deployment | Kubernetes manifests (Deployment + LoadBalancer + Ingress) |
| Monitoring | Prometheus + Grafana stack via docker-compose |

**Best model:** Logistic Regression — Test ROC-AUC **0.966**, Recall **0.93**,
Accuracy **0.885** (held-out set).

---

## Repository layout

```
heart-disease-mlops/
├── data/{raw,processed}/     # raw UCI files + cleaned heart.csv
├── src/                      # config, data_loader, preprocess, train, evaluate, predict
├── notebooks/01_eda.ipynb    # exploratory analysis
├── api/                      # FastAPI app + Pydantic schemas
├── tests/                    # pytest: data, model, API (26 tests)
├── deploy/k8s/               # Kubernetes manifests + guide
├── deploy/monitoring/        # Prometheus + Grafana configs + dashboard
├── scripts/                  # download_data, docker + k8s helper scripts
├── .github/workflows/ci.yml  # CI/CD pipeline
├── Dockerfile, docker-compose.yml
├── requirements.txt, environment.yml
└── report/                   # final report, screenshots, tracking summary
```

---

## Quickstart

### 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build data, run EDA, train

```bash
python scripts/download_data.py          # acquire raw data (reuses local zip)
python -m src.data_loader --sites cleveland
python -m src.eda
python -m src.train --mlflow             # train + log to MLflow
python -m src.mlflow_summary             # export run summary
```

### 3. Serve the API locally (no Docker)

```bash
uvicorn api.main:app --reload
# open http://localhost:8000/docs
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

### 4. Tests + lint

```bash
ruff check src/ api/ tests/ scripts/
pytest
```

### 5. Docker

```bash
docker build -t heart-disease-api:latest .
bash scripts/docker_build_and_test.sh
```

### 6. Kubernetes (local)

```bash
# see deploy/k8s/README.md for Minikube / Docker Desktop details
bash scripts/k8s_deploy_and_verify.sh
```

### 7. Monitoring stack

```bash
docker compose up --build
# API 8000 | Prometheus 9090 | Grafana 3000 (admin/admin)
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service metadata |
| GET | `/health` | Liveness/readiness (model loaded?) |
| POST | `/predict` | Patient features → prediction + confidence |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Interactive OpenAPI docs |

**`/predict` response:**
```json
{"prediction": 1, "label": "disease", "confidence": 0.986, "probability_disease": 0.986}
```

---

## Documentation

- Full report: [`report/REPORT.md`](report/REPORT.md)
- Architecture: [`report/architecture.md`](report/architecture.md)
- Reproducibility guide: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Deployment guide: [`deploy/k8s/README.md`](deploy/k8s/README.md)
- Monitoring guide: [`deploy/monitoring/README.md`](deploy/monitoring/README.md)

## Dataset

Heart Disease UCI dataset (Cleveland subset, 303 patients, 13 features + binary
target). Source: UCI Machine Learning Repository, ID 45.

## License / Academic note

Prepared independently for AIMLCZG523 Assignment 01.
