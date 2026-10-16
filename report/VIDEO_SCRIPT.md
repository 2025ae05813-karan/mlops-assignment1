# Video Recording Script — Heart Disease MLOps Pipeline

**Target length:** 6–9 minutes. Record your screen (QuickTime on macOS:
File → New Screen Recording) with mic narration. Follow the sections in order;
the commands are copy-paste ready. Narration cues are in _italics_.

> Tip: open three terminal tabs in `heart-disease-mlops/` beforehand, with the
> venv activated (`source .venv/bin/activate`). Pre-build the Docker image so
> you don't record a long build.

---

## 0. Intro (20s)
_"This is my end-to-end MLOps pipeline for heart disease risk prediction. It
covers data processing, model training with experiment tracking, automated
testing and CI/CD, containerized serving, Kubernetes deployment, and
monitoring. Let me walk through it."_

Show the repo structure:
```bash
ls
```

---

## 1. Data & EDA (45s)
_"First, data acquisition and cleaning. The raw UCI files use question marks for
missing values and a 0–4 target; my loader cleans and binarizes them."_
```bash
python -m src.data_loader --sites cleveland
head -3 data/processed/heart.csv
```
_"EDA produces class balance, histograms, and a correlation heatmap."_
Open `report/screenshots/` and show `class_balance.png` and
`correlation_heatmap.png`.

---

## 2. Training + Experiment Tracking (90s)
_"I train Logistic Regression and Random Forest, each tuned with grid search and
5-fold cross-validation, logging everything to MLflow."_
```bash
python -m src.train --mlflow
```
_"Logistic Regression won with a cross-validated ROC-AUC of 0.90 and recall of
0.93. Here are the runs in MLflow."_
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open `http://127.0.0.1:5000`, show the two runs, click one to show params,
metrics, and artifacts (confusion matrix, ROC curve, model). Stop with Ctrl+C.

---

## 3. Tests + CI/CD (60s)
_"Quality is enforced by linting and 26 unit tests across data, model, and API
code."_
```bash
ruff check src/ api/ tests/ scripts/
pytest -q
```
_"On every push, GitHub Actions runs lint, then tests, then training, uploading
artifacts. The pipeline fails on any error."_
Open the GitHub repo → **Actions** tab → show a green run and its uploaded
artifacts. (Scroll the `ci.yml` file briefly.)

---

## 4. API + Docker (75s)
_"The model is served by FastAPI with a /predict endpoint returning a prediction
and confidence, plus /health and /metrics. It runs in a self-contained Docker
container."_
```bash
docker build -t heart-disease-api:latest .   # pre-built; show quickly
bash scripts/docker_build_and_test.sh
```
Point out the `/health` response and the two `/predict` responses (low-risk →
no_disease, high-risk → disease with ~0.98 confidence). Open
`http://localhost:8000/docs` to show the interactive API.

---

## 5. Kubernetes Deployment (75s)
_"The container deploys to local Kubernetes with 2 replicas behind a
LoadBalancer."_
```bash
bash scripts/k8s_deploy_and_verify.sh
```
_"Both pods are Running, the service is exposed, and the endpoints respond."_
Show `kubectl get deployment,pods,svc -l app=heart-disease-api` and the
`curl /predict` output.

---

## 6. Monitoring (75s)
_"Finally, monitoring. Docker Compose brings up the API with Prometheus and
Grafana."_
```bash
docker compose up -d
# generate traffic
for i in $(seq 1 30); do curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' >/dev/null; done
```
Open Prometheus `http://localhost:9090` → Status → Targets (API **UP**).
Open Grafana `http://localhost:3000` (admin/admin) → the auto-loaded dashboard
showing request rate, latency, and status codes updating live.
_"And request logs are visible here:"_
```bash
docker compose logs --tail=10 api
```

---

## 7. Wrap-up (20s)
_"That's the full pipeline — from raw data to a monitored, containerized,
Kubernetes-deployed API, with tracking, testing, and CI/CD throughout. All code
is in the linked GitHub repository. Thank you."_
```bash
docker compose down
```

---

### Pre-flight checklist
- [ ] venv activated; `pip install -r requirements.txt` done
- [ ] `python -m src.train` run at least once (model exists)
- [ ] Docker image pre-built (`docker build -t heart-disease-api:latest .`)
- [ ] Local Kubernetes running (Docker Desktop K8s enabled, or `minikube start`)
- [ ] GitHub repo pushed with a green Actions run to show
- [ ] Screen recording + mic tested
