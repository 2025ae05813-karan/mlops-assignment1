# Monitoring & Logging (Task 8)

The API ships with two observability features:

1. **Request logging** — every HTTP request is logged with method, path, status
   code, and latency by the middleware in `api/main.py`
   (`log_requests`). Prediction results (label + confidence) are logged too.
   View them with `docker compose logs -f api` or `kubectl logs <pod>`.

2. **Metrics + dashboards** — the API exposes Prometheus metrics at `/metrics`
   (via `prometheus-fastapi-instrumentator`); Prometheus scrapes them and
   Grafana visualizes them.

---

## Run the full stack locally

```bash
docker compose up --build
```

| Service | URL | Notes |
|---------|-----|-------|
| API | http://localhost:8000 | `/docs`, `/predict`, `/metrics` |
| Prometheus | http://localhost:9090 | Status → Targets should show `heart-disease-api` UP |
| Grafana | http://localhost:3000 | login `admin` / `admin`; dashboard auto-loaded |

## Generate traffic (so the dashboard has data)

```bash
# a few predictions
for i in $(seq 1 30); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' >/dev/null
done
```

## Dashboard panels

The provisioned dashboard **"Heart Disease API — Monitoring"** shows:
- Request rate (req/s) by handler
- Responses by status code
- Request latency p50 / p95
- Total `/predict` requests
- Average request duration

## Screenshots to capture for the report

1. Prometheus → **Status → Targets** showing the API target **UP**.
2. Grafana dashboard with live panels after generating traffic.
3. (Optional) `docker compose logs api` showing the per-request log lines.

## Teardown

```bash
docker compose down
```
