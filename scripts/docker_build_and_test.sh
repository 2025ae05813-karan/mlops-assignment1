#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the Docker image, run the container, and exercise the API with a
# sample request (Assignment Task 6 — "Container must build and run locally
# with sample input"). Capture the terminal output as a screenshot for the
# report.
#
# Prerequisite: Docker Desktop installed and running.
# Usage: bash scripts/docker_build_and_test.sh
# ---------------------------------------------------------------------------
set -euo pipefail

IMAGE="heart-disease-api:latest"
CONTAINER="heart-disease-api"
PORT=8000

echo "==> Building image: $IMAGE"
docker build -t "$IMAGE" .

echo "==> Removing any previous container"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "==> Starting container on port $PORT"
docker run -d --name "$CONTAINER" -p "${PORT}:8000" "$IMAGE"

echo "==> Waiting for the API to become healthy..."
for i in $(seq 1 30); do
  if curl -fs "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "    API is up."
    break
  fi
  sleep 2
done

echo ""
echo "==> GET /health"
curl -s "http://localhost:${PORT}/health" | python3 -m json.tool

echo ""
echo "==> POST /predict (low-risk sample)"
curl -s -X POST "http://localhost:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' \
  | python3 -m json.tool

echo ""
echo "==> POST /predict (high-risk sample)"
curl -s -X POST "http://localhost:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' \
  | python3 -m json.tool

echo ""
echo "==> Container logs (request log lines):"
docker logs "$CONTAINER" 2>&1 | grep -E "prediction=|GET /|POST /" | tail -10 || true

echo ""
echo "==> Done. Stop + remove with:  docker rm -f $CONTAINER"
