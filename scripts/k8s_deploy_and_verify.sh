#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the API to local Kubernetes and verify the endpoints (Task 7).
# Works with Docker Desktop Kubernetes (LoadBalancer on localhost) or Minikube.
#
# Prerequisites:
#   - docker build -t heart-disease-api:latest .
#   - a running local cluster (Docker Desktop K8s or `minikube start`)
#   - kubectl on PATH
#
# Usage: bash scripts/k8s_deploy_and_verify.sh
# ---------------------------------------------------------------------------
set -euo pipefail

NS_LABEL="app=heart-disease-api"

echo "==> Applying manifests"
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

echo "==> Waiting for the deployment to become available"
kubectl rollout status deployment/heart-disease-api --timeout=180s

echo ""
echo "==> Resources"
kubectl get deployment,pods,svc -l "$NS_LABEL"

echo ""
echo "==> Resolving service URL"
URL=""
if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
  URL="$(minikube service heart-disease-api --url | head -1)"
else
  # Docker Desktop LoadBalancer -> localhost:80
  URL="http://localhost"
fi
echo "    Service URL: $URL"

echo ""
echo "==> GET /health"
curl -fs "${URL}/health" | python3 -m json.tool || echo "health check failed"

echo ""
echo "==> POST /predict (high-risk sample)"
curl -fs -X POST "${URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}' \
  | python3 -m json.tool || echo "predict call failed"

echo ""
echo "==> Done. Capture the output above as your deployment screenshot."
echo "    Teardown: kubectl delete -f deploy/k8s/service.yaml -f deploy/k8s/deployment.yaml"
