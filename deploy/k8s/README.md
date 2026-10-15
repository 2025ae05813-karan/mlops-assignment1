# Kubernetes Deployment Guide (Task 7)

Deploy the Dockerized Heart Disease API to **local Kubernetes** (Minikube or
Docker Desktop) and expose it via a LoadBalancer (or Ingress). Capture the
commands + outputs below as deployment screenshots for the report.

---

## Prerequisites

- Docker image built locally:
  ```bash
  docker build -t heart-disease-api:latest .
  ```
- A running local Kubernetes cluster (choose one):
  - **Docker Desktop:** Settings → Kubernetes → *Enable Kubernetes*.
  - **Minikube:** `minikube start`

---

## Option A — Minikube

Minikube runs its own Docker daemon, so either build *inside* it or load the image:

```bash
# Make the image available to Minikube
minikube image load heart-disease-api:latest
# (alternatively: eval $(minikube docker-env) && docker build -t heart-disease-api:latest .)

# Deploy
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

# Watch rollout
kubectl get pods -w            # wait until both pods are Running & READY 1/1

# Expose the LoadBalancer service (keep this running in a second terminal)
minikube tunnel
# ...then find the external IP:
kubectl get svc heart-disease-api

# OR, simplest one-liner that opens the service URL:
minikube service heart-disease-api --url
```

---

## Option B — Docker Desktop Kubernetes

Docker Desktop shares your local Docker images, so no image load is needed.

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

kubectl get pods            # wait for 2/2 Running
kubectl get svc heart-disease-api
# LoadBalancer is reachable at http://localhost  (port 80 -> 8000)
```

---

## Verify the deployment

```bash
# Inspect resources (good screenshot)
kubectl get deployment,pods,svc -l app=heart-disease-api

# Health check (replace <URL> with the service URL / localhost)
curl http://<URL>/health

# Prediction
curl -X POST http://<URL>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

Expected `/predict` response:
```json
{"prediction":1,"label":"disease","confidence":0.98,"probability_disease":0.98}
```

---

## Option C — Ingress (alternative to LoadBalancer)

```bash
# Enable an ingress controller
minikube addons enable ingress            # Minikube
# (Docker Desktop: kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml)

kubectl apply -f deploy/k8s/ingress.yaml

# Map the host locally
echo "127.0.0.1 heart-disease.local" | sudo tee -a /etc/hosts
curl http://heart-disease.local/health
```

---

## Screenshots to capture for the report

1. `kubectl get deployment,pods,svc -l app=heart-disease-api` showing pods **Running** and the LoadBalancer service.
2. `curl .../health` returning `{"status":"ok",...}`.
3. `curl .../predict` returning a prediction + confidence.
4. (Optional) `kubectl describe deployment heart-disease-api` showing the replica/probe config.

## Teardown

```bash
kubectl delete -f deploy/k8s/service.yaml -f deploy/k8s/deployment.yaml
# kubectl delete -f deploy/k8s/ingress.yaml   # if applied
```
