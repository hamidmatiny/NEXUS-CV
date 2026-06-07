# Deployment Guide

NEXUS-CV supports three deployment targets: **Docker Compose** (dev/demo), **Kubernetes (Helm)**, and **cloud managed** (GCP Cloud Run / AWS ECS).

---

## Docker Compose (recommended for demo)

```bash
docker compose up --build -d
```

Services and ports:

| Service | Port | Purpose |
|---------|------|---------|
| serving | 8000 | FastAPI + WebSocket + gRPC |
| ingestion | 8001, 8265 | Ray ingestion + dashboard |
| mlflow | 5001 | Experiment tracking |
| prometheus | 9090 | Metrics scrape |
| grafana | 3000 | Dashboards |

Production image:

```bash
docker build --target production -f docker/Dockerfile.serving -t nexus-cv:latest .
```

---

## Kubernetes (Helm)

```bash
helm install nexus-cv ./infra/helm/nexus-cv \
  --set image.repository=gcr.io/PROJECT/nexus-cv \
  --set image.tag=SHA \
  --set autoscaling.enabled=true
```

Key values (`infra/helm/nexus-cv/values.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replicaCount` | 2 | Static replicas when HPA disabled |
| `autoscaling.minReplicas` | 2 | HPA floor |
| `autoscaling.maxReplicas` | 10 | HPA ceiling |
| `autoscaling.targetCPUUtilizationPercentage` | 70 | CPU scale trigger |
| `autoscaling.customMetric.name` | `nexus_cv_inference_queue_depth` | Custom Prometheus metric |
| `service.httpPort` | 8000 | REST + WebSocket |
| `service.grpcPort` | 50051 | gRPC inference |

Expose via Ingress or LoadBalancer as needed.

---

## GCP (Terraform)

```bash
cd infra/terraform/gcp
terraform init
terraform apply -var="project_id=YOUR_PROJECT" -var="region=us-central1" -var="image_tag=latest"
```

Resources created:
- `google_cloud_run_service.nexus_cv_serving` — min 1, max 10 instances, 2 CPU, 4 GiB
- `google_storage_bucket.nexus_cv_models` — model artifacts
- `google_storage_bucket.nexus_cv_mlflow` — MLflow artifacts
- `google_artifact_registry_repository` — container images
- Service account with least-privilege IAM (documented in `main.tf`)

CI deploy workflow (`.github/workflows/deploy.yml`) pushes to GCR and updates Cloud Run on main merge.

Required secrets: `GCP_SA_KEY`, `GCP_PROJECT_ID`.

---

## AWS (Terraform)

```bash
cd infra/terraform/aws
terraform init
terraform apply \
  -var="aws_region=us-east-1" \
  -var="ecr_image_uri=ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/nexus-cv:latest" \
  -var="vpc_id=vpc-xxx" \
  -var='subnet_ids=["subnet-a","subnet-b"]'
```

Resources:
- ECS Fargate cluster + service (2048 CPU, 4096 MiB)
- ALB + target group (`/health` checks)
- S3 buckets for models and MLflow
- ECR repository

---

## Observability stack

Prometheus scrapes `serving:8000/metrics` and `ingestion:8001/metrics`.

Grafana dashboards in `infra/grafana/dashboards/`:
- `nexus_cv.json` — serving latency, SLA, anomalies, circuit breaker
- `nexus_cv_ingestion.json` — ingestion pipeline

Loki config at `infra/loki/loki-config.yml` for log aggregation (optional sidecar).

---

## Environment variables (production)

| Variable | Required | Description |
|----------|----------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | MLflow server URL |
| `YOLO_MODEL_PATH` | Yes | Model weights path or registry URI |
| `RECORDING_ENABLED` | No | Enable SQLite replay recording |
| `MLOPS_RETRAINING_ENABLED` | No | Enable drift-based retraining |
| `RAY_NUM_CPUS` | No | Ray cluster CPUs (default 4) |
| `RAY_NUM_GPUS` | No | Ray GPUs (default 0) |
