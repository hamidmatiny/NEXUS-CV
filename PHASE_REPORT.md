# NEXUS-CV Phase Report

Engineering log for the six-phase build of NEXUS-CV — a production-grade real-time multi-modal computer vision platform.

**Release status:** Phase 6 verified at **100% complete**. **83 automated tests passing.**

---

## Phase 1 — Ingestion Pipeline

**Goal:** Ingest multi-camera video streams, run YOLO detection, validate schema contracts, and buffer frames in a Ray actor for downstream consumption.

**Key design decisions:**
- Pandera schemas for detection validation with quarantine on contract violation
- `FrameBufferActor` as a bounded deque per camera to prevent unbounded memory growth
- Lazy YOLO model loading with fallback to `yolo11n.pt`
- Synthetic stream generator for CI and local dev without real cameras

**Performance results:**
- YOLO11n inference: ~8–15 ms/frame on M2 MacBook (MPS)
- Frame buffer push/pop: <0.1 ms per operation
- 20+ unit tests at phase completion, all CPU-only with mocked YOLO

**What was hardest:** Balancing Ray actor lifecycle with pytest — solved with session-scoped Ray init in `conftest.py` and autouse fixtures that shut down cleanly.

---

## Phase 2 — Multi-Modal Fusion

**Goal:** Fuse camera detections with simulated LiDAR/radar readings, align sensors temporally, and maintain stable tracks via Kalman filtering.

**Key design decisions:**
- `TemporalAligner` with configurable max offset (default 50 ms)
- `MultiObjectTracker` wrapping per-track Kalman filters with Hungarian assignment
- `FusionActor` as Ray actor for serial, thread-safe fusion state
- LiDAR/radar simulators generate plausible 3D readings from 2D bboxes for demo/CI

**Performance results:**
- Fusion + tracking: ~2–4 ms/frame for 10 active tracks
- Sensor alignment buffer: O(1) insert, O(n) match within small n

**What was hardest:** Consistent track ID assignment across frames when detections flicker — mitigated with IoU gating and velocity-aware prediction.

---

## Phase 3 — Stacked AI Intelligence

**Goal:** Layer scene classification, trajectory LSTM prediction, and anomaly scoring on fused tracks; expose unified `IntelligenceEnsemble`.

**Key design decisions:**
- ViT scene classifier with 30-frame hash cache and heuristic label mapping
- Trajectory LSTM with normalized coordinate features and MPS/CUDA device selection
- Anomaly scorer combining velocity, trajectory deviation, and scene-context signals
- Graceful degradation when transformers/torch unavailable (heuristic scene fallback)

**Performance results:**
- Scene classification (cached): ~0.5 ms; cold ViT forward: ~25 ms
- Trajectory LSTM: ~3 ms for 10 tracks
- Anomaly scoring: ~1 ms

**What was hardest:** BGR→PIL conversion for Hugging Face pipeline input — OpenCV frames must be converted before ViT inference.

---

## Phase 4 — Serving Layer

**Goal:** Production FastAPI gateway, Ray Serve deployments, gRPC stub, Prometheus metrics, circuit breaker, and correlation ID middleware.

**Key design decisions:**
- `LocalPipeline` for single-process dev; Ray Serve deployments for production scale
- SLA threshold at 30 ms with `SLA_BREACH_TOTAL` counter
- Circuit breaker middleware with configurable failure threshold
- Health endpoint aggregates Ray, model, and component status
- **`decode_frame()` fallback** — corrupt base64 returns `np.zeros((480, 640, 3))` instead of HTTP 500 (ADR-005)

**Performance results:**
- End-to-end serving (local pipeline, MPS): p50 ~16 ms, p95 ~28 ms (1 camera)
- WebSocket streaming: same latency as REST infer path

**What was hardest:** FastAPI lifespan integration with MLOps background scheduler without blocking startup.

---

## Phase 5 — MLOps Lifecycle

**Goal:** Experiment tracking, drift monitoring, automated retraining orchestration, model registry, and DVC data versioning.

**Key design decisions:**
- MLflow for experiment tracking and model registry
- Evidently for drift reports (HTML output to `reports/`)
- Serving-layer scheduler dumps operational parquet windows and triggers retrain on drift
- Minimum hours between retraining to prevent thrashing

**Performance results:**
- Drift check on 10k-row parquet: ~2 s
- Registry model download + cache: ~5 s (YOLO11n)

**What was hardest:** Wiring drift evaluation into the hot inference path without adding latency — solved with async background task and batched parquet writes.

---

## Phase 6 — Observability, Infrastructure & Polish ✅ 100% Complete

**Goal:** Live observability dashboard, full cloud/K8s infrastructure, CI/CD, Grafana dashboards, and documentation that makes the repo production-ready.

**Key design decisions:**
- **Dual-service Docker Compose topology** — ingestion (`8001:8000`) and serving (`8000:8000`) on `nexus-net` with independent health checks (ADR-004)
- Dashboard WebSocket pub/sub via asyncio queues (`/ws/dashboard/{camera_id}`)
- SQLite recording store gated by `RECORDING_ENABLED` for time-travel session debugger
- React 18 + Vite + Tailwind frontend with zero external UI libraries
- FastAPI metrics API on ingestion (`ingestion/app.py`) binding `0.0.0.0:8000`
- MLflow startup retry via `wait_for_mlflow()` — both services log startup runs
- Terraform modules for GCP Cloud Run and AWS ECS Fargate
- Helm chart with HPA on CPU and custom `nexus_cv_inference_queue_depth` metric
- Parallel CI jobs: lint, test (83), build, Trivy CRITICAL gate

**Performance results (verified live):**

| Metric | Observed value |
|--------|----------------|
| Ingestion YOLO (Docker CPU) | ~105 ms/frame (exceeds 30 ms SLA — expected without GPU) |
| Serving p50 (MPS, 1 cam) | ~16 ms (within SLA) |
| Dashboard WebSocket fan-out | <1 ms per subscriber |
| SQLite frame record | ~0.5 ms/frame |
| Prometheus scrape | Both `nexus-cv-ingestion` and `nexus-cv-serving` targets UP |
| MLflow startup runs | `ingestion-startup`, `serving-startup` registered |

**SLA breach surfacing verified:**
- `SLA_BREACH_TOTAL` increments when `inference_ms > 30.0`
- Structured `sla_breach` warnings in structlog JSON
- Grafana SLA breach rate panel configured (red > 1%)
- Dashboard `MetricsPanel` sparkline reflects live breach rate

**What was hardest:**
1. Port alignment — ingestion metrics server on wrong port (8001 vs expected 8000) caused Prometheus connection refused; fixed with FastAPI uvicorn on `0.0.0.0:8000`.
2. Coordinating dashboard payload shape across backend broadcast, SQLite replay, and frontend TypeScript types while keeping the gateway hot path non-blocking.
3. Prometheus metric name collision between ingestion and serving histograms — resolved by renaming to `nexus_cv_yolo_inference_duration_ms`.

---

## Deliverable Audit Summary

| Deliverable | Status | Evidence |
|-------------|--------|----------|
| 83 automated tests | ✅ Verified | `pytest tests/ -v` |
| Parallel CI (lint/test/build/security) | ✅ Verified | `.github/workflows/ci.yml` |
| Trivy CRITICAL CVE gate | ✅ Verified | `security` job, `exit-code: 1` |
| Multi-stage Docker `production` target | ✅ Verified | `docker/Dockerfile.serving` |
| Ingestion metrics on `:8000` | ✅ Verified | `curl localhost:8001/metrics` |
| Serving metrics on `:8000` | ✅ Verified | `curl localhost:8000/metrics` |
| Prometheus dual scrape | ✅ Verified | Both targets UP |
| MLflow startup telemetry | ✅ Verified | 2 runs in `nexus-cv` experiment |
| React dashboard production build | ✅ Verified | `npm run build` |
| Graceful fallback frames | ✅ Verified | `decode_frame()` in deployments.py |
| Session replay API | ✅ Verified | 3 replay API tests passing |
| Terraform GCP + AWS | ✅ Delivered | `infra/terraform/` |
| Helm HPA chart | ✅ Delivered | `infra/helm/nexus-cv/` |
| Grafana dashboards | ✅ Delivered | `infra/grafana/dashboards/` |
| Documentation suite | ✅ Delivered | README, architecture, ADRs, benchmarks, business case |

---

## Phase Summary

| Phase | Modules | Tests | Status |
|-------|---------|-------|--------|
| 1 | `ingestion/` | 20+ | ✅ Complete |
| 2 | `fusion/` | 15+ | ✅ Complete |
| 3 | `intelligence/` | 12+ | ✅ Complete |
| 4 | `serving/` | 10+ | ✅ Complete |
| 5 | `mlops/` | 8+ | ✅ Complete |
| 6 | `dashboard/`, `infra/`, CI | 5+ | ✅ **100% Complete** |

**Total: 83 tests passing.** Full Docker Compose stack operational. Terraform (GCP + AWS), Helm chart, Grafana dashboards, live React 18 dashboard, and comprehensive documentation delivered.

---

## Related Documents

- [docs/architecture.md](docs/architecture.md) — Full system architecture
- [docs/business_case.md](docs/business_case.md) — Business value and ROI
- [BENCHMARKS.md](BENCHMARKS.md) — Performance profiles and SLA analysis
- [ADR.md](ADR.md) — Architecture Decision Records (001–005)
