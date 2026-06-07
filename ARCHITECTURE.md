# NEXUS-CV Architecture

Internal engineering reference for the NEXUS-CV real-time multi-modal computer vision intelligence platform.

## Overview

NEXUS-CV ingests live camera streams, runs object detection via YOLO11, validates outputs against schema contracts, and buffers annotated frames in a distributed Ray actor for downstream analytics. Phase 1 establishes the ingestion foundation: stream capture, detection, buffering, validation, and observability scaffolding.

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Language | Python 3.12 | Core runtime |
| Stream I/O | OpenCV + asyncio | Non-blocking RTSP/file capture |
| Inference | Ultralytics YOLO11 | Object detection (PyTorch / TensorRT) |
| Distributed runtime | Ray 2.40 | Actor-based frame buffering |
| Schema validation | Pandera 0.21 | Detection contract enforcement |
| Configuration | pydantic-settings | Typed env-based config |
| Logging | structlog | Structured JSON logs |
| Experiment tracking | MLflow | Model registry and artifacts |
| Metrics | Prometheus + Grafana | Pipeline observability |
| Containerization | Docker Compose | Local and CI deployment |

## Data Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ RTSP Camera │────▶│  StreamCapture   │────▶│  FramePacket    │
│  or Synthetic│     │  (async I/O)     │     │  (BGR + meta)   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Quarantine  │◀────│ validate_        │◀────│  YOLODetector   │
│  (Parquet)  │     │ detections()     │     │  detect_batch() │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ FrameBufferActor│
                                              │  (Ray, per-cam  │
                                              │   deque buffer) │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Downstream      │
                                              │ (Phase 2+)      │
                                              └─────────────────┘
```

## Component Responsibilities

### StreamCapture (`ingestion/stream_capture.py`)

- Async generator interface yielding `FramePacket` dataclass instances.
- RTSP ingestion with exponential backoff reconnect (5 retries, 1 s base).
- Synthetic fallback (`synthetic://`) for local dev and CI without hardware.
- All blocking I/O delegated to `asyncio.to_thread`.

### YOLODetector (`ingestion/yolo_detector.py`)

- Wraps Ultralytics YOLO with lazy model loading.
- Single-frame (`detect`) and batch (`detect_batch`) inference paths.
- Configurable confidence and IoU thresholds via settings.
- TensorRT engine support via `YOLO_MODEL_PATH` (`.engine` extension).
- Logs `inference_ms` per batch for latency tracking.

### FrameBufferActor (`ingestion/frame_buffer_actor.py`)

- Ray remote actor maintaining `collections.deque(maxlen=N)` per camera.
- Stores `BufferedFrame` (packet + detections + latency).
- Exposes `push`, `get_latest`, `get_window`, and `get_stats`.
- Thread-safety guaranteed by Ray's serial actor execution model.

### Schema Contracts (`ingestion/schema_contracts.py`)

- Pandera `detection_schema` enforcing confidence range, non-negative bbox coordinates, valid class_id, and non-empty camera_id.
- Failed rows quarantined to `QUARANTINE_DIR/YYYYMMDD_HHMMSS_{camera_id}.parquet`.
- Returns `ValidationResult` with pass/fail counts.

### Configuration (`config/settings.py`)

- Single `Settings` class loaded from `.env` via pydantic-settings.
- Cached singleton via `get_settings()` for zero-overhead access.

## Deployment Topology

```
docker-compose.yml
├── ingestion    → Dockerfile.ingestion (Python 3.12, non-root nexus user)
├── mlflow       → Model registry and experiment tracking
├── prometheus   → Metrics scraping
└── grafana      → Dashboards
```

Volume mounts: `./data/quarantine`, `./models`, MLflow artifacts.

## Phase Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Ingestion, detection, buffering, validation | **Complete** |
| 2 | Multi-modal fusion (audio, metadata) | Planned |
| 3 | Real-time alerting and event bus | Planned |
| 4 | Edge deployment and TensorRT optimization | Planned |
| 5 | Production hardening (auth, multi-tenancy) | Planned |

## Non-Functional Requirements

- **Latency**: Detection batch inference logged per call; target < 50 ms on GPU for yolo11n.
- **Reliability**: RTSP reconnect with backoff; schema quarantine prevents bad data propagation.
- **Testability**: All unit tests run CPU-only with mocked YOLO; synthetic streams for CI.
- **Observability**: structlog JSON logs; Prometheus/Grafana stack in compose.
