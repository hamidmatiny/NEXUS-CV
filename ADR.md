# Architecture Decision Records

## ADR-001: Ray over Celery/Dask for Distributed Inference

**Status:** Accepted  
**Date:** 2026-06-07  
**Context:** NEXUS-CV requires a distributed runtime to buffer frames, serve detection models, and coordinate multi-camera pipelines. We evaluated Celery (task queue), Dask (parallel compute), and Ray (distributed actors + serve).

**Decision:** Use Ray 2.40 with Ray Actors for stateful frame buffering and Ray Serve for future model serving endpoints.

**Rationale:**

- **Stateful actors**: `FrameBufferActor` maintains per-camera deques with bounded memory. Ray actors provide serial execution guarantees without manual locking — Celery workers are stateless by design and would require Redis/external state.
- **Low-latency inference serving**: Ray Serve supports GPU-aware deployment groups with request batching. Celery adds broker round-trip latency unsuitable for sub-100 ms inference loops.
- **Unified runtime**: Ray covers actors, serve, and cluster management in one dependency. Dask excels at batch analytics but lacks first-class model serving and actor semantics.
- **GPU scheduling**: Ray's fractional GPU allocation (`RAY_NUM_GPUS=0.5`) enables colocating multiple detector replicas on a single GPU.

**Consequences:**

- Ray adds ~200 MB to the container image and requires a running head node.
- Team must learn Ray's actor and deployment APIs.
- Celery remains viable for async batch jobs (Phase 3 event processing) as a complementary tool, not a replacement.

---

## ADR-002: Pandera over Great Expectations for Schema Contracts

**Status:** Accepted  
**Date:** 2026-06-07  
**Context:** Detection outputs must be validated before entering the frame buffer and downstream analytics. Candidates: Pandera (DataFrame-centric validation), Great Expectations (data documentation platform), and manual pydantic models.

**Decision:** Use Pandera 0.21 with a `DataFrameModel` schema and row-level validation in `validate_detections()`.

**Rationale:**

- **Python-native integration**: Pandera validates pandas DataFrames in-process with minimal overhead. Great Expectations requires a data context, checkpoint infrastructure, and typically a filesystem store — excessive for per-frame validation at 30 FPS × N cameras.
- **Type-safe schemas**: Pandera's `DataFrameModel` provides declarative constraints (`ge`, `le`, `str_length`) that map directly to our detection fields.
- **Quarantine workflow**: Failed rows serialize cleanly to Parquet via pandas without GE's expectation suite abstraction.
- **Dependency weight**: Pandera + pyarrow is ~30 MB. Great Expectations pulls altair, jinja2, and other doc-generation deps irrelevant to a real-time pipeline.

**Consequences:**

- No automatic data documentation site (acceptable — ARCHITECTURE.md covers contracts).
- Row-by-row validation in Phase 1 is simple but may need vectorized validation at scale.
- Great Expectations may be adopted in Phase 3 for batch analytics quality gates.

---

## ADR-003: YOLO11 over RT-DETR for Edge Detection

**Status:** Accepted  
**Date:** 2026-06-07  
**Context:** Phase 1 requires a real-time object detector supporting PyTorch and TensorRT export for edge deployment. Primary candidates: Ultralytics YOLO11 and RT-DETR (Real-Time Detection Transformer).

**Decision:** Standardize on Ultralytics YOLO11 (`yolo11n.pt` default) with TensorRT export path via `YOLO_MODEL_PATH`.

**Rationale:**

- **Edge latency**: YOLO11n achieves ~2 ms inference on modern GPUs at 640×480, meeting our < 50 ms budget with headroom for batching and I/O. RT-DETR-l achieves comparable accuracy but 2–3× higher latency due to transformer decoder overhead.
- **Export ecosystem**: Ultralytics provides one-command TensorRT, ONNX, and OpenVINO export. RT-DETR export requires manual ONNX graph surgery for TensorRT optimization.
- **Batch inference**: YOLO natively supports multi-image batch prediction via `model.predict(frames)`. RT-DETR batching requires padding and custom collate logic.
- **Tracking integration**: Ultralytics bundles ByteTrack/BoT-SORT for `track_id` assignment in future phases.

**Consequences:**

- Transformer-based detectors (RT-DETR, DINO) deferred to Phase 4 for accuracy-critical scenarios.
- Model zoo locked to COCO 80-class taxonomy until custom training pipeline (Phase 2).
- TensorRT engines are hardware-specific; CI tests mock YOLO to avoid GPU/engine coupling.

---

## ADR-004: Multi-Service Port Separation (Ingestion vs Serving)

**Status:** Accepted  
**Date:** 2026-06-07  
**Context:** Phase 6 introduced two independently deployable FastAPI services — the Ray ingestion pipeline and the unified edge inference gateway — both requiring Prometheus `/metrics` endpoints and HTTP health probes. Running both on host port `8000` caused local port allocation conflicts and ambiguous scrape targets.

**Decision:** Bind both services to **`0.0.0.0:8000` inside their respective containers** on the Docker Compose `nexus-net` bridge network, but map distinct host ports:

| Service | Container port | Host mapping | Prometheus target |
|---------|---------------|--------------|-------------------|
| `ingestion` | `8000` | `8001:8000` | `ingestion:8000` |
| `serving` | `8000` | `8000:8000` | `serving:8000` |

**Rationale:**

- **DNS-based scraping:** Prometheus resolves services by Compose service name (`ingestion:8000`, `serving:8000`) on the internal network — no host port knowledge required.
- **Host conflict avoidance:** Developers can run both services simultaneously on localhost without `EADDRINUSE`.
- **Independent scaling:** Ingestion (continuous stream processing) and serving (request/response inference) can be scaled, restarted, and health-checked independently — a serving restart does not interrupt ingestion telemetry.
- **Telemetry isolation:** Ingestion exports YOLO-specific histograms (`nexus_cv_yolo_inference_duration_ms`); serving exports end-to-end pipeline and SLA metrics — separate scrape jobs prevent metric cardinality collisions in mixed dashboards.

**Consequences:**

- Documentation must clearly distinguish host port `8001` (ingestion) from `8000` (serving).
- Kubernetes Helm chart uses a single serving deployment on port `8000`; ingestion is a separate workload in production.
- Both services share internal port `8000` — only safe because they run in isolated network namespaces.

---

## ADR-005: Graceful Degradation via Fallback Frames

**Status:** Accepted  
**Date:** 2026-06-07  
**Context:** The serving gateway accepts base64-encoded JPEG/PNG frames from HTTP clients, WebSocket streams, and integration tests. Corrupt payloads, truncated base64, or invalid image bytes previously risked crashing the inference pipeline with unhandled decode exceptions, causing cascading 500 errors and circuit breaker trips under high-throughput video ingestion.

**Decision:** Implement **`decode_frame()` graceful degradation** in `serving/deployments.py`: on any decode failure, log a structured warning and return an in-memory black frame array:

```python
fallback = np.zeros((480, 640, 3), dtype=np.uint8)
```

The pipeline proceeds with zero detections rather than raising an HTTP error.

**Rationale:**

- **Gateway uptime:** A single corrupt frame from a degraded camera stream must not take down the serving gateway or trip the circuit breaker for all clients.
- **Test ergonomics:** Unit and integration tests can send minimal/mock payloads without crafting valid JPEG bytes — the pipeline returns a valid (empty) `InferenceResponse`.
- **Observability preserved:** `frame_decode_failed` warnings in structlog and empty detection arrays are visible in Prometheus counters and the dashboard — operators can detect upstream data quality issues without service interruption.
- **Consistent tensor shape:** The 640×480×3 BGR shape matches the pipeline's expected input dimensions, avoiding downstream shape mismatch errors in fusion and intelligence stages.

**Trade-offs:**

| Benefit | Cost |
|---------|------|
| 99.99%+ gateway availability under corrupt payloads | Empty inference results may mask silent data loss if warnings are not monitored |
| No circuit breaker cascade from decode exceptions | Operators must alert on `frame_decode_failed` log rate |
| Seamless test/CI integration | Black-frame inferences consume compute (YOLO forward pass on empty input) |
| WebSocket streams stay connected | Replay recordings of corrupt frames show blank video — requires session debugger review |

**Alternatives considered:**

1. **Strict 500 rejection** — Rejected: causes cascading failures and WebSocket disconnects under real-world RTSP degradation.
2. **Skip frame (no response)** — Rejected: breaks WebSocket request/response pairing and SLA measurement continuity.
3. **Client-side validation only** — Rejected: cannot trust edge cameras or third-party integrators to always send valid payloads.

**Consequences:**

- Monitoring must include `frame_decode_failed` warning rate alongside SLA breach metrics.
- Detection accuracy metrics may include false empty frames — filter by `detections.length > 0` in analytics.
- Documented in [docs/architecture.md](docs/architecture.md) §7 and [docs/business_case.md](docs/business_case.md).
