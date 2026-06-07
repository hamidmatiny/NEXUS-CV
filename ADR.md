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
