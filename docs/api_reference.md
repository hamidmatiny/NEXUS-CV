# API Reference

Base URL: `http://localhost:8000` (serving gateway)

---

## REST

### `GET /health`

Returns aggregate health status.

**Response 200/503:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_s": 123.4,
  "components": {
    "ray": {"status": "healthy", "detail": "local mode"},
    "models": {"status": "healthy"}
  }
}
```

### `GET /metrics`

Prometheus text exposition format.

Key metrics:
- `nexus_cv_serving_duration_ms` — request latency histogram
- `nexus_cv_active_tracks` — gauge per camera
- `nexus_cv_sla_breach_total` — SLA violations (>30 ms)
- `nexus_cv_anomaly_detections_total` — counter by camera
- `nexus_cv_circuit_breaker_state` — 0=closed, 1=open

### `POST /api/v1/infer`

Run single-frame inference through the full pipeline.

**Request:**
```json
{
  "camera_id": "cam_00",
  "frame_b64": "<base64 JPEG>",
  "timestamp_ns": 1710000000000000000
}
```

**Response 200:**
```json
{
  "request_id": "uuid",
  "camera_id": "cam_00",
  "timestamp_ns": 1710000000000000000,
  "detections": [...],
  "tracks": [...],
  "scene": {"scene_class": "highway", "confidence": 0.92, "top3": [...]},
  "anomalies": [...],
  "trajectories": [...],
  "inference_ms": 18.4,
  "serving_ms": 19.1
}
```

**Headers:** `X-Request-ID`, `X-Serving-Ms`, `X-Correlation-ID`

---

## WebSocket

### `WS /ws/stream/{camera_id}`

Bidirectional stream: client sends raw JPEG bytes, server responds with `InferenceResponse` JSON per frame.

### `WS /ws/dashboard/{camera_id}`

Server-push dashboard stream. JSON payload per frame:

```json
{
  "frame_b64": "<base64 JPEG>",
  "detections": [],
  "tracks": [],
  "trajectories": [],
  "anomalies": [],
  "scene": {},
  "metrics": {
    "inference_ms": 18.4,
    "active_tracks": 5,
    "sla_breach_rate": 0.02,
    "anomaly_rate": 0.1
  },
  "request_id": "uuid",
  "camera_id": "cam_00",
  "timestamp_ns": 1710000000000000000,
  "inference_ms": 18.4,
  "serving_ms": 19.1
}
```

---

## Replay API

Requires `RECORDING_ENABLED=true`.

### `GET /api/v1/replay/sessions`

List recorded sessions.

**Query:** `limit` (1–500, default 100)

### `GET /api/v1/replay/sessions/{session_id}/frames`

Paginated frame metadata.

**Query:** `offset`, `limit` (1–200, default 50)

### `GET /api/v1/replay/sessions/{session_id}/frames/{frame_id}`

Full inference payload for a recorded frame.

---

## gRPC

Protobuf definitions in `proto/nexus_cv.proto`.

Service: `NexusCVInference.Infer` — mirrors REST infer with binary frame payload.

Default port: **50051** (Helm/K8s service).

---

## Error responses

| Status | Meaning |
|--------|---------|
| 422 | Validation error (malformed request) |
| 503 | Circuit breaker open or unhealthy |
| 404 | Replay session/frame not found |
