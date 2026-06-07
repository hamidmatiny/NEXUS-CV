"""Prometheus metrics for the NEXUS-CV ingestion pipeline."""

from __future__ import annotations

import threading

from prometheus_client import Counter, Gauge, Histogram

FRAMES_PROCESSED = Counter(
    "nexus_cv_frames_processed_total",
    "Total frames processed",
    ["camera_id"],
)
DETECTIONS_TOTAL = Counter(
    "nexus_cv_detections_total",
    "Total detections across all frames",
    ["camera_id", "class_name"],
)
INFERENCE_DURATION_MS = Histogram(
    "nexus_cv_yolo_inference_duration_ms",
    "YOLO inference duration in milliseconds",
    ["camera_id"],
    buckets=[5, 10, 20, 30, 50, 100, 200, 500],
)
QUARANTINE_TOTAL = Counter(
    "nexus_cv_quarantine_total",
    "Total frames quarantined by schema validation",
    ["camera_id"],
)
ACTIVE_CAMERAS = Gauge(
    "nexus_cv_active_cameras",
    "Number of camera pipelines currently running",
)

_api_thread: threading.Thread | None = None


def start_metrics_server(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start the FastAPI metrics/health server on the given host and port.

    Metrics are exposed at ``GET /metrics`` via the shared prometheus_client
    registry populated by the ingestion pipeline counters above.

    Args:
        host: Bind address (default all interfaces for Docker).
        port: TCP port (default 8000, matching docker-compose internal port).

    Returns:
        Background daemon thread running uvicorn.
    """
    global _api_thread
    if _api_thread is not None and _api_thread.is_alive():
        return _api_thread

    import uvicorn

    from ingestion.app import app

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    _api_thread = threading.Thread(
        target=server.run,
        name="ingestion-metrics-api",
        daemon=True,
    )
    _api_thread.start()
    return _api_thread
