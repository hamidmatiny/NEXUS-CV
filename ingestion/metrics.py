"""Prometheus metrics for the NEXUS-CV ingestion pipeline."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from config.settings import get_settings

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
    "nexus_cv_inference_duration_ms",
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


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: TCP port to expose ``/metrics`` on.
    """
    _ = get_settings()
    start_http_server(port)
