"""Prometheus metrics for the Ray Serve inference gateway."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

SERVE_INFERENCE_DURATION_MS = Histogram(
    "nexus_cv_inference_duration_ms",
    "End-to-end inference duration in milliseconds",
    ["deployment", "camera_id"],
    buckets=[5, 10, 20, 30, 50, 100, 200, 500],
)
SLA_BREACH_TOTAL = Counter(
    "nexus_cv_sla_breach_total",
    "Total inference requests exceeding the 30ms SLA",
)
ACTIVE_TRACKS = Gauge(
    "nexus_cv_active_tracks",
    "Number of active fused tracks in the serving pipeline",
)
ANOMALY_DETECTIONS_TOTAL = Counter(
    "nexus_cv_anomaly_detections_total",
    "Total anomaly detections emitted by the intelligence layer",
    ["camera_id", "factor"],
)
CIRCUIT_BREAKER_STATE = Gauge(
    "nexus_cv_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open)",
)
SERVING_DURATION_MS = Histogram(
    "nexus_cv_serving_duration_ms",
    "Wall-clock serving duration from request receipt to response",
    ["endpoint"],
    buckets=[5, 10, 20, 30, 50, 100, 200, 500, 1000],
)
