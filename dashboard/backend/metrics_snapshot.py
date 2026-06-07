"""Helpers for building live dashboard metric snapshots."""

from __future__ import annotations

from prometheus_client import REGISTRY

from serving.metrics import ACTIVE_TRACKS


def _counter_total(metric_name: str) -> float:
    """Read a Prometheus counter total value.

    Args:
        metric_name: Metric name without suffix.

    Returns:
        Counter value or 0.0.
    """
    for metric in REGISTRY.collect():
        if metric.name == metric_name:
            return float(sum(s.value for s in metric.samples if s.name.endswith("_total")))
    return 0.0


def build_live_metrics(inference_ms: float, active_tracks: int | None = None) -> dict[str, float]:
    """Build a metrics snapshot for dashboard WebSocket payloads.

    Args:
        inference_ms: Latest inference latency in milliseconds.
        active_tracks: Optional active track count override.

    Returns:
        Metrics dict for dashboard consumption.
    """
    tracks = active_tracks if active_tracks is not None else int(ACTIVE_TRACKS._value.get())  # noqa: SLF001
    sla_breaches = _counter_total("nexus_cv_sla_breach")
    return {
        "inference_ms": round(inference_ms, 2),
        "active_tracks": float(tracks),
        "sla_breach_rate": sla_breaches,
        "anomaly_rate": 0.0,
    }
