"""Health check models and probes for the serving gateway."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel

from config.settings import get_settings

logger = structlog.get_logger(__name__)

HealthStatus = Literal["healthy", "degraded", "unhealthy"]
NEXUS_VERSION = "0.1.0"
_START_TIME = time.time()


class ComponentHealth(BaseModel):
    """Health status for a single system component."""

    status: HealthStatus
    last_check_ts: float
    error: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health response for the serving cluster."""

    status: HealthStatus
    components: dict[str, ComponentHealth]
    uptime_s: float
    version: str = NEXUS_VERSION


def _check_ray() -> ComponentHealth:
    """Verify Ray cluster connectivity.

    Returns:
        ComponentHealth for the Ray cluster.
    """
    ts = time.time()
    try:
        import ray

        if ray.is_initialized():
            ray.cluster_resources()
            return ComponentHealth(status="healthy", last_check_ts=ts)
        return ComponentHealth(
            status="healthy",
            last_check_ts=ts,
            error="local mode (Ray Serve not active)",
        )
    except Exception as exc:
        return ComponentHealth(status="degraded", last_check_ts=ts, error=str(exc))


def _check_deployments() -> ComponentHealth:
    """Verify Ray Serve deployments are reachable.

    Returns:
        ComponentHealth for Serve deployments.
    """
    ts = time.time()
    try:
        import ray

        if not ray.is_initialized():
            return ComponentHealth(
                status="healthy",
                last_check_ts=ts,
                error="local mode",
            )

        from ray import serve

        status = serve.status()
        if status.applications:
            return ComponentHealth(status="healthy", last_check_ts=ts)
        return ComponentHealth(
            status="degraded",
            last_check_ts=ts,
            error="No applications deployed",
        )
    except Exception as exc:
        return ComponentHealth(status="degraded", last_check_ts=ts, error=str(exc))


def _check_models() -> ComponentHealth:
    """Verify required model files exist on disk.

    Returns:
        ComponentHealth for model artifacts.
    """
    ts = time.time()
    settings = get_settings()
    paths = [Path(settings.YOLO_MODEL_PATH), Path(settings.TRAJECTORY_LSTM_PATH)]
    missing = [str(p) for p in paths if not p.exists() and p.suffix in {".pt", ".engine"}]
    if missing:
        return ComponentHealth(
            status="degraded",
            last_check_ts=ts,
            error=f"Missing models: {missing}",
        )
    return ComponentHealth(status="healthy", last_check_ts=ts)


def _check_disk() -> ComponentHealth:
    """Verify sufficient free disk space (>1 GB).

    Returns:
        ComponentHealth for disk space.
    """
    ts = time.time()
    try:
        usage = shutil.disk_usage(Path.cwd())
        free_gb = usage.free / (1024**3)
        if free_gb < 1.0:
            return ComponentHealth(
                status="unhealthy",
                last_check_ts=ts,
                error=f"Low disk space: {free_gb:.2f} GB free",
            )
        return ComponentHealth(status="healthy", last_check_ts=ts)
    except Exception as exc:
        return ComponentHealth(status="degraded", last_check_ts=ts, error=str(exc))


def collect_health() -> HealthResponse:
    """Run all health probes and aggregate status.

    Returns:
        HealthResponse with per-component results.
    """
    components = {
        "ray": _check_ray(),
        "deployments": _check_deployments(),
        "models": _check_models(),
        "disk": _check_disk(),
    }
    statuses = [c.status for c in components.values()]
    if "unhealthy" in statuses:
        aggregate: HealthStatus = "unhealthy"
    elif "degraded" in statuses:
        aggregate = "degraded"
    else:
        aggregate = "healthy"

    logger.debug("health_check_complete", status=aggregate)
    return HealthResponse(
        status=aggregate,
        components=components,
        uptime_s=round(time.time() - _START_TIME, 2),
    )
