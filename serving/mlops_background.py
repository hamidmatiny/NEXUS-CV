"""Background asyncio loop for hourly MLOps drift evaluation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

import structlog

from config.settings import get_settings
from mlops.mlflow_utils import wait_for_mlflow
from serving.mlops_scheduler import TIME_CHECK_INTERVAL_S, get_mlops_scheduler, reset_mlops_scheduler

logger = structlog.get_logger(__name__)

POLL_INTERVAL_S = 60.0
_background_task: asyncio.Task[None] | None = None


async def _mlops_background_loop() -> None:
    """Poll hourly and run drift evaluation when the interval elapses."""
    scheduler = get_mlops_scheduler()
    logger.info(
        "mlops_background_loop_started",
        poll_interval_s=POLL_INTERVAL_S,
        evaluation_interval_s=TIME_CHECK_INTERVAL_S,
    )
    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            if not scheduler.enabled:
                continue
            try:
                scheduler.check_time_based()
            except Exception as exc:
                logger.error("mlops_background_evaluation_failed", error=str(exc))
    except asyncio.CancelledError:
        logger.info("mlops_background_loop_stopped")
        raise


def start_background_orchestrator() -> asyncio.Task[None] | None:
    """Start the hourly background drift evaluation task.

    Returns:
        Asyncio task handle, or None when retraining is disabled.
    """
    global _background_task
    if not get_settings().MLOPS_RETRAINING_ENABLED:
        return None
    if _background_task is not None and not _background_task.done():
        return _background_task
    _background_task = asyncio.create_task(_mlops_background_loop())
    return _background_task


async def stop_background_orchestrator() -> None:
    """Cancel the background drift evaluation task."""
    global _background_task
    if _background_task is None:
        return
    _background_task.cancel()
    try:
        await _background_task
    except asyncio.CancelledError:
        pass
    _background_task = None


@asynccontextmanager
async def mlops_lifespan(_app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context managing the MLOps background orchestrator."""
    settings = get_settings()
    wait_for_mlflow(settings.MLFLOW_TRACKING_URI)
    start_background_orchestrator()
    yield
    await stop_background_orchestrator()
    reset_mlops_scheduler()
