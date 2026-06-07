#!/usr/bin/env python3
"""Production serving entrypoint with MLflow readiness wait."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
import uvicorn

from config.settings import get_settings
from mlops.mlflow_utils import log_service_startup
from serving.deployments import get_shared_pipeline
from serving.gateway import app, configure_pipeline

logger = structlog.get_logger(__name__)


def main() -> None:
    """Wait for MLflow, configure the pipeline, and start the gateway."""
    settings = get_settings()
    log_service_startup(
        settings.MLFLOW_TRACKING_URI,
        service="serving",
        params={"mlops_retraining_enabled": settings.MLOPS_RETRAINING_ENABLED},
    )
    configure_pipeline(get_shared_pipeline().remote)
    logger.info("serving_gateway_starting", host="0.0.0.0", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
