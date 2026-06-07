"""Entry point for the NEXUS-CV ingestion pipeline."""

from __future__ import annotations

import logging

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


def main() -> None:
    """Initialize logging and report ready state for the ingestion service."""
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )
    logger.info(
        "ingestion_service_ready",
        num_cameras=settings.NUM_CAMERAS,
        frame_buffer_size=settings.FRAME_BUFFER_SIZE,
        yolo_model=settings.YOLO_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
