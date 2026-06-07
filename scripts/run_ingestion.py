#!/usr/bin/env python3
"""Long-running ingestion pipeline entrypoint for the NEXUS-CV container."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import logging
import os
import signal
import time
from typing import TYPE_CHECKING

import ray
import structlog

from config.settings import get_settings
from ingestion import metrics
from ingestion.frame_buffer_actor import FrameBufferActor
from ingestion.schema_contracts import validate_detections
from ingestion.stream_capture import StreamCapture
from ingestion.yolo_detector import YOLODetector

if TYPE_CHECKING:
    import ray.actor

logger = structlog.get_logger(__name__)

FRAME_BUFFER_ACTOR_NAME = "frame_buffer"
DEFAULT_HEARTBEAT_FRAME_INTERVAL = 100


def _configure_logging(log_level: str) -> None:
    """Configure structlog for JSON output.

    Args:
        log_level: Logging level name (e.g. ``INFO``).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


def _heartbeat_interval() -> int:
    """Return the number of frames between pipeline heartbeat logs.

    Returns:
        Frame count interval, default 100.
    """
    raw = os.environ.get("HEARTBEAT_FRAME_INTERVAL", str(DEFAULT_HEARTBEAT_FRAME_INTERVAL))
    return max(1, int(raw))


def _camera_source_uri(camera_id: str) -> str:
    """Resolve the stream source URI for a camera.

    Args:
        camera_id: Camera identifier (e.g. ``cam_00``).

    Returns:
        RTSP URI or synthetic fallback URI.
    """
    env_key = f"CAMERA_SOURCE_{camera_id.upper()}"
    return os.environ.get(env_key, f"synthetic://{camera_id}")


async def run_camera_pipeline(
    camera_id: str,
    buffer_actor: ray.actor.ActorHandle,
    detector: YOLODetector,
    detect_lock: asyncio.Lock,
    stop: asyncio.Event,
    heartbeat_interval: int,
) -> None:
    """Run the full ingestion pipeline for a single camera.

    Reads frames, detects objects, validates schema, and pushes to the buffer
    actor until ``stop`` is set or the task is cancelled.

    Args:
        camera_id: Unique camera identifier.
        buffer_actor: Ray handle to FrameBufferActor.
        detector: Shared YOLO detector instance.
        detect_lock: Lock serializing inference across camera tasks.
        stop: Shutdown signal event.
        heartbeat_interval: Log heartbeat every N processed frames.
    """
    capture = StreamCapture(source_uri=_camera_source_uri(camera_id))
    frames_processed = 0
    detections_total = 0
    quarantine_total = 0

    try:
        async for packet in capture.read_frames(camera_id):
            if stop.is_set():
                break

            try:
                infer_start = time.perf_counter()
                async with detect_lock:
                    batch_results = await asyncio.to_thread(detector.detect_batch, [packet.frame])
                detection_latency_ms = (time.perf_counter() - infer_start) * 1000.0
                detections = batch_results[0]

                metrics.FRAMES_PROCESSED.labels(camera_id=camera_id).inc()
                metrics.INFERENCE_DURATION_MS.labels(camera_id=camera_id).observe(
                    detection_latency_ms
                )
                for det in detections:
                    metrics.DETECTIONS_TOTAL.labels(
                        camera_id=camera_id,
                        class_name=det.class_name,
                    ).inc()

                validation = await asyncio.to_thread(
                    validate_detections,
                    detections,
                    camera_id,
                    packet.timestamp_ns,
                )
                if validation.failed > 0:
                    quarantine_total += validation.failed
                    metrics.QUARANTINE_TOTAL.labels(camera_id=camera_id).inc(validation.failed)

                detections_total += len(detections)
                await asyncio.to_thread(
                    ray.get,
                    buffer_actor.push.remote(packet, detections),
                )
            except Exception as exc:
                logger.error(
                    "frame_processing_error",
                    camera_id=camera_id,
                    frame_id=packet.frame_id,
                    error=str(exc),
                )
                await asyncio.sleep(1.0)
                continue

            frames_processed += 1
            if frames_processed % heartbeat_interval == 0:
                logger.info(
                    "pipeline_heartbeat",
                    camera_id=camera_id,
                    frames_processed=frames_processed,
                    detections_total=detections_total,
                    quarantine_total=quarantine_total,
                )
    except asyncio.CancelledError:
        logger.info(
            "camera_pipeline_stopped",
            camera_id=camera_id,
            frames_processed=frames_processed,
        )
        raise


async def main() -> None:
    """Initialize Ray and run concurrent per-camera ingestion pipelines."""
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)
    heartbeat_interval = _heartbeat_interval()

    logger.info(
        "ingestion_service_ready",
        num_cameras=settings.NUM_CAMERAS,
        frame_buffer_size=settings.FRAME_BUFFER_SIZE,
        yolo_model=settings.YOLO_MODEL_PATH,
        heartbeat_frame_interval=heartbeat_interval,
    )

    if not ray.is_initialized():
        ray.init(
            num_cpus=settings.RAY_NUM_CPUS,
            num_gpus=settings.RAY_NUM_GPUS,
            ignore_reinit_error=True,
            include_dashboard=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
        )

    buffer_actor = FrameBufferActor.options(name=FRAME_BUFFER_ACTOR_NAME).remote()

    metrics.ACTIVE_CAMERAS.set(settings.NUM_CAMERAS)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    detector = YOLODetector()
    detect_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            run_camera_pipeline(
                camera_id=f"cam_{index:02d}",
                buffer_actor=buffer_actor,
                detector=detector,
                detect_lock=detect_lock,
                stop=stop,
                heartbeat_interval=heartbeat_interval,
            ),
            name=f"pipeline-cam_{index:02d}",
        )
        for index in range(settings.NUM_CAMERAS)
    ]

    def request_shutdown() -> None:
        """Cancel all pipeline tasks on SIGTERM/SIGINT."""
        logger.info("ingestion_shutdown_initiated", signal="SIGTERM/SIGINT")
        stop.set()
        for task in tasks:
            task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, request_shutdown)

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass

    metrics.ACTIVE_CAMERAS.set(0)
    ray.shutdown()
    logger.info("ingestion_shutdown_complete")


if __name__ == "__main__":
    metrics.start_metrics_server(port=8001)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    sys.exit(0)
