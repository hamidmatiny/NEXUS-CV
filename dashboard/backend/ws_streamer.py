"""Dashboard WebSocket stream broadcasting inference output."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import suppress
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["dashboard"])

_subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
_lock = asyncio.Lock()


async def subscribe(camera_id: str) -> asyncio.Queue[dict[str, Any]]:
    """Subscribe to dashboard frames for a camera.

    Args:
        camera_id: Camera identifier.

    Returns:
        Async queue receiving dashboard frame payloads.
    """
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=30)
    async with _lock:
        _subscribers[camera_id].add(queue)
    logger.info("dashboard_subscriber_added", camera_id=camera_id)
    return queue


async def unsubscribe(camera_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Remove a dashboard subscriber queue.

    Args:
        camera_id: Camera identifier.
        queue: Queue to remove.
    """
    async with _lock:
        _subscribers[camera_id].discard(queue)
        if not _subscribers[camera_id]:
            del _subscribers[camera_id]
    logger.info("dashboard_subscriber_removed", camera_id=camera_id)


async def broadcast_inference(camera_id: str, payload: dict[str, Any]) -> None:
    """Broadcast an inference payload to all dashboard subscribers.

    Args:
        camera_id: Camera identifier.
        payload: Dashboard JSON payload.
    """
    async with _lock:
        queues = list(_subscribers.get(camera_id, set()))
    for queue in queues:
        if queue.full():
            with suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.debug("dashboard_queue_full", camera_id=camera_id)


@router.websocket("/ws/dashboard/{camera_id}")
async def dashboard_stream(websocket: WebSocket, camera_id: str) -> None:
    """Stream enriched inference JSON to the live observability dashboard.

    Subscribes to inference output broadcast from the serving pipeline and
    forwards each frame as JSON to connected dashboard clients.
    """
    await websocket.accept()
    queue = await subscribe(camera_id)
    logger.info("dashboard_websocket_connected", camera_id=camera_id)
    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("dashboard_websocket_disconnected", camera_id=camera_id)
    except Exception as exc:
        logger.warning("dashboard_websocket_error", camera_id=camera_id, error=str(exc))
    finally:
        await unsubscribe(camera_id, queue)


def build_dashboard_payload(
    response_dict: dict[str, Any],
    frame_b64: str,
    metrics: dict[str, float | int],
) -> dict[str, Any]:
    """Build the dashboard WebSocket payload from an inference response.

    Args:
        response_dict: Serialized InferenceResponse.
        frame_b64: Base64-encoded JPEG frame.
        metrics: Live metric snapshot.

    Returns:
        Dashboard payload dict.
    """
    return {
        "frame_b64": frame_b64,
        "detections": response_dict.get("detections", []),
        "tracks": response_dict.get("tracks", []),
        "trajectories": response_dict.get("trajectories", []),
        "anomalies": response_dict.get("anomalies", []),
        "scene": response_dict.get("scene", {}),
        "metrics": metrics,
        "request_id": response_dict.get("request_id"),
        "camera_id": response_dict.get("camera_id"),
        "timestamp_ns": response_dict.get("timestamp_ns"),
        "inference_ms": response_dict.get("inference_ms"),
        "serving_ms": response_dict.get("serving_ms"),
    }
