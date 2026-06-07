"""Ray actor maintaining per-camera frame buffers with detections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import ray

    _ray_remote = ray.remote
except ImportError:
    ray = None  # type: ignore[assignment]

    def _ray_remote(*args: object, **kwargs: object) -> object:
        """No-op Ray remote decorator when Ray is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(cls: type) -> type:
            return cls

        return _wrap
import structlog

from config.settings import get_settings
from ingestion.stream_capture import FramePacket
from ingestion.yolo_detector import Detection

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    """A frame stored in the buffer with its detections and latency.

    Attributes:
        packet: Original captured frame packet.
        detections: Object detections associated with this frame.
        detection_latency_ms: Time spent on detection inference in milliseconds.
    """

    packet: FramePacket
    detections: list[Detection]
    detection_latency_ms: float


@_ray_remote
class FrameBufferActor:
    """Ray actor that maintains bounded deques of buffered frames per camera.

    Thread-safety is provided by Ray's actor model: each method invocation
    is processed serially within the actor, so no explicit locking is required.
    Concurrent callers receive consistent snapshots via Ray's RPC layer.
    """

    def __init__(self, buffer_size: int | None = None) -> None:
        """Initialize the frame buffer actor.

        Args:
            buffer_size: Maximum frames per camera deque. Defaults to
                ``settings.FRAME_BUFFER_SIZE``.
        """
        settings = get_settings()
        self._buffer_size = buffer_size or settings.FRAME_BUFFER_SIZE
        self._buffers: dict[str, deque[BufferedFrame]] = {}
        self._push_counts: dict[str, int] = {}
        self._eviction_counts: dict[str, int] = {}

        logger.info("frame_buffer_actor_initialized", buffer_size=self._buffer_size)

    def push(
        self,
        packet: FramePacket,
        detections: list[Detection],
    ) -> None:
        """Push a frame and its detections into the camera buffer.

        When the buffer exceeds ``buffer_size``, the oldest frame is evicted.

        Args:
            packet: Captured frame packet.
            detections: Detection results for this frame.
        """
        camera_id = packet.camera_id
        if camera_id not in self._buffers:
            self._buffers[camera_id] = deque(maxlen=self._buffer_size)
            self._push_counts[camera_id] = 0
            self._eviction_counts[camera_id] = 0

        buf = self._buffers[camera_id]
        if len(buf) >= self._buffer_size:
            self._eviction_counts[camera_id] += 1

        buffered = BufferedFrame(
            packet=packet,
            detections=detections,
            detection_latency_ms=0.0,
        )
        buf.append(buffered)
        self._push_counts[camera_id] += 1

    def get_latest(self, camera_id: str) -> BufferedFrame | None:
        """Return the most recent buffered frame for a camera.

        Args:
            camera_id: Camera identifier.

        Returns:
            Latest BufferedFrame or None if the buffer is empty.
        """
        buf = self._buffers.get(camera_id)
        if not buf:
            return None
        return buf[-1]

    def get_window(self, camera_id: str, n: int) -> list[BufferedFrame]:
        """Return the last ``n`` buffered frames for a camera.

        Args:
            camera_id: Camera identifier.
            n: Number of frames to retrieve (most recent first in order).

        Returns:
            List of up to ``n`` BufferedFrame instances, oldest to newest.
        """
        buf = self._buffers.get(camera_id)
        if not buf or n <= 0:
            return []
        return list(buf)[-n:]

    def get_stats(self) -> dict[str, object]:
        """Return buffer statistics across all cameras.

        Returns:
            Dictionary with per-camera push counts, eviction counts,
            and current buffer lengths.
        """
        stats: dict[str, object] = {
            "buffer_size": self._buffer_size,
            "cameras": {},
        }
        cameras_stats: dict[str, dict[str, int]] = {}
        for camera_id, buf in self._buffers.items():
            cameras_stats[camera_id] = {
                "current_length": len(buf),
                "total_pushes": self._push_counts.get(camera_id, 0),
                "total_evictions": self._eviction_counts.get(camera_id, 0),
            }
        stats["cameras"] = cameras_stats
        return stats
