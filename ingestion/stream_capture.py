"""Async stream capture with RTSP support and synthetic fallback."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, cast

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

SYNTHETIC_WIDTH: Final[int] = 640
SYNTHETIC_HEIGHT: Final[int] = 480
MAX_RECONNECT_RETRIES: Final[int] = 5
RECONNECT_BASE_DELAY_S: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A single captured video frame with metadata.

    Attributes:
        camera_id: Unique identifier for the source camera.
        frame_id: Monotonically increasing frame counter.
        timestamp_ns: Capture timestamp in nanoseconds since epoch.
        frame: BGR image as a numpy array (H, W, 3).
        source_uri: URI or descriptor of the frame source.
    """

    camera_id: str
    frame_id: int
    timestamp_ns: int
    frame: NDArray[np.uint8]
    source_uri: str


class StreamCapture:
    """Captures video frames from RTSP streams or synthetic sources.

    Provides an async generator interface with automatic reconnection
    using exponential backoff when RTSP streams fail.
    """

    def __init__(self, source_uri: str | None = None) -> None:
        """Initialize stream capture.

        Args:
            source_uri: RTSP URI or file path. When None or ``synthetic://``,
                frames are generated locally for development.
        """
        self._source_uri = source_uri or "synthetic://local"
        self._use_synthetic = self._source_uri.startswith("synthetic://")

    async def read_frames(self, camera_id: str) -> AsyncIterator[FramePacket]:
        """Yield frames from the configured source.

        For RTSP sources, reconnects with exponential backoff (max 5 retries,
        base delay 1 s). Falls back to synthetic generation when the URI is
        not an RTSP stream.

        Args:
            camera_id: Identifier attached to every emitted FramePacket.

        Yields:
            FramePacket instances with BGR frames and metadata.
        """
        frame_id = 0
        if self._use_synthetic:
            async for packet in self._read_synthetic(camera_id, frame_id):
                yield packet
            return

        retry_count = 0
        while retry_count <= MAX_RECONNECT_RETRIES:
            cap: cv2.VideoCapture | None = None
            try:
                cap = await asyncio.to_thread(cv2.VideoCapture, self._source_uri)
                if not cap.isOpened():
                    raise ConnectionError(f"Failed to open stream: {self._source_uri}")

                logger.info(
                    "stream_connected",
                    camera_id=camera_id,
                    source_uri=self._source_uri,
                    retry_count=retry_count,
                )
                retry_count = 0

                while True:
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret or frame is None:
                        raise ConnectionError("Frame read failed")

                    packet = FramePacket(
                        camera_id=camera_id,
                        frame_id=frame_id,
                        timestamp_ns=time.time_ns(),
                        frame=cast(NDArray[np.uint8], frame),
                        source_uri=self._source_uri,
                    )
                    frame_id += 1
                    yield packet

            except (ConnectionError, OSError) as exc:
                retry_count += 1
                if retry_count > MAX_RECONNECT_RETRIES:
                    logger.error(
                        "stream_reconnect_exhausted",
                        camera_id=camera_id,
                        source_uri=self._source_uri,
                        retries=MAX_RECONNECT_RETRIES,
                        error=str(exc),
                    )
                    raise

                delay = RECONNECT_BASE_DELAY_S * (2 ** (retry_count - 1))
                logger.warning(
                    "stream_reconnect",
                    camera_id=camera_id,
                    source_uri=self._source_uri,
                    retry_count=retry_count,
                    delay_s=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
            finally:
                if cap is not None:
                    await asyncio.to_thread(cap.release)

    async def _read_synthetic(
        self, camera_id: str, start_frame_id: int
    ) -> AsyncIterator[FramePacket]:
        """Generate synthetic frames with random motion blobs.

        Args:
            camera_id: Camera identifier for emitted packets.
            start_frame_id: Initial frame counter value.

        Yields:
            Synthetic FramePacket instances (640x480 BGR).
        """
        frame_id = start_frame_id
        rng = np.random.default_rng(seed=hash(camera_id) & 0xFFFFFFFF)

        blob_x = rng.integers(50, SYNTHETIC_WIDTH - 50)
        blob_y = rng.integers(50, SYNTHETIC_HEIGHT - 50)
        blob_vx = rng.integers(-5, 6)
        blob_vy = rng.integers(-5, 6)

        while True:
            frame = np.zeros((SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH, 3), dtype=np.uint8)
            frame[:] = (20, 20, 30)

            blob_x = int(np.clip(blob_x + blob_vx, 30, SYNTHETIC_WIDTH - 30))
            blob_y = int(np.clip(blob_y + blob_vy, 30, SYNTHETIC_HEIGHT - 30))
            if blob_x <= 30 or blob_x >= SYNTHETIC_WIDTH - 30:
                blob_vx *= -1
            if blob_y <= 30 or blob_y >= SYNTHETIC_HEIGHT - 30:
                blob_vy *= -1

            color = (
                int(rng.integers(80, 255)),
                int(rng.integers(80, 255)),
                int(rng.integers(80, 255)),
            )
            cv2.rectangle(
                frame,
                (blob_x - 25, blob_y - 25),
                (blob_x + 25, blob_y + 25),
                color,
                -1,
            )

            noise = rng.integers(0, 15, frame.shape, dtype=np.uint8)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            yield FramePacket(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ns=time.time_ns(),
                frame=frame,
                source_uri=self._source_uri,
            )
            frame_id += 1
            await asyncio.sleep(1.0 / 30.0)
