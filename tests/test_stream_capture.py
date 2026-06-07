"""Unit tests for StreamCapture."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ingestion.stream_capture import FramePacket, StreamCapture


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthetic_frame_packet_structure() -> None:
    """Synthetic frames should emit valid FramePacket instances."""
    capture = StreamCapture(source_uri="synthetic://local")
    gen = capture.read_frames("cam_test")

    packet = await asyncio.wait_for(anext(gen), timeout=5.0)

    assert isinstance(packet, FramePacket)
    assert packet.camera_id == "cam_test"
    assert packet.frame_id == 0
    assert packet.timestamp_ns > 0
    assert packet.frame.shape == (480, 640, 3)
    assert packet.frame.dtype == np.uint8
    assert packet.source_uri == "synthetic://local"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthetic_frames_increment_frame_id() -> None:
    """Frame IDs should increment monotonically."""
    capture = StreamCapture(source_uri="synthetic://local")
    gen = capture.read_frames("cam_01")

    packets = []
    for _ in range(3):
        packets.append(await asyncio.wait_for(anext(gen), timeout=5.0))

    assert [p.frame_id for p in packets] == [0, 1, 2]
    assert all(p.camera_id == "cam_01" for p in packets)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconnect_exponential_backoff() -> None:
    """RTSP failures should trigger exponential backoff reconnects."""
    capture = StreamCapture(source_uri="rtsp://invalid.local/stream")
    sleep_delays: list[float] = []

    async def mock_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with (
        patch("ingestion.stream_capture.asyncio.to_thread", return_value=mock_cap),
        patch("ingestion.stream_capture.asyncio.sleep", side_effect=mock_sleep),
    ):
        gen = capture.read_frames("cam_rtsp")
        with pytest.raises(ConnectionError):
            async for _ in gen:
                pass

    assert len(sleep_delays) == 5
    assert sleep_delays == [1.0, 2.0, 4.0, 8.0, 16.0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconnect_resets_on_success() -> None:
    """Successful reconnection after failure should yield valid frames."""
    capture = StreamCapture(source_uri="rtsp://test.local/stream")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    read_count = 0
    caps_created = 0

    class MockCap:
        """Mock VideoCapture returning test frames."""

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            nonlocal read_count
            read_count += 1
            if read_count <= 2:
                return True, frame
            return False, None

        def release(self) -> None:
            pass

    async def mock_to_thread(func: object, *args: object) -> object:
        nonlocal caps_created
        if args:
            caps_created += 1
            cap = MockCap()
            if caps_created == 1:
                cap.isOpened = lambda: False  # type: ignore[method-assign, assignment]
            return cap
        return func()  # type: ignore[operator]

    with (
        patch("ingestion.stream_capture.asyncio.to_thread", side_effect=mock_to_thread),
        patch("ingestion.stream_capture.asyncio.sleep", new_callable=AsyncMock),
    ):
        gen = capture.read_frames("cam_reset")
        packets = []
        try:
            async for packet in gen:
                packets.append(packet)
                if len(packets) >= 2:
                    break
        except ConnectionError:
            pass

    assert len(packets) >= 1
    assert packets[0].frame.shape == (480, 640, 3)
