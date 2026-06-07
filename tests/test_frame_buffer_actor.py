"""Unit tests for FrameBufferActor."""

from __future__ import annotations

import time

import pytest
import ray

from ingestion.frame_buffer_actor import BufferedFrame, FrameBufferActor
from ingestion.stream_capture import FramePacket
from ingestion.yolo_detector import Detection


@pytest.mark.unit
def test_push_and_get_latest(
    ray_cluster: None,
    synthetic_frame_packet: FramePacket,
    mock_detections: list[Detection],
) -> None:
    """Push should store frame retrievable via get_latest."""
    actor = FrameBufferActor.remote(buffer_size=5)
    ray.get(actor.push.remote(synthetic_frame_packet, mock_detections))

    latest: BufferedFrame | None = ray.get(actor.get_latest.remote("cam_00"))
    assert latest is not None
    assert latest.packet.frame_id == synthetic_frame_packet.frame_id
    assert len(latest.detections) == 2


@pytest.mark.unit
def test_get_window(
    ray_cluster: None,
    synthetic_frame: object,
    mock_detections: list[Detection],
) -> None:
    """get_window should return the last n frames in order."""
    actor = FrameBufferActor.remote(buffer_size=10)

    for i in range(5):
        packet = FramePacket(
            camera_id="cam_01",
            frame_id=i,
            timestamp_ns=time.time_ns(),
            frame=synthetic_frame,  # type: ignore[arg-type]
            source_uri="synthetic://test",
        )
        ray.get(actor.push.remote(packet, mock_detections))

    window: list[BufferedFrame] = ray.get(actor.get_window.remote("cam_01", 3))
    assert len(window) == 3
    assert [f.packet.frame_id for f in window] == [2, 3, 4]


@pytest.mark.unit
def test_buffer_eviction(
    ray_cluster: None,
    synthetic_frame: object,
    mock_detections: list[Detection],
) -> None:
    """Buffer should evict oldest frames when maxlen is exceeded."""
    buffer_size = 3
    actor = FrameBufferActor.remote(buffer_size=buffer_size)

    for i in range(5):
        packet = FramePacket(
            camera_id="cam_evict",
            frame_id=i,
            timestamp_ns=time.time_ns(),
            frame=synthetic_frame,  # type: ignore[arg-type]
            source_uri="synthetic://test",
        )
        ray.get(actor.push.remote(packet, mock_detections))

    window: list[BufferedFrame] = ray.get(actor.get_window.remote("cam_evict", 10))
    assert len(window) == buffer_size
    assert [f.packet.frame_id for f in window] == [2, 3, 4]

    stats = ray.get(actor.get_stats.remote())
    cam_stats = stats["cameras"]["cam_evict"]
    assert cam_stats["total_pushes"] == 5
    assert cam_stats["total_evictions"] == 2
    assert cam_stats["current_length"] == buffer_size


@pytest.mark.unit
def test_get_latest_empty_buffer(ray_cluster: None) -> None:
    """get_latest on empty buffer should return None."""
    actor = FrameBufferActor.remote(buffer_size=5)
    latest = ray.get(actor.get_latest.remote("nonexistent"))
    assert latest is None


@pytest.mark.unit
def test_get_stats(
    ray_cluster: None,
    synthetic_frame_packet: FramePacket,
    mock_detections: list[Detection],
) -> None:
    """get_stats should report buffer configuration and camera metrics."""
    actor = FrameBufferActor.remote(buffer_size=5)
    ray.get(actor.push.remote(synthetic_frame_packet, mock_detections))

    stats = ray.get(actor.get_stats.remote())
    assert stats["buffer_size"] == 5
    assert "cam_00" in stats["cameras"]
    assert stats["cameras"]["cam_00"]["total_pushes"] == 1
