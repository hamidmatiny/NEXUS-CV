"""End-to-end tests for FusionActor."""

from __future__ import annotations

import time

import pytest
import ray

from fusion.data_types import CameraDetectionReading
from fusion.fusion_actor import FusionActor
from fusion.kalman_tracker import MultiObjectTracker
from fusion.lidar_simulator import LiDARSimulator
from fusion.radar_simulator import RadarSimulator
from fusion.sensor_alignment import CameraLiDARProjector, TemporalAligner
from ingestion.yolo_detector import Detection


def _moving_detections(frame_idx: int) -> list[Detection]:
    """Generate detections that move slightly each frame."""
    offset = frame_idx * 5.0
    return [
        Detection(
            bbox_xyxy=(10.0 + offset, 10.0, 60.0 + offset, 60.0),
            confidence=0.92,
            class_id=2,
            class_name="car",
        ),
        Detection(
            bbox_xyxy=(200.0, 100.0, 300.0, 200.0),
            confidence=0.85,
            class_id=0,
            class_name="person",
        ),
    ]


def _build_aligned(frame_idx: int, ts: int, prev_dets: list[Detection] | None = None):
    """Build an aligned reading from synthetic simulators."""
    from fusion.data_types import AlignedReading

    dets = _moving_detections(frame_idx)
    frame_shape = (480, 640, 3)
    camera = CameraDetectionReading(sensor_id="cam_00", timestamp_ns=ts, detections=dets)
    lidar = LiDARSimulator(seed=42).generate(dets, frame_shape, ts)
    radar = RadarSimulator(seed=42).generate(dets, frame_shape, ts, prev_detections=prev_dets)
    return AlignedReading(camera=camera, lidar=lidar, radar=radar, alignment_gap_ms=0.0), dets


@pytest.mark.unit
def test_fusion_actor_enriches_confirmed_tracks(ray_cluster: None) -> None:
    """Confirmed tracks should gain LiDAR and radar modality data."""
    actor = FusionActor.remote(
        tracker=MultiObjectTracker(max_age=5, min_hits=2, iou_threshold=0.3),
        aligner=TemporalAligner(),
        projector=CameraLiDARProjector(),
    )

    prev: list[Detection] | None = None
    ts = time.time_ns()
    enriched = []
    for frame_idx in range(4):
        aligned, dets = _build_aligned(frame_idx, ts + frame_idx * 33_000_000, prev)
        enriched = ray.get(actor.process_aligned_reading.remote(aligned))
        prev = dets

    confirmed = [t for t in enriched if t.state == "confirmed"]
    assert len(confirmed) >= 1
    track = confirmed[0]
    assert "camera" in track.modalities_seen
    assert "lidar" in track.modalities_seen
    assert "radar" in track.modalities_seen
    assert track.last_bbox_3d is not None


@pytest.mark.unit
def test_fusion_actor_active_tracks_and_stats(ray_cluster: None) -> None:
    """FusionActor should expose active tracks and fusion statistics."""
    actor = FusionActor.remote(
        tracker=MultiObjectTracker(min_hits=1),
    )
    aligned, _ = _build_aligned(0, time.time_ns())
    ray.get(actor.process_aligned_reading.remote(aligned))

    active = ray.get(actor.get_active_tracks.remote())
    stats = ray.get(actor.get_fusion_stats.remote())

    assert len(active) >= 1
    assert stats.track_counts_by_state["tentative"] + stats.track_counts_by_state["confirmed"] >= 1
    assert "camera" in stats.modality_coverage_pct
    assert stats.mean_track_age >= 0.0
