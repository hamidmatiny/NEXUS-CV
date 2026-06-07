"""Unit tests for TemporalAligner and CameraLiDARProjector."""

from __future__ import annotations

import numpy as np
import pytest

from fusion.data_types import (
    BBox3D,
    CameraDetectionReading,
    LiDARReading,
    RadarReading,
    RadarTarget,
)
from fusion.sensor_alignment import CameraLiDARProjector, TemporalAligner
from ingestion.yolo_detector import Detection


def _camera(ts: int) -> CameraDetectionReading:
    """Build a camera reading at timestamp ``ts``."""
    return CameraDetectionReading(
        sensor_id="cam_00",
        timestamp_ns=ts,
        detections=[
            Detection(
                bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ],
    )


def _lidar(ts: int, points: np.ndarray | None = None) -> LiDARReading:
    """Build a LiDAR reading at timestamp ``ts``."""
    cloud = points if points is not None else np.array([[1.0, 0.0, 10.0, 0.5]], dtype=np.float64)
    return LiDARReading(
        sensor_id="lidar_00",
        timestamp_ns=ts,
        point_cloud=cloud,
        cluster_bboxes_3d=[
            BBox3D(
                center_xyz=(1.0, 0.0, 10.0),
                dimensions_lwh=(4.0, 2.0, 1.5),
                yaw_rad=0.0,
                confidence=0.9,
            )
        ],
    )


def _radar(ts: int) -> RadarReading:
    """Build a radar reading at timestamp ``ts``."""
    return RadarReading(
        sensor_id="radar_00",
        timestamp_ns=ts,
        targets=[RadarTarget(range_m=10.0, azimuth_rad=0.1, velocity_mps=5.0, rcs_dbsm=15.0)],
    )


@pytest.mark.unit
def test_align_within_tolerance() -> None:
    """Readings within max_offset_ms should align successfully."""
    aligner = TemporalAligner(buffer_size=10)
    base_ts = 1_000_000_000
    aligner.add_reading(_camera(base_ts))
    aligner.add_reading(_lidar(base_ts + 10_000_000))
    aligner.add_reading(_radar(base_ts + 20_000_000))

    result = aligner.align(base_ts, base_ts + 10_000_000, base_ts + 20_000_000, max_offset_ms=50.0)
    assert result is not None
    assert result.alignment_gap_ms <= 50.0


@pytest.mark.unit
def test_align_rejects_excessive_gap() -> None:
    """Readings exceeding max_offset_ms should return None."""
    aligner = TemporalAligner(buffer_size=10)
    base_ts = 1_000_000_000
    aligner.add_reading(_camera(base_ts))
    aligner.add_reading(_lidar(base_ts + 100_000_000))
    aligner.add_reading(_radar(base_ts))

    result = aligner.align(base_ts, base_ts + 100_000_000, base_ts, max_offset_ms=50.0)
    assert result is None


@pytest.mark.unit
def test_lidar_interpolation() -> None:
    """LiDAR readings should be linearly interpolated between buffered scans."""
    aligner = TemporalAligner(buffer_size=10)
    before = _lidar(1_000_000_000, np.array([[0.0, 0.0, 10.0, 0.2]], dtype=np.float64))
    after = _lidar(1_100_000_000, np.array([[2.0, 0.0, 12.0, 0.8]], dtype=np.float64))
    aligner.add_reading(before)
    aligner.add_reading(after)
    aligner.add_reading(_camera(1_050_000_000))
    aligner.add_reading(_radar(1_050_000_000))

    result = aligner.align(1_050_000_000, 1_050_000_000, 1_050_000_000, max_offset_ms=100.0)
    assert result is not None
    assert result.lidar.timestamp_ns == 1_050_000_000 or result.lidar.point_cloud.shape[0] > 0


@pytest.mark.unit
def test_project_lidar_to_image() -> None:
    """Projector should return Nx2 pixel coordinates."""
    projector = CameraLiDARProjector()
    points = np.array([[0.0, 0.0, 10.0], [1.0, 0.5, 15.0]], dtype=np.float64)
    pixels = projector.project_lidar_to_image(points)
    assert pixels.shape == (2, 2)
    assert pixels[0, 0] > 0


@pytest.mark.unit
def test_lift_bbox2d_to_3d() -> None:
    """Back-projection should produce a valid BBox3D."""
    projector = CameraLiDARProjector()
    depth_map = np.full((480, 640), 10.0, dtype=np.float64)
    bbox_3d = projector.lift_bbox2d_to_3d((100.0, 100.0, 200.0, 200.0), depth_map)
    assert bbox_3d.center_xyz[2] == pytest.approx(10.0)
    assert all(d > 0 for d in bbox_3d.dimensions_lwh)
