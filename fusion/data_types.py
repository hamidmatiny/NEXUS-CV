"""Multi-modal sensor fusion data types."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ingestion.yolo_detector import Detection

Modality = Literal["camera", "lidar", "radar"]
TrackState = Literal["tentative", "confirmed", "lost", "dead"]


@dataclass(frozen=True, slots=True)
class BBox3D:
    """Axis-aligned or yaw-oriented 3D bounding box.

    Attributes:
        center_xyz: Center position (x, y, z) in meters.
        dimensions_lwh: Length, width, height in meters.
        yaw_rad: Heading angle around the vertical axis in radians.
        confidence: Fusion confidence score in [0, 1].
    """

    center_xyz: tuple[float, float, float]
    dimensions_lwh: tuple[float, float, float]
    yaw_rad: float
    confidence: float


@dataclass(frozen=True, slots=True)
class RadarTarget:
    """Single radar detection target.

    Attributes:
        range_m: Slant range in meters.
        azimuth_rad: Azimuth angle in radians.
        velocity_mps: Radial velocity in meters per second.
        rcs_dbsm: Radar cross-section in dBsm.
    """

    range_m: float
    azimuth_rad: float
    velocity_mps: float
    rcs_dbsm: float


@dataclass(frozen=True, slots=True)
class SensorReading:
    """Base sensor reading with temporal and modality metadata.

    Attributes:
        sensor_id: Unique sensor identifier.
        timestamp_ns: Reading timestamp in nanoseconds since epoch.
        modality: Sensor modality label.
    """

    sensor_id: str
    timestamp_ns: int
    modality: Modality


@dataclass(frozen=True, slots=True)
class CameraDetectionReading(SensorReading):
    """Camera frame with 2D object detections.

    Attributes:
        detections: List of YOLO detections from the camera frame.
    """

    detections: list[Detection]
    modality: Modality = field(default="camera", init=False)


@dataclass(frozen=True, slots=True)
class LiDARReading(SensorReading):
    """LiDAR scan with point cloud and cluster bounding boxes.

    Attributes:
        point_cloud: Point cloud array of shape (N, 4) with XYZI columns.
        cluster_bboxes_3d: 3D bounding boxes for detected clusters.
    """

    point_cloud: NDArray[np.float64]
    cluster_bboxes_3d: list[BBox3D]
    modality: Modality = field(default="lidar", init=False)


@dataclass(frozen=True, slots=True)
class RadarReading(SensorReading):
    """Radar scan with tracked targets.

    Attributes:
        targets: List of radar detection targets.
    """

    targets: list[RadarTarget]
    modality: Modality = field(default="radar", init=False)


@dataclass(slots=True)
class Track:
    """Fused multi-modal object track.

    Attributes:
        track_id: Unique track identifier (UUID string).
        state: Current track lifecycle state.
        age_frames: Number of frames since track birth.
        modalities_seen: Set of modalities that contributed to this track.
        last_bbox_2d: Most recent 2D bounding box (x1, y1, x2, y2).
        last_bbox_3d: Most recent fused 3D bounding box, if available.
        velocity_2d: Estimated 2D velocity (vx, vy) in pixels per frame.
        class_votes: Counter of class_name votes across detections.
        anomaly_score: Anomaly score (0 = normal, higher = anomalous).
    """

    track_id: str
    state: TrackState
    age_frames: int
    modalities_seen: set[str]
    last_bbox_2d: tuple[float, float, float, float] | None
    last_bbox_3d: BBox3D | None
    velocity_2d: tuple[float, float]
    class_votes: Counter[str]
    anomaly_score: float = 0.0


@dataclass(frozen=True, slots=True)
class AlignedReading:
    """Temporally aligned multi-modal sensor readings.

    Attributes:
        camera: Camera detection reading.
        lidar: LiDAR point cloud reading.
        radar: Radar target reading.
        alignment_gap_ms: Maximum inter-sensor timestamp gap in milliseconds.
    """

    camera: CameraDetectionReading
    lidar: LiDARReading
    radar: RadarReading
    alignment_gap_ms: float


@dataclass(frozen=True, slots=True)
class FusionStats:
    """Aggregate fusion pipeline statistics.

    Attributes:
        track_counts_by_state: Number of tracks per lifecycle state.
        modality_coverage_pct: Percentage of tracks per modality (0-100).
        mean_track_age: Mean track age in frames.
    """

    track_counts_by_state: dict[str, int]
    modality_coverage_pct: dict[str, float]
    mean_track_age: float
