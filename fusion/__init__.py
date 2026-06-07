"""Fusion package for multi-modal sensor alignment and track fusion."""

from fusion.data_types import (
    AlignedReading,
    BBox3D,
    CameraDetectionReading,
    FusionStats,
    LiDARReading,
    RadarReading,
    RadarTarget,
    SensorReading,
    Track,
    TrackState,
)
from fusion.fusion_actor import FusionActor
from fusion.kalman_tracker import KalmanState, KalmanTracker, MultiObjectTracker
from fusion.lidar_simulator import LiDARSimulator
from fusion.radar_simulator import RadarSimulator
from fusion.sensor_alignment import CameraLiDARProjector, TemporalAligner

__all__ = [
    "AlignedReading",
    "BBox3D",
    "CameraDetectionReading",
    "CameraLiDARProjector",
    "FusionActor",
    "FusionStats",
    "KalmanState",
    "KalmanTracker",
    "LiDARReading",
    "LiDARSimulator",
    "MultiObjectTracker",
    "RadarReading",
    "RadarSimulator",
    "RadarTarget",
    "SensorReading",
    "TemporalAligner",
    "Track",
    "TrackState",
]
