"""Synthetic LiDAR point cloud generator for development and testing."""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

from fusion.data_types import BBox3D, LiDARReading
from ingestion.yolo_detector import Detection

logger = structlog.get_logger(__name__)

DEFAULT_SENSOR_ID = "lidar_sim_00"
POINTS_PER_CLUSTER = 80
BACKGROUND_POINTS = 500


class LiDARSimulator:
    """Generates synthetic LiDAR readings consistent with camera detections."""

    def __init__(self, sensor_id: str = DEFAULT_SENSOR_ID, seed: int = 42) -> None:
        """Initialize the LiDAR simulator.

        Args:
            sensor_id: Identifier for generated readings.
            seed: Random seed for reproducibility.
        """
        self._sensor_id = sensor_id
        self._rng = np.random.default_rng(seed)

    def _detection_to_cluster(
        self,
        det: Detection,
        frame_shape: tuple[int, ...],
        fx: float = 554.0,
        fy: float = 554.0,
        cx: float = 320.0,
        cy: float = 240.0,
    ) -> tuple[NDArray[np.float64], BBox3D]:
        """Generate a point cluster and 3D bbox for a single detection.

        Args:
            det: Camera detection.
            frame_shape: Frame shape (H, W, C).
            fx: Focal length x for depth projection.
            fy: Focal length y for depth projection.
            cx: Principal point x.
            cy: Principal point y.

        Returns:
            Tuple of (cluster points Nx4, BBox3D).
        """
        x1, y1, x2, y2 = det.bbox_xyxy
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        depth = float(self._rng.normal(10.0, 2.0))
        depth = max(depth, 2.0)
        x_lidar = (u - cx) * depth / fx
        y_lidar = (v - cy) * depth / fy
        z_lidar = depth

        spread = max((x2 - x1) * depth / fx * 0.3, 0.3)
        cluster = self._rng.normal(
            loc=[x_lidar, y_lidar, z_lidar, 0.8],
            scale=[spread, spread * 0.5, spread * 0.3, 0.1],
            size=(POINTS_PER_CLUSTER, 4),
        )
        cluster[:, 3] = np.clip(cluster[:, 3], 0.0, 1.0)

        length = spread * 2.0
        width = spread * 1.2
        height = spread * 0.8
        bbox = BBox3D(
            center_xyz=(x_lidar, y_lidar, z_lidar),
            dimensions_lwh=(length, width, height),
            yaw_rad=0.0,
            confidence=det.confidence,
        )
        return cluster.astype(np.float64), bbox

    def _ground_plane(self, frame_shape: tuple[int, ...]) -> NDArray[np.float64]:
        """Generate background road/ground plane points.

        Args:
            frame_shape: Frame shape (H, W, C).

        Returns:
            Ground plane points array of shape (N, 4).
        """
        _ = frame_shape
        x = self._rng.uniform(-20.0, 20.0, BACKGROUND_POINTS)
        y = self._rng.uniform(-2.0, 2.0, BACKGROUND_POINTS)
        z = self._rng.uniform(8.0, 30.0, BACKGROUND_POINTS)
        intensity = self._rng.uniform(0.1, 0.3, BACKGROUND_POINTS)
        return np.column_stack([x, y, z, intensity]).astype(np.float64)

    def generate(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, ...],
        timestamp_ns: int,
    ) -> LiDARReading:
        """Generate a synthetic LiDAR reading from camera detections.

        Args:
            detections: Camera detections to synthesize clusters from.
            frame_shape: Camera frame shape (H, W, C).
            timestamp_ns: Timestamp matching the camera frame.

        Returns:
            Structured LiDARReading with point cloud and cluster bboxes.
        """
        clusters: list[NDArray[np.float64]] = [self._ground_plane(frame_shape)]
        bboxes: list[BBox3D] = []

        for det in detections:
            cluster, bbox = self._detection_to_cluster(det, frame_shape)
            clusters.append(cluster)
            bboxes.append(bbox)

        point_cloud = np.vstack(clusters) if clusters else np.empty((0, 4), dtype=np.float64)

        logger.debug(
            "lidar_simulated",
            num_points=len(point_cloud),
            num_clusters=len(bboxes),
            timestamp_ns=timestamp_ns,
        )

        return LiDARReading(
            sensor_id=self._sensor_id,
            timestamp_ns=timestamp_ns,
            point_cloud=point_cloud,
            cluster_bboxes_3d=bboxes,
        )
