"""Temporal sensor alignment and camera-LiDAR geometric projection."""

from __future__ import annotations

from collections import deque
from typing import TypeVar, cast

import numpy as np
import structlog
from numpy.typing import NDArray

from config.settings import get_settings
from fusion.calibration import default_extrinsic_matrix, default_intrinsic_matrix
from fusion.data_types import (
    AlignedReading,
    BBox3D,
    CameraDetectionReading,
    LiDARReading,
    Modality,
    RadarReading,
    SensorReading,
)

logger = structlog.get_logger(__name__)

ReadingT = TypeVar("ReadingT", bound=SensorReading)


class TemporalAligner:
    """Aligns multi-modal sensor readings within a temporal tolerance window."""

    def __init__(self, buffer_size: int | None = None) -> None:
        """Initialize per-modality sliding buffers.

        Args:
            buffer_size: Max readings per modality. Defaults to settings value.
        """
        settings = get_settings()
        size = buffer_size or settings.FUSION_ALIGNMENT_BUFFER_SIZE
        self._buffer_size = size
        self._buffers: dict[Modality, deque[SensorReading]] = {
            "camera": deque(maxlen=size),
            "lidar": deque(maxlen=size),
            "radar": deque(maxlen=size),
        }
        self._alignment_gaps_ms: list[float] = []

    def add_reading(self, reading: SensorReading) -> None:
        """Add a sensor reading to the modality buffer.

        Args:
            reading: Sensor reading to buffer.
        """
        self._buffers[reading.modality].append(reading)

    def _nearest_reading(
        self,
        modality: Modality,
        target_ts: int,
    ) -> SensorReading | None:
        """Find the reading nearest to a target timestamp.

        Args:
            modality: Sensor modality buffer to search.
            target_ts: Target timestamp in nanoseconds.

        Returns:
            Nearest reading or None if buffer is empty.
        """
        buf = self._buffers[modality]
        if not buf:
            return None
        return min(buf, key=lambda r: abs(r.timestamp_ns - target_ts))

    def _interpolate_lidar(
        self,
        before: LiDARReading,
        after: LiDARReading,
        target_ts: int,
    ) -> LiDARReading:
        """Linearly interpolate a LiDAR reading at ``target_ts``.

        Args:
            before: Earlier LiDAR reading.
            after: Later LiDAR reading.
            target_ts: Target timestamp in nanoseconds.

        Returns:
            Interpolated LiDARReading.
        """
        dt = after.timestamp_ns - before.timestamp_ns
        if dt <= 0:
            return before

        alpha = (target_ts - before.timestamp_ns) / dt
        alpha = float(np.clip(alpha, 0.0, 1.0))

        if before.point_cloud.shape == after.point_cloud.shape:
            interp_cloud = (1.0 - alpha) * before.point_cloud + alpha * after.point_cloud
        else:
            n = min(len(before.point_cloud), len(after.point_cloud))
            interp_cloud = (1.0 - alpha) * before.point_cloud[:n] + alpha * after.point_cloud[:n]

        merged_bboxes = before.cluster_bboxes_3d if alpha < 0.5 else after.cluster_bboxes_3d
        return LiDARReading(
            sensor_id=before.sensor_id,
            timestamp_ns=target_ts,
            point_cloud=interp_cloud,
            cluster_bboxes_3d=merged_bboxes,
        )

    def _get_lidar_at(
        self,
        target_ts: int,
        max_offset_ns: int,
    ) -> LiDARReading | None:
        """Retrieve or interpolate a LiDAR reading at ``target_ts``.

        Args:
            target_ts: Target timestamp in nanoseconds.
            max_offset_ns: Maximum allowed offset in nanoseconds.

        Returns:
            LiDARReading or None if unavailable within tolerance.
        """
        buf = sorted(self._buffers["lidar"], key=lambda r: r.timestamp_ns)
        if not buf:
            return None

        nearest = min(buf, key=lambda r: abs(r.timestamp_ns - target_ts))
        if abs(nearest.timestamp_ns - target_ts) <= max_offset_ns:
            return cast(LiDARReading, nearest)

        before: LiDARReading | None = None
        after: LiDARReading | None = None
        for reading in buf:
            lidar = cast(LiDARReading, reading)
            if lidar.timestamp_ns <= target_ts:
                before = lidar
            elif after is None:
                after = lidar
                break

        if before is not None and after is not None:
            gap_before = target_ts - before.timestamp_ns
            gap_after = after.timestamp_ns - target_ts
            if max(gap_before, gap_after) <= max_offset_ns:
                return self._interpolate_lidar(before, after, target_ts)

        if abs(nearest.timestamp_ns - target_ts) <= max_offset_ns:
            return cast(LiDARReading, nearest)
        return None

    def align(
        self,
        camera_ts: int,
        lidar_ts: int,
        radar_ts: int,
        max_offset_ms: float = 50.0,
    ) -> AlignedReading | None:
        """Align readings nearest to the given reference timestamps.

        Args:
            camera_ts: Reference camera timestamp (nanoseconds).
            lidar_ts: Reference LiDAR timestamp (nanoseconds).
            radar_ts: Reference radar timestamp (nanoseconds).
            max_offset_ms: Maximum allowed inter-sensor gap in milliseconds.

        Returns:
            AlignedReading if all modalities are within tolerance, else None.
        """
        max_offset_ns = int(max_offset_ms * 1_000_000)

        camera_reading = self._nearest_reading("camera", camera_ts)
        radar_reading = self._nearest_reading("radar", radar_ts)
        lidar_reading = self._get_lidar_at(lidar_ts, max_offset_ns)

        if camera_reading is None or lidar_reading is None or radar_reading is None:
            return None

        timestamps = [
            camera_reading.timestamp_ns,
            lidar_reading.timestamp_ns,
            radar_reading.timestamp_ns,
        ]
        gap_ns = max(timestamps) - min(timestamps)
        gap_ms = gap_ns / 1_000_000.0

        if gap_ms > max_offset_ms:
            return None

        self._alignment_gaps_ms.append(gap_ms)
        if len(self._alignment_gaps_ms) % 50 == 0:
            gaps = np.array(self._alignment_gaps_ms, dtype=np.float64)
            logger.info(
                "alignment_gap_stats",
                mean_ms=round(float(gaps.mean()), 3),
                p99_ms=round(float(np.percentile(gaps, 99)), 3),
                samples=len(gaps),
            )

        return AlignedReading(
            camera=cast(CameraDetectionReading, camera_reading),
            lidar=lidar_reading,
            radar=cast(RadarReading, radar_reading),
            alignment_gap_ms=gap_ms,
        )


class CameraLiDARProjector:
    """Projects LiDAR points to image space and lifts 2D boxes to 3D."""

    def __init__(
        self,
        intrinsic_matrix: NDArray[np.float64] | None = None,
        extrinsic_matrix: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize with camera calibration matrices.

        Args:
            intrinsic_matrix: 3x3 pinhole intrinsic matrix K.
            extrinsic_matrix: 4x4 LiDAR-to-camera extrinsic transform.
        """
        self._K = intrinsic_matrix if intrinsic_matrix is not None else default_intrinsic_matrix()
        self._E = extrinsic_matrix if extrinsic_matrix is not None else default_extrinsic_matrix()

    def project_lidar_to_image(self, points_3d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project 3D LiDAR points to 2D pixel coordinates.

        Args:
            points_3d: Array of shape (N, 3) or (N, 4) with XYZ(I) columns.

        Returns:
            Pixel coordinates array of shape (N, 2).
        """
        xyz = points_3d[:, :3]
        ones = np.ones((len(xyz), 1), dtype=np.float64)
        hom = np.hstack([xyz, ones])
        cam = (self._E @ hom.T).T[:, :3]
        valid = cam[:, 2] > 0.01
        pixels = np.zeros((len(xyz), 2), dtype=np.float64)
        if not np.any(valid):
            return pixels

        proj = (self._K @ cam[valid].T).T
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]
        pixels[valid, 0] = proj[:, 0]
        pixels[valid, 1] = proj[:, 1]
        return pixels

    def lift_bbox2d_to_3d(
        self,
        bbox_2d: tuple[float, float, float, float],
        depth_map: NDArray[np.float64],
    ) -> BBox3D:
        """Back-project a 2D bounding box to 3D using a LiDAR depth map.

        Args:
            bbox_2d: 2D box (x1, y1, x2, y2) in pixels.
            depth_map: Per-pixel depth array aligned with the camera image.

        Returns:
            Estimated 3D bounding box.
        """
        x1, y1, x2, y2 = bbox_2d
        h, w = depth_map.shape[:2]
        cx = int(np.clip((x1 + x2) / 2.0, 0, w - 1))
        cy = int(np.clip((y1 + y2) / 2.0, 0, h - 1))
        depth = float(depth_map[cy, cx]) if depth_map[cy, cx] > 0 else 10.0

        fx = self._K[0, 0]
        fy = self._K[1, 1]
        px = self._K[0, 2]
        py = self._K[1, 2]

        x_cam = (cx - px) * depth / fx
        y_cam = (cy - py) * depth / fy
        z_cam = depth

        box_w_px = max(x2 - x1, 1.0)
        box_h_px = max(y2 - y1, 1.0)
        width_m = box_w_px * depth / fx
        height_m = box_h_px * depth / fy
        length_m = max(width_m, height_m) * 1.5

        return BBox3D(
            center_xyz=(float(x_cam), float(y_cam), float(z_cam)),
            dimensions_lwh=(length_m, width_m, height_m),
            yaw_rad=0.0,
            confidence=0.8,
        )
