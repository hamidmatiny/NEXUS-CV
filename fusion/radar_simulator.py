"""Synthetic radar target generator for development and testing."""

from __future__ import annotations

import math

import numpy as np
import structlog

from fusion.data_types import RadarReading, RadarTarget
from ingestion.yolo_detector import Detection

logger = structlog.get_logger(__name__)

DEFAULT_SENSOR_ID = "radar_sim_00"

RCS_BY_CLASS: dict[str, float] = {
    "car": 15.0,
    "person": 0.0,
    "truck": 25.0,
    "bus": 20.0,
    "motorcycle": 5.0,
    "bicycle": 2.0,
}

DEFAULT_RCS_DBSM = 5.0
DEFAULT_FPS = 30.0


class RadarSimulator:
    """Generates synthetic radar returns consistent with camera detections."""

    def __init__(self, sensor_id: str = DEFAULT_SENSOR_ID, seed: int = 42) -> None:
        """Initialize the radar simulator.

        Args:
            sensor_id: Identifier for generated readings.
            seed: Random seed for reproducibility.
        """
        self._sensor_id = sensor_id
        self._rng = np.random.default_rng(seed)

    def _rcs_for_class(self, class_name: str) -> float:
        """Return radar cross-section for a detection class.

        Args:
            class_name: COCO class name.

        Returns:
            RCS in dBsm.
        """
        return RCS_BY_CLASS.get(class_name, DEFAULT_RCS_DBSM)

    def _bbox_center(self, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        """Compute bbox center coordinates.

        Args:
            bbox: Bounding box (x1, y1, x2, y2).

        Returns:
            Center (cx, cy).
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _velocity_from_displacement(
        self,
        curr: Detection,
        prev: Detection | None,
        fps: float = DEFAULT_FPS,
    ) -> float:
        """Estimate radial velocity from cross-frame bbox displacement.

        Args:
            curr: Current frame detection.
            prev: Previous frame detection matched by proximity.
            fps: Assumed frame rate for time conversion.

        Returns:
            Estimated velocity in m/s with gaussian noise applied.
        """
        if prev is None:
            base_velocity = 0.0
        else:
            cx0, cy0 = self._bbox_center(prev.bbox_xyxy)
            cx1, cy1 = self._bbox_center(curr.bbox_xyxy)
            displacement_px = math.hypot(cx1 - cx0, cy1 - cy0)
            base_velocity = displacement_px / fps * 0.05

        noise = float(self._rng.normal(0.0, 0.5))
        return base_velocity + noise

    def _match_prev(
        self,
        det: Detection,
        prev_detections: list[Detection],
    ) -> Detection | None:
        """Find the nearest previous detection to ``det`` by bbox center distance.

        Args:
            det: Current detection.
            prev_detections: Previous frame detections.

        Returns:
            Matched previous detection or None.
        """
        if not prev_detections:
            return None
        cx, cy = self._bbox_center(det.bbox_xyxy)
        best: Detection | None = None
        best_dist = float("inf")
        for prev in prev_detections:
            px, py = self._bbox_center(prev.bbox_xyxy)
            dist = math.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best = prev
        return best if best_dist < 100.0 else None

    def generate(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, ...],
        timestamp_ns: int,
        prev_detections: list[Detection] | None = None,
        fps: float = DEFAULT_FPS,
    ) -> RadarReading:
        """Generate synthetic radar targets from camera detections.

        Args:
            detections: Current frame camera detections.
            frame_shape: Camera frame shape (H, W, C).
            timestamp_ns: Timestamp matching the camera frame.
            prev_detections: Previous frame detections for velocity estimation.
            fps: Assumed frame rate for velocity computation.

        Returns:
            Structured RadarReading with synthetic targets.
        """
        _ = frame_shape
        prev = prev_detections or []
        targets: list[RadarTarget] = []

        for det in detections:
            cx, _ = self._bbox_center(det.bbox_xyxy)
            azimuth = math.atan2(cx - 320.0, 554.0)
            range_m = float(self._rng.normal(10.0, 2.0))
            range_m = max(range_m, 2.0)
            matched_prev = self._match_prev(det, prev)
            velocity = self._velocity_from_displacement(det, matched_prev, fps)
            rcs = self._rcs_for_class(det.class_name)

            targets.append(
                RadarTarget(
                    range_m=range_m,
                    azimuth_rad=azimuth,
                    velocity_mps=velocity,
                    rcs_dbsm=rcs,
                )
            )

        logger.debug(
            "radar_simulated",
            num_targets=len(targets),
            timestamp_ns=timestamp_ns,
        )

        return RadarReading(
            sensor_id=self._sensor_id,
            timestamp_ns=timestamp_ns,
            targets=targets,
        )
