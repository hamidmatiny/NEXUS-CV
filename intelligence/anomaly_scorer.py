"""Stateful anomaly scoring for fused tracks."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import structlog

from fusion.data_types import Track
from intelligence.data_types import AnomalyScore, ScenePrediction

logger = structlog.get_logger(__name__)

ANOMALY_THRESHOLD = 0.65
FACTOR_WEIGHT = 0.25
FPS = 30.0
METERS_PER_PIXEL = 0.05
HIGH_SPEED_KMH = 50.0
RESURRECTION_WINDOW = 5
NEAR_MISS_IOU = 0.15


@dataclass
class _SceneStats:
    """Rolling statistics for a scene class."""

    velocities: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    object_counts: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    bbox_areas: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def update(self, tracks: list[Track]) -> None:
        """Update rolling stats from current tracks.

        Args:
            tracks: Active confirmed tracks in the scene.
        """
        confirmed = [t for t in tracks if t.state == "confirmed"]
        self.object_counts.append(len(confirmed))
        for track in confirmed:
            speed = math.hypot(track.velocity_2d[0], track.velocity_2d[1])
            self.velocities.append(speed)
            if track.last_bbox_2d is not None:
                x1, y1, x2, y2 = track.last_bbox_2d
                self.bbox_areas.append(max((x2 - x1) * (y2 - y1), 1.0))

    def velocity_zscore(self, speed: float) -> float:
        """Compute z-score of a velocity against the rolling baseline.

        Args:
            speed: Track speed in pixels per frame.

        Returns:
            Absolute z-score.
        """
        if len(self.velocities) < 2:
            return 0.0
        arr = np.array(self.velocities, dtype=np.float64)
        std = float(arr.std())
        if std < 1e-6:
            return 0.0
        return abs((speed - float(arr.mean())) / std)


def _iou(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """Compute IoU between two axis-aligned boxes.

    Args:
        box_a: First box (x1, y1, x2, y2).
        box_b: Second box (x1, y1, x2, y2).

    Returns:
        Intersection-over-union score.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _speed_kmh(velocity: tuple[float, float]) -> float:
    """Convert pixel/frame velocity to km/h.

    Args:
        velocity: (vx, vy) in pixels per frame.

    Returns:
        Speed in km/h.
    """
    speed_mps = math.hypot(velocity[0], velocity[1]) * FPS * METERS_PER_PIXEL
    return speed_mps * 3.6


def _dominant_class(track: Track) -> str:
    """Return the most-voted class name for a track.

    Args:
        track: Fused track.

    Returns:
        Dominant class label or ``unknown``.
    """
    if not track.class_votes:
        return "unknown"
    return track.class_votes.most_common(1)[0][0]


class AnomalyScorer:
    """Scores track anomalies using scene-aware rolling statistics."""

    def __init__(self) -> None:
        """Initialize per-scene rolling statistics and death registry."""
        self._scene_stats: dict[str, _SceneStats] = defaultdict(_SceneStats)
        self._dead_tracks: dict[str, int] = {}
        self._frame_counter = 0

    def _check_velocity_anomaly(self, track: Track, scene: ScenePrediction) -> str | None:
        """Check for velocity z-score exceeding 3 sigma.

        Args:
            track: Track to evaluate.
            scene: Current scene prediction.

        Returns:
            Factor description or None.
        """
        stats = self._scene_stats[scene.scene_class]
        speed = math.hypot(track.velocity_2d[0], track.velocity_2d[1])
        z = stats.velocity_zscore(speed)
        if z > 3.0:
            return f"velocity_anomaly:z={z:.1f}"
        return None

    def _check_wrong_class_speed(self, track: Track) -> str | None:
        """Check for person-class tracks at highway speeds.

        Args:
            track: Track to evaluate.

        Returns:
            Factor description or None.
        """
        if _dominant_class(track) == "person" and _speed_kmh(track.velocity_2d) > HIGH_SPEED_KMH:
            return f"wrong_class_speed:{_speed_kmh(track.velocity_2d):.0f}km/h"
        return None

    def _check_near_miss(self, track: Track, all_tracks: list[Track]) -> str | None:
        """Check for near-miss geometry with a different-class track.

        Args:
            track: Track to evaluate.
            all_tracks: All active tracks.

        Returns:
            Factor description or None.
        """
        if track.last_bbox_2d is None or track.state != "confirmed":
            return None
        my_class = _dominant_class(track)
        for other in all_tracks:
            if other.track_id == track.track_id or other.state != "confirmed":
                continue
            if other.last_bbox_2d is None:
                continue
            if _dominant_class(other) == my_class:
                continue
            if _iou(track.last_bbox_2d, other.last_bbox_2d) > NEAR_MISS_IOU:
                return f"near_miss:iou={_iou(track.last_bbox_2d, other.last_bbox_2d):.2f}"
        return None

    def _check_resurrection(self, track: Track) -> str | None:
        """Check for track flickering (death and rebirth within 5 frames).

        Args:
            track: Track to evaluate.

        Returns:
            Factor description or None.
        """
        if track.age_frames > 1:
            return None
        for dead_id, death_frame in self._dead_tracks.items():
            if (
                self._frame_counter - death_frame <= RESURRECTION_WINDOW
                and dead_id != track.track_id
            ):
                return f"track_resurrection:rebirthed_within_{RESURRECTION_WINDOW}_frames"
        return None

    def register_dead_track(self, track_id: str) -> None:
        """Register a track death for resurrection detection.

        Args:
            track_id: Identifier of the dead track.
        """
        self._dead_tracks[track_id] = self._frame_counter

    def advance_frame(self) -> None:
        """Advance the internal frame counter."""
        self._frame_counter += 1

    def score(self, track: Track, scene: ScenePrediction, all_tracks: list[Track]) -> AnomalyScore:
        """Compute an anomaly score for a single track.

        Args:
            track: Track to score.
            scene: Current scene classification.
            all_tracks: All active tracks for contextual checks.

        Returns:
            AnomalyScore with contributing factors.
        """
        self._scene_stats[scene.scene_class].update(all_tracks)

        factors: list[str] = []
        checks = [
            self._check_velocity_anomaly(track, scene),
            self._check_wrong_class_speed(track),
            self._check_near_miss(track, all_tracks),
            self._check_resurrection(track),
        ]
        for factor in checks:
            if factor is not None:
                factors.append(factor)

        raw_score = min(len(factors) * FACTOR_WEIGHT, 1.0)
        is_anomalous = raw_score >= ANOMALY_THRESHOLD

        if is_anomalous:
            logger.info(
                "anomaly_detected",
                track_id=track.track_id,
                score=raw_score,
                factors=factors,
            )

        return AnomalyScore(
            track_id=track.track_id,
            score=raw_score,
            contributing_factors=factors,
            is_anomalous=is_anomalous,
        )
