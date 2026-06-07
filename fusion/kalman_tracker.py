"""Constant-velocity Kalman filter and multi-object tracker."""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from fusion.data_types import Track, TrackState
from ingestion.yolo_detector import Detection

logger = structlog.get_logger(__name__)

STATE_DIM = 8
MEAS_DIM = 4


@dataclass(frozen=True, slots=True)
class KalmanState:
    """Kalman filter state snapshot.

    Attributes:
        mean: State mean vector [cx, cy, w, h, vcx, vcy, vw, vh].
        covariance: State covariance matrix (8x8).
    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]


def _bbox_to_measurement(bbox: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Convert xyxy bbox to [cx, cy, w, h] measurement vector.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2).

    Returns:
        Measurement vector of shape (4,).
    """
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=np.float64)


def _measurement_to_bbox(meas: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Convert [cx, cy, w, h] measurement to xyxy bbox.

    Args:
        meas: Measurement vector.

    Returns:
        Bounding box as (x1, y1, x2, y2).
    """
    cx, cy, w, h = meas[:4]
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute intersection-over-union between two axis-aligned boxes.

    Args:
        box_a: First box (x1, y1, x2, y2).
        box_b: Second box (x1, y1, x2, y2).

    Returns:
        IoU score in [0, 1].
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


class KalmanTracker:
    """Constant-velocity Kalman filter in image space.

    State vector: [cx, cy, w, h, vcx, vcy, vw, vh].
    """

    def __init__(self, initial_measurement: NDArray[np.float64]) -> None:
        """Initialize the filter from a bbox measurement.

        Args:
            initial_measurement: [cx, cy, w, h] vector.
        """
        self._mean = np.zeros(STATE_DIM, dtype=np.float64)
        self._mean[:MEAS_DIM] = initial_measurement
        self._covariance = np.eye(STATE_DIM, dtype=np.float64)
        self._covariance[:MEAS_DIM, :MEAS_DIM] *= 10.0
        self._covariance[MEAS_DIM:, MEAS_DIM:] *= 1000.0

        self._F = np.eye(STATE_DIM, dtype=np.float64)
        for i in range(MEAS_DIM):
            self._F[i, MEAS_DIM + i] = 1.0

        self._H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
        for i in range(MEAS_DIM):
            self._H[i, i] = 1.0

        self._Q = np.eye(STATE_DIM, dtype=np.float64) * 0.01
        self._R = np.eye(MEAS_DIM, dtype=np.float64) * 1.0

    def predict(self) -> NDArray[np.float64]:
        """Advance the state by one time step.

        Returns:
            Predicted state mean vector.
        """
        self._mean = self._F @ self._mean
        self._covariance = self._F @ self._covariance @ self._F.T + self._Q
        return self._mean.copy()

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Incorporate a new bbox measurement.

        Args:
            measurement: [cx, cy, w, h] vector.

        Returns:
            Updated state mean vector.
        """
        innovation = measurement - self._H @ self._mean
        cov_innovation = self._H @ self._covariance @ self._H.T + self._R
        kalman_gain = self._covariance @ self._H.T @ np.linalg.inv(cov_innovation)
        self._mean = self._mean + kalman_gain @ innovation
        identity = np.eye(STATE_DIM, dtype=np.float64)
        self._covariance = (identity - kalman_gain @ self._H) @ self._covariance
        return self._mean.copy()

    def get_state(self) -> KalmanState:
        """Return the current filter state.

        Returns:
            KalmanState with mean and covariance.
        """
        return KalmanState(mean=self._mean.copy(), covariance=self._covariance.copy())


@dataclass
class _InternalTrack:
    """Internal tracker state for MultiObjectTracker."""

    track_id: str
    kalman: KalmanTracker
    hits: int
    age: int
    time_since_update: int
    state: TrackState
    class_votes: Counter[str]
    last_detection: Detection | None = None

    def to_track(self) -> Track:
        """Convert internal state to a public Track object.

        Returns:
            Track dataclass instance.
        """
        mean = self.kalman.get_state().mean
        bbox = _measurement_to_bbox(mean)
        return Track(
            track_id=self.track_id,
            state=self.state,
            age_frames=self.age,
            modalities_seen={"camera"},
            last_bbox_2d=bbox,
            last_bbox_3d=None,
            velocity_2d=(float(mean[4]), float(mean[5])),
            class_votes=Counter(self.class_votes),
            anomaly_score=0.0,
        )


class MultiObjectTracker:
    """Multi-object tracker with Hungarian assignment and Kalman filtering."""

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """Initialize the multi-object tracker.

        Args:
            max_age: Maximum frames without a match before track death.
            min_hits: Hits required to promote tentative → confirmed.
            iou_threshold: Minimum IoU for detection-track assignment.
        """
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._tracks: list[_InternalTrack] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections.

        Args:
            detections: Current frame detections.

        Returns:
            List of tentative and confirmed tracks (excludes dead tracks).
        """
        for track in self._tracks:
            track.kalman.predict()
            track.age += 1
            track.time_since_update += 1

        predicted_bboxes = [_measurement_to_bbox(t.kalman.get_state().mean) for t in self._tracks]

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if self._tracks and detections:
            cost = np.zeros((len(self._tracks), len(detections)), dtype=np.float64)
            for i, pred_bbox in enumerate(predicted_bboxes):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - _iou(pred_bbox, det.bbox_xyxy)

            row_ind, col_ind = linear_sum_assignment(cost)
            for row, col in zip(row_ind, col_ind, strict=True):
                if 1.0 - cost[row, col] >= self._iou_threshold:
                    matched_tracks.add(row)
                    matched_dets.add(col)
                    track = self._tracks[row]
                    det = detections[col]
                    meas = _bbox_to_measurement(det.bbox_xyxy)
                    track.kalman.update(meas)
                    track.hits += 1
                    track.time_since_update = 0
                    track.last_detection = det
                    track.class_votes[det.class_name] += 1
                    if track.hits >= self._min_hits:
                        track.state = "confirmed"
                    else:
                        track.state = "tentative"

        for j, det in enumerate(detections):
            if j not in matched_dets:
                meas = _bbox_to_measurement(det.bbox_xyxy)
                new_track = _InternalTrack(
                    track_id=str(uuid.uuid4()),
                    kalman=KalmanTracker(meas),
                    hits=1,
                    age=1,
                    time_since_update=0,
                    state="tentative",
                    class_votes=Counter({det.class_name: 1}),
                    last_detection=det,
                )
                self._tracks.append(new_track)

        surviving: list[_InternalTrack] = []
        for i, track in enumerate(self._tracks):
            if i not in matched_tracks:
                if track.time_since_update > 0 and track.state == "confirmed":
                    track.state = "lost"
                if track.time_since_update >= self._max_age:
                    track.state = "dead"
                    continue
            if track.state != "dead":
                surviving.append(track)

        self._tracks = surviving
        return [t.to_track() for t in self._tracks if t.state in ("tentative", "confirmed")]
