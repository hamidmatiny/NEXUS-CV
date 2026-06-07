"""Unit tests for KalmanTracker and MultiObjectTracker."""

from __future__ import annotations

import numpy as np
import pytest

from fusion.kalman_tracker import KalmanTracker, MultiObjectTracker, _iou
from ingestion.yolo_detector import Detection


def _det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: str = "car",
) -> Detection:
    """Build a test Detection."""
    return Detection(
        bbox_xyxy=(x1, y1, x2, y2),
        confidence=0.9,
        class_id=2,
        class_name=class_name,
    )


@pytest.mark.unit
def test_kalman_predict_update_cycle() -> None:
    """Predict then update should move state toward the measurement."""
    initial = np.array([100.0, 100.0, 50.0, 50.0], dtype=np.float64)
    tracker = KalmanTracker(initial)

    predicted = tracker.predict()
    assert predicted.shape == (8,)
    assert predicted[0] == pytest.approx(100.0)
    assert predicted[4] == pytest.approx(0.0)

    measurement = np.array([110.0, 105.0, 52.0, 48.0], dtype=np.float64)
    updated = tracker.update(measurement)
    assert updated[0] > 100.0
    assert updated[1] > 100.0

    state = tracker.get_state()
    assert state.mean.shape == (8,)
    assert state.covariance.shape == (8, 8)


@pytest.mark.unit
def test_track_birth_and_confirmation() -> None:
    """Tracks should start tentative and promote to confirmed at min_hits."""
    mot = MultiObjectTracker(max_age=5, min_hits=3, iou_threshold=0.3)
    det = _det(10.0, 10.0, 60.0, 60.0)

    tracks = mot.update([det])
    assert len(tracks) == 1
    assert tracks[0].state == "tentative"

    for _ in range(2):
        slightly_moved = _det(12.0, 12.0, 62.0, 62.0)
        tracks = mot.update([slightly_moved])

    assert len(tracks) == 1
    assert tracks[0].state == "confirmed"


@pytest.mark.unit
def test_track_death_after_max_age() -> None:
    """Tracks without matches should die after max_age frames."""
    mot = MultiObjectTracker(max_age=2, min_hits=1, iou_threshold=0.3)
    mot.update([_det(10.0, 10.0, 60.0, 60.0)])

    tracks = mot.update([])
    assert len(tracks) == 1

    tracks = mot.update([])
    assert len(tracks) == 0


@pytest.mark.unit
def test_hungarian_assignment_iou_threshold() -> None:
    """Detections below IoU threshold should spawn new tracks."""
    mot = MultiObjectTracker(max_age=5, min_hits=1, iou_threshold=0.5)
    mot.update([_det(10.0, 10.0, 60.0, 60.0)])

    far_det = _det(200.0, 200.0, 260.0, 260.0)
    tracks = mot.update([far_det])
    assert len(tracks) == 2


@pytest.mark.unit
def test_iou_identical_boxes() -> None:
    """Identical boxes should have IoU of 1.0."""
    box = (0.0, 0.0, 100.0, 100.0)
    assert _iou(box, box) == pytest.approx(1.0)


@pytest.mark.unit
def test_iou_non_overlapping() -> None:
    """Non-overlapping boxes should have IoU of 0.0."""
    assert _iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


@pytest.mark.unit
def test_iou_partial_overlap() -> None:
    """Partially overlapping boxes should have IoU between 0 and 1."""
    iou = _iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0))
    assert 0.0 < iou < 1.0
