"""Unit tests for YOLODetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ingestion.yolo_detector import Detection, YOLODetector


def _make_mock_result(
    boxes_data: list[tuple[list[float], float, int, str]],
) -> MagicMock:
    """Build a mock Ultralytics result object.

    Args:
        boxes_data: List of (xyxy, conf, class_id, class_name) tuples.

    Returns:
        Mock result with boxes attribute.
    """
    result = MagicMock()
    result.names = {0: "person", 2: "car"}

    mock_boxes = []
    for xyxy, conf, class_id, _ in boxes_data:
        box = MagicMock()
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = xyxy
        box.cls = [MagicMock()]
        box.cls[0].item.return_value = class_id
        box.conf = [MagicMock()]
        box.conf[0].item.return_value = conf
        box.id = None
        mock_boxes.append(box)

    result.boxes = mock_boxes
    return result


@pytest.mark.unit
def test_detect_single_frame(synthetic_frame: np.ndarray) -> None:
    """Single-frame detect should return a list of Detection objects."""
    mock_result = _make_mock_result([([10.0, 20.0, 100.0, 200.0], 0.95, 0, "person")])

    with patch.object(YOLODetector, "_ensure_model") as mock_ensure:
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_ensure.return_value = mock_model

        detector = YOLODetector(model_path="fake.pt")
        detections = detector.detect(synthetic_frame)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.bbox_xyxy == (10.0, 20.0, 100.0, 200.0)
    assert det.confidence == 0.95
    assert det.class_id == 0
    assert det.class_name == "person"
    assert det.track_id is None


@pytest.mark.unit
def test_detect_batch_inference(synthetic_frame: np.ndarray) -> None:
    """Batch detect should return per-frame detection lists."""
    frames = [synthetic_frame, synthetic_frame.copy()]
    mock_results = [
        _make_mock_result([([10.0, 20.0, 50.0, 60.0], 0.9, 0, "person")]),
        _make_mock_result([([100.0, 100.0, 200.0, 200.0], 0.85, 2, "car")]),
    ]

    with patch.object(YOLODetector, "_ensure_model") as mock_ensure:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_ensure.return_value = mock_model

        detector = YOLODetector(model_path="fake.pt")
        batch_results = detector.detect_batch(frames)

    assert len(batch_results) == 2
    assert len(batch_results[0]) == 1
    assert len(batch_results[1]) == 1
    assert batch_results[0][0].class_name == "person"
    assert batch_results[1][0].class_name == "car"

    mock_model.predict.assert_called_once()
    call_kwargs = mock_model.predict.call_args
    assert call_kwargs[1]["conf"] == 0.45
    assert call_kwargs[1]["iou"] == 0.5


@pytest.mark.unit
def test_detect_batch_empty() -> None:
    """Empty batch should return empty list without model call."""
    with patch.object(YOLODetector, "_ensure_model") as mock_ensure:
        detector = YOLODetector(model_path="fake.pt")
        result = detector.detect_batch([])

    assert result == []
    mock_ensure.assert_not_called()


@pytest.mark.unit
def test_detection_output_schema() -> None:
    """Detection dataclass should expose required fields with correct types."""
    det = Detection(
        bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
        confidence=0.5,
        class_id=0,
        class_name="person",
        track_id=42,
    )
    assert len(det.bbox_xyxy) == 4
    assert all(isinstance(v, float) for v in det.bbox_xyxy)
    assert isinstance(det.confidence, float)
    assert isinstance(det.class_id, int)
    assert isinstance(det.class_name, str)
    assert det.track_id == 42
