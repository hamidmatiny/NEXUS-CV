"""Tests for detection schema validation."""

from __future__ import annotations

import pytest

from ingestion.schema_contracts import validate_detections
from ingestion.yolo_detector import Detection


@pytest.mark.unit
def test_validate_valid_detections(mock_detections: list[Detection]) -> None:
    """Valid detections should pass schema validation."""
    result = validate_detections(mock_detections, camera_id="cam_00", timestamp_ns=1_000_000_000)
    assert result.is_valid is True
    assert result.passed == 2
    assert result.failed == 0
    assert result.quarantine_path is None


@pytest.mark.unit
def test_validate_invalid_confidence_quarantines(
    tmp_path: object,
    settings: object,
) -> None:
    """Invalid confidence should quarantine failed rows."""
    bad = Detection(
        bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
        confidence=1.5,
        class_id=0,
        class_name="person",
    )
    result = validate_detections([bad], camera_id="cam_bad", timestamp_ns=2_000_000_000)
    assert result.is_valid is False
    assert result.failed == 1
    assert result.quarantine_path is not None
    assert result.quarantine_path.exists()


@pytest.mark.unit
def test_validate_empty_detections() -> None:
    """Empty detection list should be valid with zero counts."""
    result = validate_detections([], camera_id="cam_empty")
    assert result.is_valid is True
    assert result.passed == 0
    assert result.failed == 0
