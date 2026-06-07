"""Pytest fixtures for NEXUS-CV unit and integration tests."""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray

from config.settings import Settings, get_settings
from ingestion.stream_capture import FramePacket
from ingestion.yolo_detector import Detection, YOLODetector

if TYPE_CHECKING:
    pass


@pytest.fixture
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Provide test settings with isolated quarantine directory.

    Args:
        tmp_path_factory: Pytest temporary path factory.

    Returns:
        Settings instance configured for testing.
    """
    quarantine = tmp_path_factory.mktemp("quarantine")
    return Settings(
        NUM_CAMERAS=2,
        FRAME_BUFFER_SIZE=5,
        YOLO_MODEL_PATH="yolo11n.pt",
        YOLO_CONFIDENCE_THRESHOLD=0.45,
        YOLO_IOU_THRESHOLD=0.5,
        QUARANTINE_DIR=quarantine,
        LOG_LEVEL="DEBUG",
        RAY_NUM_CPUS=2,
        RAY_NUM_GPUS=0.0,
    )


@pytest.fixture(autouse=True)
def _override_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Override cached settings singleton for each test.

    Args:
        settings: Test settings fixture.
        monkeypatch: Pytest monkeypatch fixture.
    """
    get_settings.cache_clear()
    monkeypatch.setattr("config.settings.get_settings", lambda: settings)
    monkeypatch.setattr("ingestion.frame_buffer_actor.get_settings", lambda: settings)
    monkeypatch.setattr("ingestion.yolo_detector.get_settings", lambda: settings)
    monkeypatch.setattr("ingestion.schema_contracts.get_settings", lambda: settings)
    monkeypatch.setattr("serving.health.get_settings", lambda: settings)
    monkeypatch.setattr("mlops.mlflow_utils.wait_for_mlflow", lambda *a, **k: True)
    monkeypatch.setattr("serving.mlops_background.wait_for_mlflow", lambda *a, **k: True)


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Create a 640x480 BGR test frame.

    Returns:
        Random BGR numpy array.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_frame_packet(synthetic_frame: np.ndarray) -> FramePacket:
    """Create a synthetic FramePacket for testing.

    Args:
        synthetic_frame: Base frame array.

    Returns:
        FramePacket with test metadata.
    """
    return FramePacket(
        camera_id="cam_00",
        frame_id=1,
        timestamp_ns=time.time_ns(),
        frame=synthetic_frame,
        source_uri="synthetic://test",
    )


@pytest.fixture
def mock_detections() -> list[Detection]:
    """Return a list of valid mock detections.

    Returns:
        Two Detection instances with valid schema values.
    """
    return [
        Detection(
            bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
            confidence=0.92,
            class_id=0,
            class_name="person",
        ),
        Detection(
            bbox_xyxy=(150.0, 50.0, 300.0, 250.0),
            confidence=0.78,
            class_id=2,
            class_name="car",
        ),
    ]


@pytest.fixture
def mock_yolo_detector(mock_detections: list[Detection]) -> MagicMock:
    """Provide a mock YOLODetector returning random detections.

    Args:
        mock_detections: Detections to return from mock.

    Returns:
        MagicMock configured as YOLODetector.
    """
    detector = MagicMock(spec=YOLODetector)
    detector.detect.return_value = mock_detections
    detector.detect_batch.return_value = [mock_detections]
    return detector


@pytest.fixture(scope="session")
def ray_cluster() -> Generator[None, None, None]:
    """Start a local Ray cluster for the test session.

    Yields:
        None after Ray initialization.
    """
    if not ray.is_initialized():
        ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True, logging_level="ERROR")
    yield
    if ray.is_initialized():
        ray.shutdown()
