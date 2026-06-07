"""Unit tests for Ray Serve deployment pipeline stages."""

from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from intelligence.data_types import IntelligenceOutput, ScenePrediction
from serving.deployments import (
    DetectionDeployment,
    FusionDeployment,
    IntelligenceDeployment,
    LocalPipeline,
    decode_frame,
)
from serving.schemas import InferenceRequest, PipelineResult


@pytest.mark.unit
def test_decode_frame_roundtrip(synthetic_frame: np.ndarray) -> None:
    """decode_frame converts base64 JPEG back to BGR array."""
    ok, buf = cv2.imencode(".jpg", synthetic_frame)
    assert ok
    frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    decoded = decode_frame(frame_b64)
    assert decoded.shape == synthetic_frame.shape
    assert decoded.dtype == np.uint8


@pytest.mark.unit
@pytest.mark.asyncio
async def test_detection_deployment(
    mock_yolo_detector: MagicMock,
    synthetic_frame: np.ndarray,
    encoded_frame: str,
) -> None:
    """DetectionDeployment returns detections in PipelineResult."""
    with patch("serving.deployments.YOLODetector", return_value=mock_yolo_detector):
        deployment = DetectionDeployment()
    request = InferenceRequest(
        camera_id="cam_00",
        frame_b64=encoded_frame,
        timestamp_ns=time.time_ns(),
    )
    result = await deployment(request)
    assert result.camera_id == "cam_00"
    assert len(result.detections) == 2
    assert result.detection_ms >= 0.0
    assert result.frame is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fusion_deployment(
    mock_detections: list,
    synthetic_frame: np.ndarray,
) -> None:
    """FusionDeployment enriches partial result with tracks."""
    deployment = FusionDeployment()
    partial = PipelineResult(
        camera_id="cam_00",
        timestamp_ns=time.time_ns(),
        frame=synthetic_frame,
        detections=mock_detections,
        detection_ms=1.0,
    )
    result = await deployment(partial)
    assert result.fusion_ms >= 0.0
    assert isinstance(result.tracks, list)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_intelligence_deployment(
    mock_detections: list,
    synthetic_frame: np.ndarray,
) -> None:
    """IntelligenceDeployment populates scene and anomalies."""
    scene = ScenePrediction(scene_class="urban", confidence=0.85, top3=[("urban", 0.85)])
    output = IntelligenceOutput(
        frame_id=0,
        camera_id="cam_00",
        scene=scene,
        trajectories=[],
        anomalies=[],
        inference_total_ms=5.0,
    )
    mock_ensemble = MagicMock()

    async def _run_async(*_args: object, **_kwargs: object) -> IntelligenceOutput:
        return output

    mock_ensemble.run_async = _run_async

    with patch("serving.deployments.IntelligenceEnsemble", return_value=mock_ensemble):
        deployment = IntelligenceDeployment()

    partial = PipelineResult(
        camera_id="cam_00",
        timestamp_ns=time.time_ns(),
        frame=synthetic_frame,
        detections=mock_detections,
        tracks=[],
        detection_ms=1.0,
        fusion_ms=2.0,
    )
    result = await deployment(partial)
    assert result.scene is not None
    assert result.scene.scene_class == "urban"
    assert result.intelligence_ms >= 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_pipeline_chain(
    mock_yolo_detector: MagicMock,
    mock_detections: list,
    synthetic_frame: np.ndarray,
    encoded_frame: str,
) -> None:
    """LocalPipeline chains detection, fusion, and intelligence."""
    scene = ScenePrediction(scene_class="highway", confidence=0.9, top3=[("highway", 0.9)])
    mock_ensemble = MagicMock()

    async def _run_async(*_args: object, **_kwargs: object) -> IntelligenceOutput:
        return IntelligenceOutput(
            frame_id=0,
            camera_id="cam_00",
            scene=scene,
            trajectories=[],
            anomalies=[],
            inference_total_ms=4.0,
        )

    mock_ensemble.run_async = _run_async

    with (
        patch("serving.deployments.YOLODetector", return_value=mock_yolo_detector),
        patch("serving.deployments.IntelligenceEnsemble", return_value=mock_ensemble),
    ):
        pipeline = LocalPipeline()

    request = InferenceRequest(
        camera_id="cam_00",
        frame_b64=encoded_frame,
        timestamp_ns=time.time_ns(),
    )
    result = await pipeline.remote(request)
    assert len(result.detections) == 2
    assert result.scene is not None
    assert result.inference_ms >= 0.0


@pytest.fixture
def encoded_frame(synthetic_frame: np.ndarray) -> str:
    """Encode synthetic frame as base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", synthetic_frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")
