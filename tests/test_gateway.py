"""Integration tests for the NEXUS-CV serving gateway."""

from __future__ import annotations

import base64
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from intelligence.data_types import ScenePrediction
from serving.deployments import LocalPipeline, reset_shared_pipeline
from serving.gateway import configure_pipeline, create_app
from serving.middleware import reset_circuit_breaker
from serving.schemas import InferenceRequest, PipelineResult


@pytest.fixture(autouse=True)
def _reset_serving_state() -> None:
    """Reset shared pipeline and circuit breaker between tests."""
    reset_shared_pipeline()
    reset_circuit_breaker()


@pytest.fixture
def encoded_frame(synthetic_frame: np.ndarray) -> str:
    """Encode a synthetic frame as base64 JPEG.

    Args:
        synthetic_frame: BGR numpy array.

    Returns:
        Base64-encoded JPEG string.
    """
    ok, buf = cv2.imencode(".jpg", synthetic_frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture
def mock_pipeline(
    mock_detections: list,
    mock_yolo_detector: MagicMock,
    synthetic_frame: np.ndarray,
) -> LocalPipeline:
    """Build a LocalPipeline with mocked detection and intelligence stages.

    Args:
        mock_detections: Fixture detections.
        mock_yolo_detector: Mock YOLO detector.
        synthetic_frame: Test frame array.

    Returns:
        LocalPipeline configured with mocks.
    """
    detection = MagicMock()

    async def _detect(request: InferenceRequest) -> PipelineResult:
        return PipelineResult(
            camera_id=request.camera_id,
            timestamp_ns=request.timestamp_ns,
            frame=synthetic_frame,
            detections=mock_detections,
            detection_ms=2.0,
        )

    detection.side_effect = _detect

    intelligence = MagicMock()
    scene = ScenePrediction(scene_class="highway", confidence=0.9, top3=[("highway", 0.9)])

    async def _intel(partial: PipelineResult) -> PipelineResult:
        partial.scene = scene
        partial.anomalies = []
        partial.trajectories = []
        partial.intelligence_ms = 3.0
        return partial

    intelligence.side_effect = _intel

    with patch("serving.deployments.YOLODetector", return_value=mock_yolo_detector):
        fusion = __import__("serving.deployments", fromlist=["FusionDeployment"]).FusionDeployment()
        pipeline = LocalPipeline(detection=detection, fusion=fusion, intelligence=intelligence)
    return pipeline


@pytest.fixture
async def gateway_client(mock_pipeline: LocalPipeline) -> AsyncClient:
    """Provide an httpx AsyncClient wired to the gateway app.

    Args:
        mock_pipeline: Mocked local pipeline.

    Yields:
        Configured AsyncClient instance.
    """
    configure_pipeline(mock_pipeline.remote)
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_endpoint(gateway_client: AsyncClient) -> None:
    """GET /health returns structured health response."""
    response = await gateway_client.get("/health")
    assert response.status_code in {200, 503}
    body = response.json()
    assert "status" in body
    assert "components" in body
    assert "uptime_s" in body
    assert body["version"] == "0.1.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metrics_endpoint(gateway_client: AsyncClient) -> None:
    """GET /metrics returns Prometheus text format."""
    response = await gateway_client.get("/metrics")
    assert response.status_code == 200
    assert "nexus_cv_serving_duration_ms" in response.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_infer_endpoint(
    gateway_client: AsyncClient,
    encoded_frame: str,
) -> None:
    """POST /api/v1/infer returns InferenceResponse JSON."""
    payload = {
        "camera_id": "cam_00",
        "frame_b64": encoded_frame,
        "timestamp_ns": time.time_ns(),
    }
    response = await gateway_client.post("/api/v1/infer", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["camera_id"] == "cam_00"
    assert "request_id" in body
    assert "detections" in body
    assert "scene" in body
    assert body["scene"]["scene_class"] == "highway"
    assert "X-Request-ID" in response.headers
    assert "X-Serving-Ms" in response.headers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_stream(
    mock_pipeline: LocalPipeline,
    synthetic_frame: np.ndarray,
) -> None:
    """WebSocket /ws/stream/{camera_id} accepts JPEG and returns JSON."""
    configure_pipeline(mock_pipeline.remote)
    app = create_app()
    ok, jpeg = cv2.imencode(".jpg", synthetic_frame)
    assert ok

    async with app.router.lifespan_context(app):
        from starlette.testclient import TestClient

        with TestClient(app) as client:
            with client.websocket_connect("/ws/stream/cam_00") as ws:
                ws.send_bytes(jpeg.tobytes())
                data = ws.receive_json()
                assert data["camera_id"] == "cam_00"
                assert "request_id" in data
                assert "scene" in data
